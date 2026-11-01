"""Traffic / cycle model of a tiled matmul on an accelerator with a software-managed scratchpad.

Loop nest (output-stationary, C tile lives in the PSUM-like accumulator)::

    for i in M/tm:            # C row block
      for j in N/tn:          # C col block
        for k in K/tk:        # DMA A[i,k] (tm x tk) and B[k,j] (tk x tn) into SBUF
          PSUM[i,j] += A[i,k] @ B[k,j]
        DMA PSUM[i,j] -> HBM

Every A element is fetched once per column block and every B element once per row block,
so HBM bytes = b * (M*K*ceil(N/tn) + K*N*ceil(M/tm) + M*N), exact even for ragged edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass(frozen=True)
class Accel:
    """Model parameters. Defaults are a Trn1-NeuronCore-like *assumption*, not a spec sheet."""

    partitions: int = 128
    sbuf_bytes: int = 24 * 2**20
    psum_bytes: int = 2 * 2**20
    pe_rows: int = 128
    pe_cols: int = 128
    moving_max: int = 512  # free-dim columns streamed per matmul instruction
    freq_hz: float = 1.4e9
    hbm_bw: float = 410e9
    dma_overhead_s: float = 1e-6  # fixed cost per DMA descriptor
    bytes_per_el: int = 2
    acc_bytes: int = 4  # fp32 accumulation in PSUM
    sbuf_bufs: int = 2  # double buffering of A/B tiles

    @property
    def peak_flops(self) -> float:
        return 2.0 * self.pe_rows * self.pe_cols * self.freq_hz

    @property
    def ridge(self) -> float:
        return self.peak_flops / self.hbm_bw


@dataclass(frozen=True)
class TileStats:
    tm: int
    tn: int
    tk: int
    hbm_bytes: int
    dma_count: int
    flops: int
    intensity: float
    sbuf_footprint: int
    psum_footprint: int
    compute_s: float
    dma_s: float

    @property
    def time_s(self) -> float:
        """Double-buffered: DMA and compute overlap, the slower one binds."""
        return max(self.compute_s, self.dma_s)

    @property
    def bound(self) -> str:
        return "compute" if self.compute_s >= self.dma_s else "memory"


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def sbuf_footprint(tm: int, tn: int, tk: int, acc: Accel) -> int:
    return acc.sbuf_bufs * acc.bytes_per_el * (tm * tk + tk * tn)


def psum_footprint(tm: int, tn: int, acc: Accel) -> int:
    return acc.acc_bytes * tm * tn


def fits(tm: int, tn: int, tk: int, acc: Accel) -> bool:
    return (sbuf_footprint(tm, tn, tk, acc) <= acc.sbuf_bytes
            and psum_footprint(tm, tn, acc) <= acc.psum_bytes)


def compute_cycles(m: int, n: int, k: int, tm: int, tn: int, tk: int, acc: Accel) -> int:
    """PE-array cycles: per instruction, load a stationary (<=P x P) block, then stream columns.

    Stationary block is A^T[k:k+P, m:m+P]; moving operand is B[k:k+P, n:n+moving_max].
    """
    total = 0
    for mi in range(0, m, tm):
        mt = min(tm, m - mi)
        for ni in range(0, n, tn):
            nt = min(tn, n - ni)
            ks = [min(tk, k - kk) for kk in range(0, k, tk)]
            for kt in ks:
                n_stat = _cdiv(mt, acc.pe_cols) * _cdiv(kt, acc.pe_rows)
                n_mov = _cdiv(nt, acc.moving_max)
                total += n_stat * (n_mov * acc.pe_rows + nt)
    return total


def tile_stats(m: int, n: int, k: int, tm: int, tn: int, tk: int, acc: Accel = Accel(),
               with_cycles: bool = True) -> TileStats:
    b = acc.bytes_per_el
    gm, gn, gk = _cdiv(m, tm), _cdiv(n, tn), _cdiv(k, tk)
    hbm = b * (m * k * gn + k * n * gm + m * n)
    dmas = gm * gn * (2 * gk + 1)
    flops = 2 * m * n * k
    cyc = compute_cycles(m, n, k, tm, tn, tk, acc) if with_cycles else 0
    return TileStats(tm, tn, tk, hbm, dmas, flops, flops / hbm,
                     sbuf_footprint(tm, tn, tk, acc), psum_footprint(tm, tn, acc),
                     cyc / acc.freq_hz, hbm / acc.hbm_bw + dmas * acc.dma_overhead_s)


def search_tiles(m: int, n: int, k: int, acc: Accel = Accel(), step: int = 128,
                 max_tile: int = 4096) -> TileStats | None:
    """Max-intensity tile under SBUF+PSUM capacity; ties broken by fewer DMAs."""
    best: TileStats | None = None
    cands = range(step, max_tile + 1, step)
    for tm, tn, tk in product(cands, cands, cands):
        if tm > max(m, step) or tn > max(n, step) or tk > max(k, step):
            continue
        if not fits(tm, tn, tk, acc):
            continue
        s = tile_stats(m, n, k, tm, tn, tk, acc, with_cycles=False)
        if best is None or (s.intensity, -s.dma_count) > (best.intensity, -best.dma_count):
            best = s
    if best is None:
        return None
    return tile_stats(m, n, k, best.tm, best.tn, best.tk, acc)


def tiled_matmul(a: np.ndarray, b: np.ndarray, tm: int, tn: int, tk: int,
                 acc_dtype: type = np.float64) -> tuple[np.ndarray, dict[str, int]]:
    """Execute the loop nest above in numpy and count element loads/stores against HBM."""
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError("inner dims differ")
    c = np.empty((m, n), dtype=np.result_type(a, b))
    loads = stores = dmas = 0
    for i in range(0, m, tm):
        for j in range(0, n, tn):
            psum = np.zeros((min(tm, m - i), min(tn, n - j)), dtype=acc_dtype)
            for kk in range(0, k, tk):
                a_t = a[i:i + tm, kk:kk + tk]
                b_t = b[kk:kk + tk, j:j + tn]
                loads += a_t.size + b_t.size
                dmas += 2
                psum += a_t.astype(acc_dtype) @ b_t.astype(acc_dtype)
            c[i:i + tm, j:j + tn] = psum
            stores += psum.size
            dmas += 1
    return c, {"loads": loads, "stores": stores, "dmas": dmas}


def best_square_tile(acc: Accel, step: int = 128) -> int:
    """Largest t (multiple of step) with tm=tn=t, tk=step feasible."""
    t = step
    while fits(t + step, t + step, step, acc):
        t += step
    return t


def intensity_upper_bound(tm: int, tn: int, bytes_per_el: int) -> float:
    """Ignoring C traffic: 2MNK / (b*(MK*N/tn + KN*M/tm)) = 2 / (b*(1/tn + 1/tm))."""
    return 2.0 / (bytes_per_el * (1.0 / tn + 1.0 / tm))


__all__ = ["Accel", "TileStats", "tile_stats", "search_tiles", "tiled_matmul", "fits",
           "best_square_tile", "intensity_upper_bound", "compute_cycles"]
