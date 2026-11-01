"""Register-level simulator of a weight-stationary P x P systolic array.

PE(i, j) holds weight W[i, j] (i = reduction index k, j = output column n).
Activations enter row i from the left skewed by i cycles and move right one PE per cycle;
partial sums move down one PE per cycle and leave the bottom row. Row m of A therefore
reaches PE(i, j) at cycle m + i + j and its result for column j exits at cycle m + P-1 + j,
so one weight block takes M + 2P - 2 cycles (M streaming + 2(P-1) fill/drain).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SystolicStats:
    p: int
    m: int
    n: int
    k: int
    blocks: int
    stream_cycles: int  # simulated cycles summed over blocks (fill + stream + drain)
    weight_load_cycles: int
    useful_macs: int
    active_pe_cycles: int  # PE-cycles that saw a valid (possibly zero-padded) activation

    @property
    def serial_cycles(self) -> int:
        """Weights loaded before each block, blocks run back to back."""
        return self.stream_cycles + self.weight_load_cycles

    @property
    def pipelined_cycles(self) -> int:
        """Double-buffered weights and back-to-back streams: fill/drain paid once."""
        per_block = max(self.m, self.p)
        return self.p + self.blocks * per_block + 2 * self.p - 2

    def utilization(self, pipelined: bool = False) -> float:
        cyc = self.pipelined_cycles if pipelined else self.serial_cycles
        return self.useful_macs / (self.p * self.p * cyc)


def block_cycles(m: int, p: int) -> int:
    return m + 2 * p - 2


def _run_block(a_blk: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Simulate one weight block. a_blk: (M, P) activations, w: (P, P). Returns (M, P) output."""
    m, p = a_blk.shape
    act = np.zeros((p, p))
    valid = np.zeros((p, p), dtype=bool)
    psum = np.zeros((p, p))
    out = np.zeros((m, p))
    rows = np.arange(p)
    cols = np.arange(p)
    n_cycles = block_cycles(m, p)
    active = 0
    for t in range(n_cycles):
        act[:, 1:] = act[:, :-1]
        valid[:, 1:] = valid[:, :-1]
        mi = t - rows
        ok = (mi >= 0) & (mi < m)
        act[:, 0] = np.where(ok, a_blk[np.clip(mi, 0, m - 1), rows], 0.0)
        valid[:, 0] = ok
        prod = act * w
        new = np.empty_like(psum)
        new[0] = prod[0]
        new[1:] = psum[:-1] + prod[1:]
        psum = new
        active += int(valid.sum())
        mo = t - (p - 1) - cols
        done = (mo >= 0) & (mo < m)
        out[mo[done], cols[done]] = psum[p - 1, done]
    return out, n_cycles, active


def systolic_matmul(a: np.ndarray, b: np.ndarray, p: int) -> tuple[np.ndarray, SystolicStats]:
    """C = A @ B on a P x P weight-stationary array; B is tiled into P x P weight blocks."""
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError("inner dims differ")
    kp, np_ = -(-k // p) * p, -(-n // p) * p
    a_pad = np.zeros((m, kp))
    a_pad[:, :k] = a
    b_pad = np.zeros((kp, np_))
    b_pad[:k, :n] = b
    c = np.zeros((m, np_))
    stream = active = blocks = 0
    for kb in range(0, kp, p):
        for nb in range(0, np_, p):
            out, cyc, act = _run_block(a_pad[:, kb:kb + p], b_pad[kb:kb + p, nb:nb + p])
            c[:, nb:nb + p] += out
            stream += cyc
            active += act
            blocks += 1
    stats = SystolicStats(p, m, n, k, blocks, stream, blocks * p, m * n * k, active)
    return c[:, :n], stats
