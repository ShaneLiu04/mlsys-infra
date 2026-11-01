"""Analytic shared-memory bank-conflict model for the smem accesses in sgemm.cu (warp 0, one k step).

32 banks x 4 B. A w-byte access by a warp is served in 32*w/128 phases (4 B: 1 phase of 32 lanes,
16 B: 4 phases of 8 lanes); within a phase each bank delivers one distinct 4 B word per wavefront and
identical words are broadcast. wavefronts = sum over phases of max over banks of distinct words.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

BANKS = 32


def wavefronts(word_addrs_per_lane: list[list[int]], width_bytes: int = 4) -> int:
    lanes_per_phase = 32 * 4 // width_bytes if width_bytes <= 4 else 128 // width_bytes
    lanes_per_phase = min(32, lanes_per_phase)
    total = 0
    for p0 in range(0, len(word_addrs_per_lane), lanes_per_phase):
        banks: dict[int, set[int]] = defaultdict(set)
        for lane in word_addrs_per_lane[p0:p0 + lanes_per_phase]:
            for w in lane:
                banks[w % BANKS].add(w)
        total += max((len(s) for s in banks.values()), default=0)
    return total


def access(fn: Callable[[int], int], width_bytes: int = 4) -> int:
    """fn(lane) -> first 4 B word index accessed by that lane (32 lanes of warp 0)."""
    n = width_bytes // 4
    return wavefronts([[fn(t) + i for i in range(n)] for t in range(32)], width_bytes)


def ideal(width_bytes: int) -> int:
    return max(1, width_bytes // 4)


def kernel_patterns() -> dict[str, dict[str, tuple[int, int]]]:
    """{kernel: {access: (wavefronts, ideal)}} for the inner-loop smem accesses of each sgemm.cu kernel."""
    out: dict[str, dict[str, tuple[int, int]]] = {}
    k = 3  # any k inside the BK loop gives the same pattern

    # k_smem<T>: block (T, T), lane -> (tx = t % T, ty = t // T)
    for T in (8, 16, 32):
        out[f"smem{T}"] = {
            "A load As[ty][k]": (access(lambda t: (t // T) * T + k), 1),
            "B load Bs[k][tx]": (access(lambda t: k * T + t % T), 1),
            "A store As[ty][tx]": (access(lambda t: (t // T) * T + t % T), 1),
        }
    # k_reg1d<64,64,8,8>: tcol = tid % 64, trow = tid / 64; As[(trow*8+i)*8 + k], Bs[k*64 + tcol]
    out["reg1d"] = {
        "A load As[(trow*TM+i)*BK+k]": (access(lambda t: ((t // 64) * 8 + 0) * 8 + k), 1),
        "B load Bs[k*BN+tcol]": (access(lambda t: k * 64 + t % 64), 1),
        "A store As[r*BK+c]": (access(lambda t: t), 1),
    }
    # k_reg2d<128,128,8,8,8>: tcol = tid % 16, trow = tid / 16; As stored transposed As[c*BM + r]
    out["reg2d"] = {
        "A load As[k*BM+trow*TM+i]": (access(lambda t: k * 128 + (t // 16) * 8), 1),
        "B load Bs[k*BN+tcol*TN+j]": (access(lambda t: k * 128 + (t % 16) * 8), 1),
        "A store As[c*BM+r] (transpose)": (access(lambda t: (t % 8) * 128 + t // 8), 1),
        "B store Bs[r*BN+c]": (access(lambda t: t), 1),
    }
    # k_reg2d<..., NC=true>: As padded to 132 floats per k row, thread columns tcol + 16*j
    out["reg2d_nc"] = {
        "A load As[k*LDA+trow*TM+i]": (access(lambda t: k * 132 + (t // 16) * 8), 1),
        "B load Bs[k*BN+tcol+j*16]": (access(lambda t: k * 128 + t % 16), 1),
        "A store As[c*LDA+r] (transpose)": (access(lambda t: (t % 8) * 132 + t // 8), 1),
        "B store Bs[r*BN+c]": (access(lambda t: t), 1),
    }
    # k_vec: float4 loads As[k][trow*8], Bs[k][tcol*8]; A store transposes a float4 into 4 scalars
    out["vec4"] = {
        "A load float4 As[k][trow*8]": (access(lambda t: k * 128 + (t // 16) * 8, 16), ideal(16)),
        "B load float4 Bs[k][tcol*8]": (access(lambda t: k * 128 + (t % 16) * 8, 16), ideal(16)),
        "A store As[ak+q][ar] (transpose)": (access(lambda t: ((t & 1) * 4) * 128 + (t >> 1)), 1),
        "B store float4 Bs[br][bc]": (access(lambda t: (t >> 5) * 128 + (t & 31) * 4, 16), ideal(16)),
    }
    return out


def rows() -> list[dict]:
    res = []
    for kern, acc in kernel_patterns().items():
        for name, (w, i) in acc.items():
            res.append({"kernel": kern, "access": name, "wavefronts": w, "ideal": i, "conflict_degree": w / i})
    return res


if __name__ == "__main__":
    for r in rows():
        print(r)
