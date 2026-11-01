"""HBM-traffic model of attention (forward, one head) and decode intensity for MHA/GQA/MQA."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[5] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import mlsys_calc as calc  # noqa: E402


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def naive_elements(t: int, d: int) -> int:
    """Standard attention: read Q,K, write S; read S, write P; read P,V, write O = 4Td + 4T^2."""
    return 4 * t * d + 4 * t * t


def flash_blocks(m_bytes: int, d: int, bytes_per_el: int = 2) -> tuple[int, int]:
    """FlashAttention paper choice: Bc = floor(M / 4d), Br = min(Bc, d) (M in elements)."""
    m_el = m_bytes // bytes_per_el
    bc = max(1, m_el // (4 * d))
    return min(bc, d), bc


def flash_elements(t: int, d: int, br: int, bc: int, causal: bool = False) -> int:
    """Matches flash_forward's IOCounter: Q once, K/V once per visited (Q-block, KV-block), O+lse once."""
    kv = 0
    for r0 in range(0, t, br):
        r1 = min(r0 + br, t)
        keys = min(t, _cdiv(r1, bc) * bc) if causal else t
        kv += keys * 2 * d
    return t * d + kv + t * (d + 1)


def attention_io_row(t: int, d: int, m_bytes: int, bytes_per_el: int = 2,
                     causal: bool = False) -> dict[str, float]:
    br, bc = flash_blocks(m_bytes, d, bytes_per_el)
    naive = naive_elements(t, d) * bytes_per_el
    flash = flash_elements(t, d, br, bc, causal) * bytes_per_el
    return {"T": t, "d": d, "sram_kib": m_bytes // 1024, "Br": br, "Bc": bc, "causal": causal,
            "naive_mb": naive / 1e6, "flash_mb": flash / 1e6, "ratio": naive / flash,
            "kv_passes": _cdiv(t, br)}


def decode_attention_intensity(model_key: str = "llama3-8b", gpu_key: str = "h100", seq: int = 4096,
                               bytes_per_el: int = 2) -> list[dict[str, float | str]]:
    """Per decode step, per sequence: FLOPs = 4*H*hd*T (QK^T and PV), bytes = K+V cache read.

    Intensity = 2*(H/H_kv)/bytes, independent of T and of batch (every sequence has its own KV).
    """
    base = calc.MODELS[model_key]
    gpu = calc.GPUS[gpu_key]
    rows = []
    for name, kv_heads in (("MHA", base.n_heads), ("GQA-8", 8), ("GQA-4", 4), ("MQA", 1)):
        m = replace(base, n_kv_heads=kv_heads)
        flops = 4 * m.layers * m.n_heads * m.head_dim * seq
        kv = calc.kv_cache_bytes(m, seq, 1, bytes_per_el)
        rows.append({"variant": name, "n_heads": m.n_heads, "n_kv_heads": kv_heads, "seq": seq,
                     "kv_bytes_per_token_kib": calc.kv_cache_bytes(m, 1, 1, bytes_per_el) / 1024,
                     "kv_cache_gb_per_seq": kv / calc.GB, "attn_flops_per_token_g": flops / 1e9,
                     "intensity_flop_per_byte": flops / kv, "gpu_ridge": gpu.ridge,
                     "attn_time_us_per_seq": calc.roofline_time(flops, kv, gpu)[0] * 1e6,
                     "bound": calc.roofline_time(flops, kv, gpu)[1]})
    return rows
