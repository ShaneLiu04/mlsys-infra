"""Paged KV-cache decode attention: Triton kernel (block-table walk + online softmax) and a
PyTorch gather reference.

KV pool layout per layer: k_pool, v_pool [num_blocks, block_size, Hk, D]. A sequence's logical
token t lives in physical block ``block_table[s, t // block_size]`` at offset ``t % block_size``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover - CPU-only environments
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _paged_decode_kernel(Q, K, V, BT, LENS, O, scale,
                             s_qb, s_qh, s_kb, s_kt, s_kh, s_btb, s_ob, s_oh,
                             GROUP: tl.constexpr, BLOCK: tl.constexpr, D: tl.constexpr,
                             PAGES: tl.constexpr):
        """One program per (sequence, query head); PAGES pages are loaded per loop iteration."""
        b = tl.program_id(0)
        h = tl.program_id(1)
        kvh = h // GROUP
        n = tl.load(LENS + b)
        offs_d = tl.arange(0, D)
        offs_t = tl.arange(0, PAGES * BLOCK)
        page_of = offs_t // BLOCK
        in_page = offs_t % BLOCK
        q = tl.load(Q + b * s_qb + h * s_qh + offs_d).to(tl.float32) * scale
        l_i = tl.sum(tl.zeros([1], dtype=tl.float32), 0)
        m_i = l_i - float("inf")
        acc = tl.zeros([D], dtype=tl.float32)
        n_pages = tl.cdiv(n, BLOCK)
        for p0 in range(0, n_pages, PAGES):
            pg = p0 + page_of
            tok = pg * BLOCK + in_page
            valid = tok < n
            blk = tl.load(BT + b * s_btb + pg, mask=pg < n_pages, other=0)
            off = blk * s_kb + in_page * s_kt + kvh * s_kh
            k = tl.load(K + off[:, None] + offs_d[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
            s = tl.sum(k * q[None, :], 1)
            s = tl.where(valid, s, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(s, 0))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new)
            v = tl.load(V + off[:, None] + offs_d[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
            acc = acc * alpha + tl.sum(p[:, None] * v, 0)
            l_i = l_i * alpha + tl.sum(p, 0)
            m_i = m_new
        tl.store(O + b * s_ob + h * s_oh + offs_d, (acc / l_i).to(O.dtype.element_ty))


def paged_decode(q: torch.Tensor, k_pool: torch.Tensor, v_pool: torch.Tensor,
                 block_table: torch.Tensor, lens: torch.Tensor, tokens_per_iter: int = 64) -> torch.Tensor:
    """q [B,H,D]; pools [NB,bs,Hk,D]; block_table [B,maxb] int32; lens [B] int32 -> [B,H,D]."""
    B, H, D = q.shape
    _, bs, Hk, _ = k_pool.shape
    assert k_pool.stride() == v_pool.stride()
    o = torch.empty_like(q)
    pages = max(1, tokens_per_iter // bs)
    _paged_decode_kernel[(B, H)](
        q, k_pool, v_pool, block_table, lens, o, 1.0 / math.sqrt(D),
        q.stride(0), q.stride(1), k_pool.stride(0), k_pool.stride(1), k_pool.stride(2),
        block_table.stride(0), o.stride(0), o.stride(1),
        GROUP=H // Hk, BLOCK=bs, D=D, PAGES=pages, num_warps=4)
    return o


def gather_decode(q: torch.Tensor, k_pool: torch.Tensor, v_pool: torch.Tensor,
                  block_table: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    """PyTorch reference: gather pages into a padded [B,Hk,Lmax,D] tensor, masked SDPA."""
    B, H, D = q.shape
    _, bs, Hk, _ = k_pool.shape
    maxb = block_table.shape[1]
    idx = block_table.long()
    k = k_pool[idx].view(B, maxb * bs, Hk, D).transpose(1, 2)
    v = v_pool[idx].view(B, maxb * bs, Hk, D).transpose(1, 2)
    mask = torch.arange(maxb * bs, device=q.device)[None, :] < lens[:, None].long()
    G = H // Hk
    o = F.scaled_dot_product_attention(q.view(B, Hk, G, D), k, v, attn_mask=mask[:, None, None, :])
    return o.reshape(B, H, D)


def padded_decode(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    """Contiguous padded cache k/v [B,Hk,Lmax,D] (max-length preallocation), masked SDPA."""
    B, H, D = q.shape
    Hk, Lmax = k.shape[1], k.shape[2]
    mask = torch.arange(Lmax, device=q.device)[None, :] < lens[:, None].long()
    o = F.scaled_dot_product_attention(q.view(B, Hk, H // Hk, D), k, v, attn_mask=mask[:, None, None, :])
    return o.reshape(B, H, D)


def reference_decode(q, k_pool, v_pool, block_table, lens) -> torch.Tensor:
    """fp32 per-sequence loop (ground truth)."""
    B, H, D = q.shape
    _, bs, Hk, _ = k_pool.shape
    G = H // Hk
    out = torch.empty(B, H, D, dtype=torch.float32, device=q.device)
    for b in range(B):
        n = int(lens[b])
        nb = -(-n // bs)
        blocks = block_table[b, :nb].long()
        k = k_pool[blocks].reshape(-1, Hk, D)[:n].float()
        v = v_pool[blocks].reshape(-1, Hk, D)[:n].float()
        for h in range(H):
            s = (k[:, h // G] @ q[b, h].float()) / math.sqrt(D)
            out[b, h] = torch.softmax(s, 0) @ v[:, h // G]
    return out


def build_paged(lens: list[int], bs: int, Hk: int, D: int, dtype=torch.bfloat16, device="cuda",
                shuffle: bool = True, seed: int = 0, extra_blocks: int = 0):
    """Random KV pool holding sequences of ``lens``; physical blocks shuffled (paged) or in order."""
    nbs = [-(-n // bs) for n in lens]
    total = sum(nbs) + extra_blocks
    g = torch.Generator(device="cpu").manual_seed(seed)
    k_pool = torch.randn(total, bs, Hk, D, generator=g).to(device=device, dtype=dtype)
    v_pool = torch.randn(total, bs, Hk, D, generator=g).to(device=device, dtype=dtype)
    phys = torch.randperm(total, generator=g) if shuffle else torch.arange(total)
    maxb = max(nbs)
    bt = torch.zeros(len(lens), maxb, dtype=torch.int32)
    c = 0
    for i, nb in enumerate(nbs):
        bt[i, :nb] = phys[c:c + nb].to(torch.int32)
        c += nb
    return k_pool, v_pool, bt.to(device), torch.tensor(lens, dtype=torch.int32, device=device)
