"""vLLM-style paged KV cache: block allocator, block tables, CoW fork, swap, recompute.

Physical K/V live in pools of shape [num_blocks, block_size, heads, head_dim]. A sequence
owns a block table (list of physical block ids). Blocks are refcounted so that forked
sequences (parallel sampling, beam search) share the prompt; the first write into a shared
block triggers copy-on-write. With ``heads=0`` only the bookkeeping runs (scheduler use).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class OutOfBlocks(RuntimeError):
    pass


class BlockAllocator:
    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        self.refcount = np.zeros(num_blocks, dtype=np.int64)
        self._free = list(range(num_blocks - 1, -1, -1))

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self) -> int:
        if not self._free:
            raise OutOfBlocks("no free blocks")
        b = self._free.pop()
        self.refcount[b] = 1
        return b

    def incref(self, b: int) -> None:
        assert self.refcount[b] > 0
        self.refcount[b] += 1

    def decref(self, b: int) -> None:
        assert self.refcount[b] > 0
        self.refcount[b] -= 1
        if self.refcount[b] == 0:
            self._free.append(b)


@dataclass
class CopyEvent:
    src: int
    dst: int


class BlockManager:
    def __init__(self, num_gpu_blocks: int, block_size: int, heads: int = 0, head_dim: int = 0,
                 num_cpu_blocks: int = 0, dtype=np.float64) -> None:
        self.block_size = block_size
        self.gpu = BlockAllocator(num_gpu_blocks)
        self.cpu = BlockAllocator(num_cpu_blocks)
        self.tables: dict[int, list[int]] = {}
        self.lengths: dict[int, int] = {}
        self.swapped: dict[int, list[int]] = {}
        self.copies: list[CopyEvent] = []
        self.store = heads > 0
        if self.store:
            shape = (block_size, heads, head_dim)
            self.k = np.zeros((num_gpu_blocks, *shape), dtype=dtype)
            self.v = np.zeros_like(self.k)
            self.k_cpu = np.zeros((num_cpu_blocks, *shape), dtype=dtype)
            self.v_cpu = np.zeros_like(self.k_cpu)

    # ------------------------------------------------------------------ allocation

    def blocks_needed(self, n_tokens: int) -> int:
        return -(-n_tokens // self.block_size)

    def can_allocate(self, n_tokens: int, watermark: int = 0) -> bool:
        return self.gpu.num_free - self.blocks_needed(n_tokens) >= watermark

    def add(self, seq: int) -> None:
        if seq in self.tables or seq in self.swapped:
            raise KeyError(f"sequence {seq} exists")
        self.tables[seq] = []
        self.lengths[seq] = 0

    def _writable_slot(self, seq: int) -> tuple[int, int]:
        table = self.tables[seq]
        pos = self.lengths[seq]
        off = pos % self.block_size
        if off == 0:
            table.append(self.gpu.allocate())
        else:
            last = table[-1]
            if self.gpu.refcount[last] > 1:  # copy-on-write
                new = self.gpu.allocate()
                if self.store:
                    self.k[new] = self.k[last]
                    self.v[new] = self.v[last]
                self.gpu.decref(last)
                table[-1] = new
                self.copies.append(CopyEvent(last, new))
        return table[-1], off

    def append(self, seq: int, k: np.ndarray | None = None, v: np.ndarray | None = None) -> None:
        """Append one token; k, v have shape [heads, head_dim] when storage is enabled."""
        b, off = self._writable_slot(seq)
        if self.store:
            self.k[b, off] = k
            self.v[b, off] = v
        self.lengths[seq] += 1

    def append_many(self, seq: int, n: int, k: np.ndarray | None = None, v: np.ndarray | None = None) -> None:
        if self.store:
            for i in range(n):
                self.append(seq, k[i], v[i])
            return
        while n > 0:
            self._writable_slot(seq)
            take = min(n, self.block_size - self.lengths[seq] % self.block_size)
            self.lengths[seq] += take
            n -= take

    def slots_needed_for_append(self, seq: int) -> int:
        """Blocks the next append will allocate (new block or CoW copy)."""
        table, pos = self.tables[seq], self.lengths[seq]
        if pos % self.block_size == 0:
            return 1
        return int(self.gpu.refcount[table[-1]] > 1)

    def fork(self, parent: int, child: int) -> None:
        self.add(child)
        self.tables[child] = list(self.tables[parent])
        self.lengths[child] = self.lengths[parent]
        for b in self.tables[child]:
            self.gpu.incref(b)

    def free(self, seq: int) -> None:
        if seq in self.swapped:
            for b in self.swapped.pop(seq):
                self.cpu.decref(b)
        else:
            for b in self.tables.pop(seq):
                self.gpu.decref(b)
        self.tables.pop(seq, None)
        del self.lengths[seq]

    # ------------------------------------------------------------------ preemption

    def swap_out(self, seqs: list[int]) -> int:
        """Move a sequence group to CPU, preserving sharing inside the group. Returns blocks moved."""
        mapping: dict[int, int] = {}
        for s in seqs:
            cpu_table = []
            for b in self.tables[s]:
                if b not in mapping:
                    mapping[b] = self.cpu.allocate()
                    if self.store:
                        self.k_cpu[mapping[b]] = self.k[b]
                        self.v_cpu[mapping[b]] = self.v[b]
                else:
                    self.cpu.incref(mapping[b])
                cpu_table.append(mapping[b])
                self.gpu.decref(b)
            self.swapped[s] = cpu_table
            del self.tables[s]
        return len(mapping)

    def swap_in(self, seqs: list[int]) -> int:
        mapping: dict[int, int] = {}
        for s in seqs:
            table = []
            for c in self.swapped[s]:
                if c not in mapping:
                    mapping[c] = self.gpu.allocate()
                    if self.store:
                        self.k[mapping[c]] = self.k_cpu[c]
                        self.v[mapping[c]] = self.v_cpu[c]
                else:
                    self.gpu.incref(mapping[c])
                table.append(mapping[c])
                self.cpu.decref(c)
            self.tables[s] = table
            del self.swapped[s]
        return len(mapping)

    def preempt_recompute(self, seq: int) -> int:
        """Drop all KV of ``seq``; returns tokens that must be prefilled again."""
        n = self.lengths[seq]
        self.free(seq)
        return n

    # ------------------------------------------------------------------ reading

    def gather(self, seq: int) -> tuple[np.ndarray, np.ndarray]:
        n = self.lengths[seq]
        table = np.asarray(self.tables[seq], dtype=np.int64)
        pos = np.arange(n)
        blocks, offs = table[pos // self.block_size], pos % self.block_size
        return self.k[blocks, offs], self.v[blocks, offs]

    def check_invariants(self) -> None:
        for alloc, tables in ((self.gpu, self.tables), (self.cpu, self.swapped)):
            counts = np.zeros(alloc.num_blocks, dtype=np.int64)
            for t in tables.values():
                for b in t:
                    counts[b] += 1
            assert np.array_equal(counts, alloc.refcount), "refcount mismatch"
            free = alloc._free
            assert len(set(free)) == len(free), "double free"
            assert set(free) == set(np.nonzero(alloc.refcount == 0)[0].tolist()), "free list mismatch"
        for s, t in self.tables.items():
            assert len(t) == self.blocks_needed(self.lengths[s]), "table length mismatch"


def contiguous_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Single-query attention: q [heads, d], k/v [n, heads, d]."""
    scores = np.einsum("hd,nhd->hn", q, k) / np.sqrt(q.shape[-1])
    scores -= scores.max(-1, keepdims=True)
    w = np.exp(scores)
    w /= w.sum(-1, keepdims=True)
    return np.einsum("hn,nhd->hd", w, v)


def paged_attention(q: np.ndarray, mgr: BlockManager, seq: int) -> np.ndarray:
    """Decode attention over the paged K/V of ``seq``, gathered through its block table."""
    k, v = mgr.gather(seq)
    return contiguous_attention(q, k, v)


def paged_attention_blockwise(q: np.ndarray, mgr: BlockManager, seq: int) -> np.ndarray:
    """Kernel-style variant: iterate blocks with an online softmax, never materialising K/V."""
    n, bs = mgr.lengths[seq], mgr.block_size
    h, d = q.shape
    m = np.full(h, -np.inf)
    s = np.zeros(h)
    acc = np.zeros((h, d))
    for i, b in enumerate(mgr.tables[seq]):
        cnt = min(bs, n - i * bs)
        kb, vb = mgr.k[b, :cnt], mgr.v[b, :cnt]
        sc = np.einsum("hd,nhd->hn", q, kb) / np.sqrt(d)
        m_new = np.maximum(m, sc.max(-1))
        scale = np.exp(m - m_new)
        p = np.exp(sc - m_new[:, None])
        s = s * scale + p.sum(-1)
        acc = acc * scale[:, None] + np.einsum("hn,nhd->hd", p, vb)
        m = m_new
    return acc / s[:, None]
