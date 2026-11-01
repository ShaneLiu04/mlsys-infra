import numpy as np
import pytest

from servesim.block_manager import (BlockManager, OutOfBlocks, contiguous_attention,
                                    paged_attention, paged_attention_blockwise)

H, D, BS = 4, 8, 4


def _mgr(blocks=32, cpu=32):
    return BlockManager(blocks, BS, H, D, num_cpu_blocks=cpu)


def _fill(mgr, seq, n, rng):
    k = rng.normal(size=(n, H, D))
    v = rng.normal(size=(n, H, D))
    mgr.append_many(seq, n, k, v)
    return k, v


def test_allocate_append_free_no_leak():
    rng = np.random.default_rng(0)
    mgr = _mgr()
    for s, n in enumerate([1, 4, 5, 13]):
        mgr.add(s)
        _fill(mgr, s, n, rng)
        assert len(mgr.tables[s]) == -(-n // BS)
    mgr.check_invariants()
    assert mgr.gpu.num_free == 32 - (1 + 1 + 2 + 4)
    for s in range(4):
        mgr.free(s)
    mgr.check_invariants()
    assert mgr.gpu.num_free == 32


@pytest.mark.parametrize("n", [1, 3, 4, 17, 64])
def test_paged_attention_equals_contiguous(n):
    rng = np.random.default_rng(n)
    mgr = _mgr(blocks=40)
    mgr.add(9)  # interleave another sequence so physical blocks are non-contiguous
    mgr.add(0)
    ks, vs = [], []
    for i in range(n):
        k, v = rng.normal(size=(2, H, D))
        mgr.append(0, k, v)
        ks.append(k)
        vs.append(v)
        mgr.append(9, k, v)
    q = rng.normal(size=(H, D))
    ref = contiguous_attention(q, np.stack(ks), np.stack(vs))
    np.testing.assert_array_equal(paged_attention(q, mgr, 0), ref)
    np.testing.assert_allclose(paged_attention_blockwise(q, mgr, 0), ref, rtol=1e-12, atol=1e-14)


def test_fork_copy_on_write_and_refcounts():
    rng = np.random.default_rng(1)
    mgr = _mgr()
    mgr.add(0)
    k0, v0 = _fill(mgr, 0, 6, rng)  # blocks: [full, half]
    mgr.fork(0, 1)
    mgr.fork(0, 2)
    shared = mgr.tables[0]
    assert all(mgr.gpu.refcount[b] == 3 for b in shared)
    mgr.check_invariants()
    k1, v1 = rng.normal(size=(2, H, D))
    mgr.append(1, k1, v1)  # writes into shared half block -> CoW
    assert mgr.tables[1][0] == shared[0] and mgr.tables[1][1] != shared[1]
    assert mgr.gpu.refcount[shared[1]] == 2 and mgr.gpu.refcount[shared[0]] == 3
    assert mgr.copies[-1].src == shared[1]
    k_seq0, _ = mgr.gather(0)
    np.testing.assert_array_equal(k_seq0, k0)  # parent unaffected
    k_seq1, _ = mgr.gather(1)
    np.testing.assert_array_equal(k_seq1, np.concatenate([k0, k1[None]]))
    mgr.check_invariants()
    for s in (0, 1, 2):
        mgr.free(s)
    mgr.check_invariants()
    assert mgr.gpu.num_free == 32


def test_fork_at_block_boundary_needs_no_copy():
    rng = np.random.default_rng(2)
    mgr = _mgr()
    mgr.add(0)
    _fill(mgr, 0, 8, rng)
    mgr.fork(0, 1)
    assert mgr.slots_needed_for_append(1) == 1
    mgr.append(1, *rng.normal(size=(2, H, D)))
    assert not mgr.copies
    mgr.check_invariants()


def test_swap_out_in_preserves_data_and_sharing():
    rng = np.random.default_rng(3)
    mgr = _mgr()
    mgr.add(0)
    _fill(mgr, 0, 10, rng)
    mgr.fork(0, 1)
    mgr.append(1, *rng.normal(size=(2, H, D)))
    before = {s: mgr.gather(s) for s in (0, 1)}
    q = rng.normal(size=(H, D))
    att = paged_attention(q, mgr, 1)
    moved = mgr.swap_out([0, 1])
    assert moved == 4  # 2 shared + parent tail + child's CoW tail
    assert mgr.gpu.num_free == 32
    mgr.check_invariants()
    mgr.swap_in([0, 1])
    mgr.check_invariants()
    for s in (0, 1):
        np.testing.assert_array_equal(mgr.gather(s)[0], before[s][0])
        np.testing.assert_array_equal(mgr.gather(s)[1], before[s][1])
    np.testing.assert_array_equal(paged_attention(q, mgr, 1), att)
    assert mgr.tables[0][0] == mgr.tables[1][0]


def test_recompute_preemption_frees_everything():
    rng = np.random.default_rng(4)
    mgr = _mgr()
    mgr.add(0)
    _fill(mgr, 0, 11, rng)
    assert mgr.preempt_recompute(0) == 11
    assert mgr.gpu.num_free == 32
    mgr.check_invariants()


def test_out_of_blocks():
    mgr = BlockManager(2, BS)
    mgr.add(0)
    mgr.append_many(0, 8)
    assert not mgr.can_allocate(1)
    with pytest.raises(OutOfBlocks):
        mgr.append(0)
