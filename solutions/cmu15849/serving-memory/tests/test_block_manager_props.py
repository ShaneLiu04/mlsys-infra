"""Random allocate / fork / write / swap / free sequences keep the block-manager invariants.

The stored K/V of every live sequence is checked against an independent copy, so a
copy-on-write that aliased a parent's tokens would fail here.
"""

import numpy as np

from servesim.block_manager import BlockManager, OutOfBlocks


def test_random_operations_preserve_tokens_and_refcounts():
    rng = np.random.default_rng(0)
    h, d, bs = 2, 4, 8
    mgr = BlockManager(48, bs, h, d, num_cpu_blocks=48)
    tokens: dict[int, tuple[list[np.ndarray], list[np.ndarray]]] = {}
    next_id = 0

    def fresh() -> int:
        nonlocal next_id
        next_id += 1
        mgr.add(next_id)
        tokens[next_id] = ([], [])
        return next_id

    for _ in range(300):
        op = int(rng.integers(0, 5))
        live = [s for s in tokens if s in mgr.tables]
        if op == 0 or not tokens:
            fresh()
        elif op == 1 and live:
            seq = int(rng.choice(live))
            k = rng.normal(size=(h, d))
            v = rng.normal(size=(h, d))
            try:
                mgr.append(seq, k, v)
            except OutOfBlocks:
                pass
            else:
                tokens[seq][0].append(k)
                tokens[seq][1].append(v)
        elif op == 2 and live:
            parent = int(rng.choice(live))
            next_id += 1
            child = next_id
            mgr.fork(parent, child)
            tokens[child] = ([k.copy() for k in tokens[parent][0]],
                             [v.copy() for v in tokens[parent][1]])
        elif op == 3 and live:
            seq = int(rng.choice(live))
            mgr.free(seq)
            tokens.pop(seq)
        elif op == 4 and live:
            seq = int(rng.choice(live))
            mgr.swap_out([seq])
            mgr.swap_in([seq])
        mgr.check_invariants()
        for seq in list(tokens):
            if seq not in mgr.tables:
                continue
            got_k, got_v = mgr.gather(seq)
            ks, vs = tokens[seq]
            if not ks:
                assert got_k.shape[0] == 0
                continue
            np.testing.assert_allclose(got_k, np.stack(ks))
            np.testing.assert_allclose(got_v, np.stack(vs))
    for seq in list(tokens):
        if seq in mgr.tables or seq in mgr.swapped:
            mgr.free(seq)
    mgr.check_invariants()
    assert mgr.gpu.num_free == mgr.gpu.num_blocks
    assert mgr.cpu.num_free == mgr.cpu.num_blocks
