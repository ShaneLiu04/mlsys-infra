"""All-reduce algorithms over P simulated ranks, with an alpha-beta cost model.

Every algorithm moves real numpy arrays through ``SimNetwork.step``; a step is a set of
messages that are sent concurrently. Cost of a step = alpha + beta * (bytes on the busiest
port or link). Two topologies are modelled:

* ``full``: every pair has a dedicated link (switch / fat tree); only ports contend.
* ``ring``: bidirectional physical ring; a message to a rank d hops away occupies d links,
  so long-distance exchanges (halving-doubling) contend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import mlsys_calc

ALGORITHMS = ("naive", "ring", "halving_doubling", "tree")


@dataclass
class CommStats:
    world: int
    steps: int = 0
    bytes_sent: np.ndarray = field(default=None)  # type: ignore[assignment]
    bytes_recv: np.ndarray = field(default=None)  # type: ignore[assignment]
    port_bytes: list[int] = field(default_factory=list)  # busiest port per step
    link_bytes: list[int] = field(default_factory=list)  # busiest ring link per step

    def __post_init__(self) -> None:
        self.bytes_sent = np.zeros(self.world, dtype=np.int64)
        self.bytes_recv = np.zeros(self.world, dtype=np.int64)

    def time(self, alpha: float, beta: float, topology: str = "full", scale: float = 1.0) -> float:
        """Modelled time; ``scale`` rescales message bytes (traces are linear in size)."""
        per_step = self.port_bytes if topology == "full" else [
            max(p, l) for p, l in zip(self.port_bytes, self.link_bytes)]
        return self.steps * alpha + beta * scale * float(sum(per_step))

    def merge(self, other: "CommStats") -> None:
        self.steps += other.steps
        self.bytes_sent += other.bytes_sent
        self.bytes_recv += other.bytes_recv
        self.port_bytes += other.port_bytes
        self.link_bytes += other.link_bytes


class SimNetwork:
    """Delivers a batch of concurrent messages and records traffic."""

    def __init__(self, world: int) -> None:
        self.world = world
        self.stats = CommStats(world)

    def _ring_route(self, src: int, dst: int) -> tuple[int, int]:
        fwd = (dst - src) % self.world
        bwd = (src - dst) % self.world
        if fwd < bwd or (fwd == bwd and src < dst):
            return +1, fwd
        return -1, bwd

    def step(self, msgs: list[tuple[int, int, np.ndarray]]) -> list[tuple[int, int, np.ndarray]]:
        """Send ``(src, dst, payload)`` messages concurrently; returns delivered copies."""
        if not msgs:
            return []
        sent = np.zeros(self.world, dtype=np.int64)
        recv = np.zeros(self.world, dtype=np.int64)
        links: dict[tuple[int, int], int] = {}
        out = []
        for src, dst, payload in msgs:
            nb = int(payload.nbytes)
            sent[src] += nb
            recv[dst] += nb
            direction, hops = self._ring_route(src, dst)
            node = src
            for _ in range(hops):
                key = (node, direction)
                links[key] = links.get(key, 0) + nb
                node = (node + direction) % self.world
            out.append((dst, src, payload.copy()))
        s = self.stats
        s.steps += 1
        s.bytes_sent += sent
        s.bytes_recv += recv
        s.port_bytes.append(int(max(sent.max(), recv.max())))
        s.link_bytes.append(int(max(links.values())))
        return out


def _chunks(n: int, p: int) -> list[slice]:
    if n % p:
        raise ValueError(f"length {n} must be divisible by world size {p}")
    c = n // p
    return [slice(i * c, (i + 1) * c) for i in range(p)]


def ring_reduce_scatter(bufs: list[np.ndarray], net: SimNetwork) -> list[np.ndarray]:
    """After P-1 steps rank r holds the fully reduced chunk r."""
    p = len(bufs)
    work = [b.copy() for b in bufs]
    sl = _chunks(len(work[0]), p)
    for s in range(p - 1):
        msgs = [(r, (r + 1) % p, work[r][sl[(r - s - 1) % p]]) for r in range(p)]
        for dst, src, data in net.step(msgs):
            work[dst][sl[(src - s - 1) % p]] += data
    return [work[r][sl[r]].copy() for r in range(p)]


def ring_all_gather(shards: list[np.ndarray], net: SimNetwork) -> list[np.ndarray]:
    """Inverse of reduce-scatter: rank r contributes shard r, everyone ends with all."""
    p = len(shards)
    c = len(shards[0])
    work = [np.zeros(c * p, dtype=shards[0].dtype) for _ in range(p)]
    for r in range(p):
        work[r][r * c:(r + 1) * c] = shards[r]
    for s in range(p - 1):
        msgs = [(r, (r + 1) % p, work[r][((r - s) % p) * c:((r - s) % p + 1) * c]) for r in range(p)]
        for dst, src, data in net.step(msgs):
            k = (src - s) % p
            work[dst][k * c:(k + 1) * c] = data
    return work


def ring_allreduce(bufs: list[np.ndarray], net: SimNetwork) -> list[np.ndarray]:
    return ring_all_gather(ring_reduce_scatter(bufs, net), net)


def naive_allreduce(bufs: list[np.ndarray], net: SimNetwork, root: int = 0) -> list[np.ndarray]:
    """Gather to root then broadcast; the root's single port serialises 2(P-1) messages."""
    p = len(bufs)
    acc = bufs[root].copy()
    for r in range(p):
        if r != root:
            for _, _, data in net.step([(r, root, bufs[r])]):
                acc += data
    out = [None] * p
    out[root] = acc
    for r in range(p):
        if r != root:
            out[r] = net.step([(root, r, acc)])[0][2]
    return out  # type: ignore[return-value]


def tree_allreduce(bufs: list[np.ndarray], net: SimNetwork) -> list[np.ndarray]:
    """Binomial-tree reduce to rank 0, then binomial-tree broadcast (whole vector per hop)."""
    p = len(bufs)
    work = [b.copy() for b in bufs]
    rounds = math.ceil(math.log2(p)) if p > 1 else 0
    for k in range(rounds):
        d = 1 << k
        msgs = [(r, r - d, work[r]) for r in range(p) if r % (2 * d) == d]
        for dst, _, data in net.step(msgs):
            work[dst] += data
    for k in reversed(range(rounds)):
        d = 1 << k
        msgs = [(r, r + d, work[r]) for r in range(p) if r % (2 * d) == 0 and r + d < p]
        for dst, _, data in net.step(msgs):
            work[dst] = data
    return work


def halving_doubling_allreduce(bufs: list[np.ndarray], net: SimNetwork) -> list[np.ndarray]:
    """Rabenseifner: recursive-halving reduce-scatter + recursive-doubling all-gather."""
    p = len(bufs)
    if p & (p - 1):
        raise ValueError("halving-doubling needs a power-of-two world size")
    n = len(bufs[0])
    _chunks(n, p)
    work = [b.copy() for b in bufs]
    lo = [0] * p
    hi = [n] * p
    d = p // 2
    while d >= 1:
        msgs, keep = [], []
        for r in range(p):
            mid = (lo[r] + hi[r]) // 2
            if r & d:
                msgs.append((r, r ^ d, work[r][lo[r]:mid]))
                keep.append((mid, hi[r]))
            else:
                msgs.append((r, r ^ d, work[r][mid:hi[r]]))
                keep.append((lo[r], mid))
        for r, (a, b) in enumerate(keep):
            lo[r], hi[r] = a, b
        for dst, _, data in net.step(msgs):
            work[dst][lo[dst]:hi[dst]] += data
        d //= 2
    d = 1
    while d < p:
        msgs = [(r, r ^ d, work[r][lo[r]:hi[r]]) for r in range(p)]
        new = [(min(lo[r], lo[r ^ d]), max(hi[r], hi[r ^ d])) for r in range(p)]
        for dst, src, data in net.step(msgs):
            work[dst][lo[src]:hi[src]] = data
        for r in range(p):
            lo[r], hi[r] = new[r]
        d *= 2
    return work


def allreduce(alg: str, bufs: list[np.ndarray]) -> tuple[list[np.ndarray], CommStats]:
    net = SimNetwork(len(bufs))
    fn = {"naive": naive_allreduce, "ring": ring_allreduce,
          "halving_doubling": halving_doubling_allreduce, "tree": tree_allreduce}[alg]
    return fn(bufs, net), net.stats


def closed_form_time(alg: str, nbytes: float, p: int, alpha: float, beta: float) -> float:
    """Alpha-beta formulas on a full (non-blocking) topology."""
    lg = math.ceil(math.log2(p))
    if alg == "naive":
        return 2 * (p - 1) * (alpha + nbytes * beta)
    if alg == "ring":
        return mlsys_calc.ring_allreduce_time(nbytes, p, 1.0 / beta, alpha)
    if alg == "halving_doubling":
        return 2 * lg * alpha + 2 * (p - 1) / p * nbytes * beta
    if alg == "tree":
        return 2 * lg * (alpha + nbytes * beta)
    raise KeyError(alg)
