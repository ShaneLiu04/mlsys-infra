"""Tiny dataflow-graph IR with an interpreter, rewrite passes and an HBM-traffic cost model.

Passes: constant folding, common-subexpression elimination, elementwise fusion (optionally
absorbing a producing matmul as an epilogue), and a TASO-style substitution that merges
matmuls sharing an input:  X@W1, X@W2  ->  slice(X@[W1|W2]).

A ``fused`` node carries a body: a list of steps ``(op, args, attrs)`` where each arg is
``("in", k)`` (k-th external input) or ``("tmp", j)`` (result of step j). It reads its
external inputs from HBM once and writes only its final result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import count

import numpy as np

import mlsys_calc

ELEMENTWISE = {"add", "mul", "relu", "gelu", "exp"}
_ids = count()


@dataclass(eq=False)
class Node:
    op: str
    inputs: list["Node"] = field(default_factory=list)
    attrs: dict = field(default_factory=dict)
    uid: int = field(default_factory=lambda: next(_ids))

    def __repr__(self) -> str:
        return f"{self.op}#{self.uid}"


@dataclass
class Graph:
    outputs: list[Node]

    def nodes(self) -> list[Node]:
        order: list[Node] = []
        seen: set[int] = set()
        stack: list[tuple[Node, bool]] = [(n, False) for n in reversed(self.outputs)]
        while stack:
            n, expanded = stack.pop()
            if expanded:
                order.append(n)
                continue
            if n.uid in seen:
                continue
            seen.add(n.uid)
            stack.append((n, True))
            for i in reversed(n.inputs):
                if i.uid not in seen:
                    stack.append((i, False))
        return order

    def kernels(self) -> list[Node]:
        return [n for n in self.nodes() if n.op not in ("input", "constant")]

    def consumers(self) -> dict[int, int]:
        c: dict[int, int] = {}
        for n in self.nodes():
            for i in n.inputs:
                c[i.uid] = c.get(i.uid, 0) + 1
        for o in self.outputs:
            c[o.uid] = c.get(o.uid, 0) + 1
        return c


# --------------------------------------------------------------------------- builders

def inp(name: str, shape: tuple[int, ...]) -> Node:
    return Node("input", attrs={"name": name, "shape": tuple(shape)})


def const(value: np.ndarray) -> Node:
    value = np.asarray(value, dtype=np.float64)
    return Node("constant", attrs={"value": value, "shape": value.shape})


def op(name: str, *inputs: Node, **attrs) -> Node:
    return Node(name, list(inputs), attrs)


# --------------------------------------------------------------------------- semantics

def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def _layernorm(x: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def apply_op(name: str, args: list[np.ndarray], attrs: dict) -> np.ndarray:
    if name == "matmul":
        return args[0] @ args[1]
    if name == "add":
        return args[0] + args[1]
    if name == "mul":
        return args[0] * args[1]
    if name == "relu":
        return np.maximum(args[0], 0.0)
    if name == "gelu":
        return _gelu(args[0])
    if name == "exp":
        return np.exp(args[0])
    if name == "sum":
        return args[0].sum(axis=attrs.get("axis"), keepdims=attrs.get("keepdims", False))
    if name == "transpose":
        return args[0].T
    if name == "layernorm":
        return _layernorm(*args)
    if name == "concat":
        return np.concatenate(args, axis=attrs["axis"])
    if name == "slice":
        sl = [slice(None)] * args[0].ndim
        sl[attrs["axis"]] = slice(attrs["start"], attrs["stop"])
        return args[0][tuple(sl)]
    if name == "fused":
        tmp: list[np.ndarray] = []
        for sub, refs, sub_attrs in attrs["body"]:
            vals = [args[k] if kind == "in" else tmp[k] for kind, k in refs]
            tmp.append(apply_op(sub, vals, sub_attrs))
        return tmp[-1]
    raise KeyError(name)


def evaluate(g: Graph, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    vals: dict[int, np.ndarray] = {}
    for n in g.nodes():
        if n.op == "input":
            vals[n.uid] = np.asarray(feeds[n.attrs["name"]], dtype=np.float64)
        elif n.op == "constant":
            vals[n.uid] = n.attrs["value"]
        else:
            vals[n.uid] = apply_op(n.op, [vals[i.uid] for i in n.inputs], n.attrs)
    return [vals[o.uid] for o in g.outputs]


def infer_shapes(g: Graph) -> dict[int, tuple[int, ...]]:
    shp: dict[int, tuple[int, ...]] = {}

    def rule(name: str, s: list[tuple[int, ...]], attrs: dict) -> tuple[int, ...]:
        if name == "matmul":
            assert s[0][-1] == s[1][0], (s[0], s[1])
            return (*s[0][:-1], s[1][-1])
        if name in ELEMENTWISE:
            return tuple(np.broadcast_shapes(*s))
        if name == "layernorm":
            return s[0]
        if name == "sum":
            ax = attrs.get("axis")
            if ax is None:
                return ()
            out = list(s[0])
            if attrs.get("keepdims"):
                out[ax] = 1
            else:
                out.pop(ax)
            return tuple(out)
        if name == "transpose":
            return tuple(reversed(s[0]))
        if name == "concat":
            out = list(s[0])
            out[attrs["axis"]] = sum(x[attrs["axis"]] for x in s)
            return tuple(out)
        if name == "slice":
            out = list(s[0])
            out[attrs["axis"]] = attrs["stop"] - attrs["start"]
            return tuple(out)
        if name == "fused":
            tmp: list[tuple[int, ...]] = []
            for sub, refs, sub_attrs in attrs["body"]:
                tmp.append(rule(sub, [s[k] if kind == "in" else tmp[k] for kind, k in refs], sub_attrs))
            return tmp[-1]
        raise KeyError(name)

    for n in g.nodes():
        if n.op in ("input", "constant"):
            shp[n.uid] = tuple(n.attrs["shape"])
        else:
            shp[n.uid] = rule(n.op, [shp[i.uid] for i in n.inputs], n.attrs)
    return shp


# --------------------------------------------------------------------------- passes

def _rebuild(g: Graph, fn) -> Graph:
    """Map every node in topo order; ``fn(node, new_inputs)`` returns the replacement."""
    new: dict[int, Node] = {}
    for n in g.nodes():
        new[n.uid] = fn(n, [new[i.uid] for i in n.inputs])
    return Graph([new[o.uid] for o in g.outputs])


def constant_fold(g: Graph) -> Graph:
    def fn(n: Node, ins: list[Node]) -> Node:
        if n.op in ("input", "constant"):
            return n
        if all(i.op == "constant" for i in ins):
            return const(apply_op(n.op, [i.attrs["value"] for i in ins], n.attrs))
        return Node(n.op, ins, dict(n.attrs))
    return _rebuild(g, fn)


def cse(g: Graph) -> Graph:
    table: dict[tuple, Node] = {}

    def fn(n: Node, ins: list[Node]) -> Node:
        if n.op in ("input", "constant"):
            return n
        key = (n.op, tuple(i.uid for i in ins), tuple(sorted((k, repr(v)) for k, v in n.attrs.items())))
        if key not in table:
            table[key] = Node(n.op, ins, dict(n.attrs))
        return table[key]
    return _rebuild(g, fn)


def merge_shared_matmuls(g: Graph) -> Graph:
    """TASO-style substitution: matmuls with the same LHS and constant RHS become one matmul."""
    groups: dict[int, list[Node]] = {}
    for n in g.nodes():
        if n.op == "matmul" and n.inputs[1].op == "constant":
            groups.setdefault(n.inputs[0].uid, []).append(n)
    replace: dict[int, Node] = {}
    for mats in groups.values():
        if len(mats) < 2:
            continue
        ws = [m.inputs[1].attrs["value"] for m in mats]
        widths = [m.inputs[1].attrs["shape"][1] for m in mats]
        k = mats[0].inputs[1].attrs["shape"][0]
        wcat = (lazy_const((k, sum(widths))) if any(w is None for w in ws)
                else const(np.concatenate(ws, axis=1)))
        big = Node("matmul", [mats[0].inputs[0], wcat])
        off = 0
        for m, wd in zip(mats, widths):
            replace[m.uid] = Node("slice", [big], {"axis": -1, "start": off, "stop": off + wd})
            off += wd

    def fn(n: Node, ins: list[Node]) -> Node:
        if n.uid in replace:
            return replace[n.uid]
        if n.op in ("input", "constant"):
            return n
        return Node(n.op, ins, dict(n.attrs))
    return _rebuild(g, fn)


def fuse_elementwise(g: Graph, epilogue: bool = False) -> Graph:
    """Greedy producer-consumer fusion of elementwise ops (single-consumer intermediates).

    With ``epilogue`` a group may also start with one matmul (GEMM + bias/activation epilogue).
    """
    cons = g.consumers()
    group_of: dict[int, int] = {}
    members: dict[int, list[Node]] = {}
    has_mm: dict[int, bool] = {}
    for n in g.nodes():
        fusible = n.op in ELEMENTWISE or (epilogue and n.op == "matmul")
        if not fusible:
            continue
        gid = n.uid
        members[gid] = []
        has_mm[gid] = n.op == "matmul"
        if n.op in ELEMENTWISE:
            for i in n.inputs:
                pg = group_of.get(i.uid)
                if pg is None or cons[i.uid] != 1 or members[pg][-1] is not i:
                    continue
                if has_mm[pg] and has_mm[gid]:
                    continue
                members[gid] = members.pop(pg) + members[gid]
                has_mm[gid] = has_mm[gid] or has_mm.pop(pg)
                for m in members[gid]:
                    group_of[m.uid] = gid
        members[gid].append(n)
        group_of[n.uid] = gid

    fused_root = {gid: ms for gid, ms in members.items() if len(ms) > 1}
    inside = {m.uid for ms in fused_root.values() for m in ms}
    new: dict[int, Node] = {}
    for n in g.nodes():
        if n.uid in inside and n.uid not in fused_root:
            continue
        if n.uid in fused_root:
            ms = fused_root[n.uid]
            idx = {m.uid: j for j, m in enumerate(ms)}
            ext: list[Node] = []
            ext_idx: dict[int, int] = {}
            body = []
            for m in ms:
                refs = []
                for i in m.inputs:
                    if i.uid in idx:
                        refs.append(("tmp", idx[i.uid]))
                    else:
                        if i.uid not in ext_idx:
                            ext_idx[i.uid] = len(ext)
                            ext.append(new[i.uid])
                        refs.append(("in", ext_idx[i.uid]))
                body.append((m.op, refs, dict(m.attrs)))
            new[n.uid] = Node("fused", ext, {"body": body, "ops": [m.op for m in ms]})
        elif n.op in ("input", "constant"):
            new[n.uid] = n
        else:
            new[n.uid] = Node(n.op, [new[i.uid] for i in n.inputs], dict(n.attrs))
    return Graph([new[o.uid] for o in g.outputs])


# --------------------------------------------------------------------------- cost model

_FLOPS_PER_EL = {"add": 1, "mul": 1, "relu": 1, "exp": 4, "gelu": 8, "layernorm": 8, "sum": 1}


def kernel_cost(g: Graph, bytes_per_el: int = 2, gpu: mlsys_calc.GPU = mlsys_calc.GPUS["h100"]) -> list[dict]:
    """One row per kernel: HBM bytes = external inputs read once + output written once.

    ``slice`` along the last axis is treated as a view (zero traffic, no launch).
    """
    shp = infer_shapes(g)
    rows = []
    for n in g.kernels():
        if n.op == "slice":
            continue
        out = int(np.prod(shp[n.uid]))
        rd = sum(int(np.prod(shp[i.uid])) for i in n.inputs)
        ops = n.attrs.get("ops", [n.op])
        flops = 0
        if n.op == "fused":
            tmp: list[tuple[int, ...]] = []
            ins = [shp[i.uid] for i in n.inputs]
            for sub, refs, _ in n.attrs["body"]:
                s = [ins[k] if kind == "in" else tmp[k] for kind, k in refs]
                if sub == "matmul":
                    o = (*s[0][:-1], s[1][-1])
                    flops += 2 * int(np.prod(o)) * s[0][-1]
                else:
                    o = tuple(np.broadcast_shapes(*s))
                    flops += _FLOPS_PER_EL.get(sub, 1) * int(np.prod(o))
                tmp.append(o)
        elif n.op == "matmul":
            flops = 2 * out * shp[n.inputs[0].uid][-1]
        else:
            flops = _FLOPS_PER_EL.get(n.op, 1) * out
        nbytes = (rd + out) * bytes_per_el
        t, bound = mlsys_calc.roofline_time(flops, nbytes, gpu)
        rows.append({"kernel": "+".join(ops), "read_MB": rd * bytes_per_el / 1e6,
                     "write_MB": out * bytes_per_el / 1e6, "bytes": nbytes, "flops": flops,
                     "time_us": t * 1e6, "bound": bound})
    return rows


def lazy_const(shape: tuple[int, ...]) -> Node:
    """Constant with a shape but no materialised value (cost-model-only graphs)."""
    return Node("constant", attrs={"value": None, "shape": tuple(shape)})


def _weight(rng: np.random.Generator | None, *shape: int) -> Node:
    return const(rng.normal(0, 0.02, shape)) if rng is not None else lazy_const(shape)


def transformer_mlp(tokens: int, d: int, d_ff: int, rng: np.random.Generator | None = None) -> Graph:
    """y = x + (GeLU(LN(x) W1 + b1) W2 + b2); weights random if ``rng`` else shape-only."""
    x = inp("x", (tokens, d))
    h = op("layernorm", x, _weight(rng, d), _weight(rng, d))
    u = op("gelu", op("add", op("matmul", h, _weight(rng, d, d_ff)), _weight(rng, d_ff)))
    o = op("add", op("matmul", u, _weight(rng, d_ff, d)), _weight(rng, d))
    return Graph([op("add", o, x)])


def qkv_projection(tokens: int, d: int, d_kv: int, rng: np.random.Generator | None = None) -> Graph:
    """Q, K, V = h Wq, h Wk, h Wv after a LayerNorm: the canonical shared-input matmuls."""
    x = inp("x", (tokens, d))
    h = op("layernorm", x, _weight(rng, d), _weight(rng, d))
    return Graph([op("matmul", h, _weight(rng, d, n)) for n in (d, d_kv, d_kv)])
