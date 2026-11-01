"""Tiny reverse-mode autodiff over numpy tensors (topological-order backward)."""

from __future__ import annotations

from typing import Callable

import numpy as np


def _unbroadcast(g: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum ``g`` over axes that were broadcast to produce it from ``shape``."""
    while g.ndim > len(shape):
        g = g.sum(0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(i, keepdims=True)
    return g


class Tensor:
    def __init__(self, data, parents: tuple["Tensor", ...] = (), op: str = "") -> None:
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._parents = parents
        self._backward: Callable[[], None] = lambda: None
        self.op = op

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, op={self.op!r})"

    # ------------------------------------------------------------------ ops

    def __add__(self, other: "Tensor | float") -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "add")

        def bw() -> None:
            self.grad += _unbroadcast(out.grad, self.shape)
            other.grad += _unbroadcast(out.grad, other.shape)
        out._backward = bw
        return out

    __radd__ = __add__

    def __mul__(self, other: "Tensor | float") -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "mul")

        def bw() -> None:
            self.grad += _unbroadcast(out.grad * other.data, self.shape)
            other.grad += _unbroadcast(out.grad * self.data, other.shape)
        out._backward = bw
        return out

    __rmul__ = __mul__

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __sub__(self, other: "Tensor | float") -> "Tensor":
        return self + (-other if isinstance(other, Tensor) else -float(other))

    def __matmul__(self, other: "Tensor") -> "Tensor":
        out = Tensor(self.data @ other.data, (self, other), "matmul")

        def bw() -> None:
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = bw
        return out

    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(self.data, 0), (self,), "relu")

        def bw() -> None:
            self.grad += out.grad * (self.data > 0)
        out._backward = bw
        return out

    def exp(self) -> "Tensor":
        out = Tensor(np.exp(self.data), (self,), "exp")

        def bw() -> None:
            self.grad += out.grad * out.data
        out._backward = bw
        return out

    def log(self) -> "Tensor":
        out = Tensor(np.log(self.data), (self,), "log")

        def bw() -> None:
            self.grad += out.grad / self.data
        out._backward = bw
        return out

    def sum(self, axis: int | None = None, keepdims: bool = False) -> "Tensor":
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def bw() -> None:
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += np.broadcast_to(g, self.shape)
        out._backward = bw
        return out

    def softmax_cross_entropy(self, labels: np.ndarray) -> "Tensor":
        """Mean cross-entropy of logits [n, c] against integer labels; fused for stability."""
        z = self.data - self.data.max(1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(1, keepdims=True))
        n = len(labels)
        out = Tensor(-logp[np.arange(n), labels].mean(), (self,), "softmax_xent")

        def bw() -> None:
            p = np.exp(logp)
            p[np.arange(n), labels] -= 1.0
            self.grad += out.grad * p / n
        out._backward = bw
        return out

    # ------------------------------------------------------------------ backward

    def backward(self) -> None:
        order: list[Tensor] = []
        seen: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:  # iterative DFS post-order; deep graphs would overflow recursion
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if id(p) not in seen:
                    stack.append((p, False))
        self.grad = np.ones_like(self.data)
        for node in reversed(order):
            node._backward()


def numerical_grad(f: Callable[[], float], x: Tensor, eps: float = 1e-6) -> np.ndarray:
    g = np.zeros_like(x.data)
    it = np.nditer(x.data, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x.data[i]
        x.data[i] = old + eps
        fp = f()
        x.data[i] = old - eps
        fm = f()
        x.data[i] = old
        g[i] = (fp - fm) / (2 * eps)
    return g
