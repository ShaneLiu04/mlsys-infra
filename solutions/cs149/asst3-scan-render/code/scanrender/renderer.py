"""Ordered semi-transparent circle renderer (CS149 asst3 ``render``).

Requirements kept from the assignment: (1) atomicity — each pixel update is a
read-modify-write; (2) order — for every pixel, circles are blended in input
order. ``over`` blending ``c <- a*rgb + (1-a)*c`` is not commutative, so a
renderer that parallelizes over circles without ordering gives a wrong image.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scan import exclusive_scan


@dataclass
class Scene:
    centers: np.ndarray  # (N, 2) float32, (x, y) in pixel units
    radii: np.ndarray    # (N,) float32
    colors: np.ndarray   # (N, 3) float32
    alphas: np.ndarray   # (N,) float32

    @property
    def n(self) -> int:
        return int(self.radii.size)

    def permuted(self, perm: np.ndarray) -> "Scene":
        return Scene(self.centers[perm], self.radii[perm], self.colors[perm], self.alphas[perm])


@dataclass
class RenderStats:
    circle_pixel_tests: int = 0
    bbox_tests: int = 0
    pairs: int = 0


def make_scene(kind: str, size: int, seed: int = 0) -> Scene:
    """``few_big``: 16 circles r~U(40,140)·size/512; ``many_small``: 8000 circles r~U(1.5,5);
    ``mixed``: 2000 circles with log-uniform radius 2..40."""
    rng = np.random.default_rng(seed)
    s = size / 512.0
    if kind == "few_big":
        n = 16
        r = rng.uniform(40, 140, n) * s
    elif kind == "many_small":
        n = 8000
        r = rng.uniform(1.5, 5.0, n) * s
    elif kind == "mixed":
        n = 2000
        r = np.exp(rng.uniform(np.log(2), np.log(40), n)) * s
    else:
        raise ValueError(kind)
    centers = rng.uniform(0, size, (n, 2))
    colors = rng.uniform(0, 1, (n, 3))
    alphas = rng.uniform(0.3, 0.8, n)
    f = np.float32
    return Scene(centers.astype(f), r.astype(f), colors.astype(f), alphas.astype(f))


def blank_image(height: int, width: int) -> np.ndarray:
    return np.ones((height, width, 3), dtype=np.float32)


def _inside(px: np.ndarray, py: np.ndarray, cx: np.float32, cy: np.float32, r: np.float32) -> np.ndarray:
    dx = px - cx
    dy = py - cy
    return dx * dx + dy * dy <= r * r


def _blend(dst: np.ndarray, rgb: np.ndarray, alpha: np.float32) -> np.ndarray:
    return alpha * rgb + (np.float32(1.0) - alpha) * dst


def _pixel_centers(y0: int, y1: int, x0: int, x1: int) -> tuple[np.ndarray, np.ndarray]:
    ys = np.arange(y0, y1, dtype=np.float32) + np.float32(0.5)
    xs = np.arange(x0, x1, dtype=np.float32) + np.float32(0.5)
    return np.meshgrid(xs, ys)


def circle_bboxes(scene: Scene, height: int, width: int) -> np.ndarray:
    """Clipped integer pixel bounding boxes (x0, x1, y0, y1), half-open, shape (N, 4)."""
    cx, cy, r = scene.centers[:, 0], scene.centers[:, 1], scene.radii
    x0 = np.clip(np.floor(cx - r), 0, width).astype(np.int64)
    x1 = np.clip(np.ceil(cx + r) + 1, 0, width).astype(np.int64)
    y0 = np.clip(np.floor(cy - r), 0, height).astype(np.int64)
    y1 = np.clip(np.ceil(cy + r) + 1, 0, height).astype(np.int64)
    return np.stack([x0, x1, y0, y1], axis=1)


def render_reference(scene: Scene, height: int, width: int) -> tuple[np.ndarray, RenderStats]:
    """Sequential: for each circle in order, blend every pixel of its bounding box that it covers."""
    img = blank_image(height, width)
    st = RenderStats()
    for k, (x0, x1, y0, y1) in enumerate(circle_bboxes(scene, height, width)):
        if x0 >= x1 or y0 >= y1:
            continue
        px, py = _pixel_centers(y0, y1, x0, x1)
        m = _inside(px, py, scene.centers[k, 0], scene.centers[k, 1], scene.radii[k])
        st.circle_pixel_tests += m.size
        st.pairs += 1
        region = img[y0:y1, x0:x1]
        region[m] = _blend(region[m], scene.colors[k], scene.alphas[k])
    return img, st


def brute_force_tests(scene: Scene, height: int, width: int) -> int:
    """Circle-pixel tests if every pixel tested every circle (no culling at all)."""
    return scene.n * height * width


def render_unordered(scene: Scene, height: int, width: int, seed: int) -> np.ndarray:
    """Parallel-by-circle without ordering, emulated as one random interleaving of circle updates."""
    perm = np.random.default_rng(seed).permutation(scene.n)
    return render_reference(scene.permuted(perm), height, width)[0]


def bin_circles(scene: Scene, height: int, width: int, tile: int) -> tuple[list[np.ndarray], RenderStats]:
    """Per-tile ordered circle lists via bbox-overlap flags -> exclusive scan -> scatter."""
    bb = circle_bboxes(scene, height, width)
    tx_n = (width + tile - 1) // tile
    ty_n = (height + tile - 1) // tile
    ids = np.arange(scene.n, dtype=np.int64)
    lists: list[np.ndarray] = []
    st = RenderStats()
    nonempty = (bb[:, 0] < bb[:, 1]) & (bb[:, 2] < bb[:, 3])
    for ty in range(ty_n):
        ty0, ty1 = ty * tile, min((ty + 1) * tile, height)
        row_hit = nonempty & (bb[:, 2] < ty1) & (bb[:, 3] > ty0)
        for tx in range(tx_n):
            tx0, tx1 = tx * tile, min((tx + 1) * tile, width)
            flags = row_hit & (bb[:, 0] < tx1) & (bb[:, 1] > tx0)
            st.bbox_tests += scene.n
            pos = exclusive_scan(flags.astype(np.int64))
            count = int(pos[-1] + flags[-1]) if scene.n else 0
            lst = np.empty(count, dtype=np.int64)
            lst[pos[flags]] = ids[flags]
            lists.append(lst)
    return lists, st


def render_tiled(scene: Scene, height: int, width: int, tile: int = 16) -> tuple[np.ndarray, RenderStats]:
    """Tile-binned renderer: each tile is independent (parallel over tiles); within a tile,
    every pixel blends its tile's circles in input order, so the result equals the reference."""
    img = blank_image(height, width)
    lists, st = bin_circles(scene, height, width, tile)
    tx_n = (width + tile - 1) // tile
    for t, lst in enumerate(lists):
        if lst.size == 0:
            continue
        ty, tx = divmod(t, tx_n)
        y0, y1 = ty * tile, min((ty + 1) * tile, height)
        x0, x1 = tx * tile, min((tx + 1) * tile, width)
        px, py = _pixel_centers(y0, y1, x0, x1)
        region = img[y0:y1, x0:x1]
        for k in lst:
            m = _inside(px, py, scene.centers[k, 0], scene.centers[k, 1], scene.radii[k])
            st.circle_pixel_tests += m.size
            st.pairs += 1
            if m.any():
                region[m] = _blend(region[m], scene.colors[k], scene.alphas[k])
    return img, st
