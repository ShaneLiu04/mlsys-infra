"""Mandelbrot iteration counts (prog1) and row-assignment schedules.

Semantics follow the CS149 starter ``mandel()``: test ``|z|^2 > 4`` first, then
update, and return the number of completed updates (at most ``max_iter``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VIEW1 = (-2.0, -1.0, 1.0, 1.0)
VIEW2 = (-1.5 + 0.21, -1.0 + 0.75, -1.5 + 0.21 + 0.6, -1.0 + 0.75 + 0.4)


@dataclass(frozen=True)
class MandelSpec:
    width: int = 1600
    height: int = 1200
    max_iter: int = 256
    view: tuple[float, float, float, float] = VIEW1


def mandel_scalar(c_re: float, c_im: float, max_iter: int) -> int:
    """Pure-Python scalar kernel, identical control flow to the C++ starter."""
    z_re, z_im = c_re, c_im
    i = 0
    while i < max_iter:
        if z_re * z_re + z_im * z_im > 4.0:
            break
        new_re = z_re * z_re - z_im * z_im
        new_im = 2.0 * z_re * z_im
        z_re = c_re + new_re
        z_im = c_im + new_im
        i += 1
    return i


def row_coords(spec: MandelSpec, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Complex-plane coordinates (float32) of every pixel in ``rows``."""
    x0, y0, x1, y1 = spec.view
    dx = np.float32((x1 - x0) / spec.width)
    dy = np.float32((y1 - y0) / spec.height)
    xs = np.float32(x0) + np.arange(spec.width, dtype=np.float32) * dx
    ys = np.float32(y0) + rows.astype(np.float32) * dy
    c_re = np.broadcast_to(xs, (rows.size, spec.width)).ravel()
    c_im = np.repeat(ys, spec.width)
    return np.ascontiguousarray(c_re), c_im


def mandel_points(c_re: np.ndarray, c_im: np.ndarray, max_iter: int) -> np.ndarray:
    """Vectorized kernel with active-set compaction.

    Escaped points are dropped every iteration, so the cost is proportional to
    the total number of iterations (the real work), not to ``N * max_iter``.
    """
    n = c_re.size
    out = np.full(n, max_iter, dtype=np.int32)
    idx = np.arange(n)
    cr = c_re.astype(np.float32, copy=True)
    ci = c_im.astype(np.float32, copy=True)
    zr = cr.copy()
    zi = ci.copy()
    for it in range(max_iter):
        zr2 = zr * zr
        zi2 = zi * zi
        esc = (zr2 + zi2) > np.float32(4.0)
        if esc.any():
            out[idx[esc]] = it
            keep = ~esc
            idx, cr, ci, zr, zi = idx[keep], cr[keep], ci[keep], zr[keep], zi[keep]
            zr2, zi2 = zr2[keep], zi2[keep]
            if idx.size == 0:
                break
        new_im = np.float32(2.0) * zr * zi
        zr = cr + (zr2 - zi2)
        zi = ci + new_im
    return out


def mandel_points_masked(c_re: np.ndarray, c_im: np.ndarray, max_iter: int) -> np.ndarray:
    """Vectorized kernel *without* compaction: one gang of width N, lanes masked off.

    Runs until every lane is done, so cost ~ N * max(iterations). This is the
    numpy picture of SIMD divergence with gang width = array length.
    """
    cr = c_re.astype(np.float32)
    ci = c_im.astype(np.float32)
    zr, zi = cr.copy(), ci.copy()
    count = np.zeros(cr.size, dtype=np.int32)
    active = np.ones(cr.size, dtype=bool)
    for _ in range(max_iter):
        active &= (zr * zr + zi * zi) <= np.float32(4.0)
        if not active.any():
            break
        new_re = zr * zr - zi * zi
        new_im = np.float32(2.0) * zr * zi
        zr = np.where(active, cr + new_re, zr)
        zi = np.where(active, ci + new_im, zi)
        count += active
    return count


BLOCK_POINTS = 16384


def mandel_rows(spec: MandelSpec, rows: np.ndarray, block_points: int | None = BLOCK_POINTS) -> np.ndarray:
    """Iteration counts for the given rows, shape (len(rows), width).

    Rows are strip-mined into kernel calls of about ``block_points`` pixels so the
    numpy temporaries stay cache-resident; without this, per-pixel cost depends on
    how many rows a task owns and schedules cannot be compared fairly.
    """
    rows = np.asarray(rows)
    step = rows.size if block_points is None else max(1, block_points // spec.width)
    out = np.empty((rows.size, spec.width), dtype=np.int32)
    for s in range(0, rows.size, step):
        r = rows[s:s + step]
        c_re, c_im = row_coords(spec, r)
        out[s:s + r.size] = mandel_points(c_re, c_im, spec.max_iter).reshape(r.size, spec.width)
    return out


def mandel_image(spec: MandelSpec, block_points: int | None = BLOCK_POINTS) -> np.ndarray:
    return mandel_rows(spec, np.arange(spec.height), block_points)


def static_blocks(height: int, workers: int) -> list[np.ndarray]:
    """Contiguous row blocks, one per worker (the starter's naive split)."""
    return [b for b in np.array_split(np.arange(height), workers)]


def interleaved_rows(height: int, workers: int) -> list[np.ndarray]:
    """Worker w gets rows w, w+P, w+2P, ... (static, but spreads the expensive middle)."""
    return [np.arange(w, height, workers) for w in range(workers)]


def dynamic_chunks(height: int, chunk_rows: int) -> list[np.ndarray]:
    """Small contiguous chunks handed out on demand by a work queue."""
    return [np.arange(s, min(s + chunk_rows, height)) for s in range(0, height, chunk_rows)]


def assignment_work(counts: np.ndarray, assignment: list[np.ndarray]) -> np.ndarray:
    """Total iterations (+1 per pixel for the final test) in each part of an assignment."""
    row_work = (counts.astype(np.int64) + 1).sum(axis=1)
    return np.array([row_work[a].sum() for a in assignment], dtype=np.int64)
