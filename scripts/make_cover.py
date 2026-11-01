"""Render assets/mlsys-infra-cover.png: three courses, one system stack."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Polygon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BG_TOP = "#04070e"
BG_BOTTOM = "#0a1424"
CREAM = "#f4f1e8"
AMBER = "#e8b45a"
CYAN = "#6fe7dd"
BLUE = "#7aa2f7"
CORAL = "#e06c5a"
MUTED = "#8a97ad"
PANEL = "#081426"
GOLD = "#d9b26a"
RULE = "#233a54"


def pick(names: tuple[str, ...], fallback: str) -> str:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in names:
        if name in installed:
            return name
    return fallback


CJK = pick(("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "PingFang SC"), "DejaVu Sans")
SERIF = pick(("Times New Roman", "Georgia", "Cambria", "DejaVu Serif"), "DejaVu Serif")
MONO = pick(("Consolas", "Cascadia Mono", "DejaVu Sans Mono"), "DejaVu Sans Mono")
plt.rcParams["axes.unicode_minus"] = False


def tracked(text: str) -> str:
    return " ".join(text)


def gradient_rule(ax: plt.Axes, x0: float, x1: float, y: float,
                  colors: list[str], lw: float = 1.2, alpha: float = 0.8,
                  zorder: float = 4) -> None:
    count = 140
    xs = np.linspace(x0, x1, count)
    segments = [[(xs[i], y), (xs[i + 1], y)] for i in range(count - 1)]
    cmap = LinearSegmentedColormap.from_list("rule", colors)
    collection = LineCollection(segments, colors=[cmap(i / (count - 2)) for i in range(count - 1)],
                                linewidths=lw, alpha=alpha, zorder=zorder, capstyle="butt")
    ax.add_collection(collection)


def radial_glow(ax: plt.Axes, cx: float, cy: float, sx: float, sy: float,
                color: str, alpha: float) -> None:
    yy, xx = np.mgrid[0:1:320j, 0:1:320j]
    strength = np.exp(-((((xx - cx) / sx) ** 2) + (((yy - cy) / sy) ** 2)))
    rgba = np.zeros((320, 320, 4))
    rgba[..., :3] = to_rgb(color)
    rgba[..., 3] = strength * alpha
    ax.imshow(rgba, extent=[0, 1, 0, 1], origin="lower", aspect="auto", zorder=0.1)


def draw_tracked(ax: plt.Axes, fig: plt.Figure, x: float, y: float, text: str,
                 fontsize: float, family: str, colors, tracking: float,
                 alpha: float = 1.0, zorder: float = 5,
                 offset: tuple[float, float] = (0.0, 0.0)) -> float:
    cursor = x + offset[0]
    renderer = fig.canvas.get_renderer()
    for index, char in enumerate(text):
        color = colors[index] if isinstance(colors, (list, tuple)) else colors
        item = ax.text(cursor, y + offset[1], char, fontsize=fontsize, family=family,
                       color=color, alpha=alpha, va="baseline", zorder=zorder)
        bounds = item.get_window_extent(renderer).transformed(ax.transData.inverted())
        cursor = bounds.x1 + tracking
    return cursor - tracking


def main() -> None:
    fig = plt.figure(figsize=(12, 6), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")

    gradient = np.linspace(0, 1, 600).reshape(-1, 1)
    ax.imshow(gradient, extent=[0, 1, 0, 1], aspect="auto", origin="lower",
              cmap=LinearSegmentedColormap.from_list("bg", [BG_TOP, BG_BOTTOM]), zorder=0)

    horizon = 0.55
    vanish = (0.78, horizon)
    for x in np.linspace(-0.25, 1.25, 22):
        ax.plot([vanish[0], x], [vanish[1], 1.0], color="#7aa2f7", alpha=0.018,
                lw=0.5, zorder=0.15)
        ax.plot([vanish[0], x], [vanish[1], 0.0], color="#7aa2f7", alpha=0.015,
                lw=0.5, zorder=0.15)
    for i in range(1, 11):
        offset = (i / 10) ** 1.7 * 0.55
        for y in (horizon - offset, horizon + offset):
            ax.plot([0, 1], [y, y], color="#7aa2f7", alpha=0.012, lw=0.5, zorder=0.15)

    radial_glow(ax, 0.80, 0.43, 0.32, 0.34, CYAN, 0.10)
    radial_glow(ax, 0.22, 0.68, 0.38, 0.40, AMBER, 0.05)
    radial_glow(ax, 0.93, 0.18, 0.20, 0.20, CORAL, 0.05)

    ax.plot([0.56, 0.74, 0.96], [0.10, 0.36, 0.36], color=CYAN, alpha=0.10,
            lw=1.1, zorder=0.2)
    ax.plot([0.74, 0.74], [0.10, 0.36], color=CYAN, alpha=0.08, lw=0.6,
            ls=(0, (2, 3)), zorder=0.2)


    for x, name, color in (
        (0.055, "STANFORD CS149", AMBER),
        (0.175, "STANFORD CS229S", CYAN),
        (0.300, "CMU 15-849", CORAL),
    ):
        ax.add_patch(Circle((x, 0.928), 0.0040, facecolor=color, edgecolor="none", zorder=4))
        ax.text(x + 0.011, 0.928, name, color=CREAM, fontsize=9.0, family=MONO, va="baseline")
    ax.text(0.068, 0.898, "SELF-STUDY REPOSITORY  /  INTERVIEW PREP",
            color=MUTED, fontsize=7.5, family=MONO, va="baseline")
    ax.text(0.945, 0.928, "2026  ·  MLSYS-INFRA", color=MUTED, fontsize=7.5,
            family=MONO, ha="right", va="baseline")
    gradient_rule(ax, 0.055, 0.945, 0.872, [AMBER, CYAN, CORAL], lw=0.8, alpha=0.38)

    ax.text(0.055, 0.815, tracked("MACHINE LEARNING SYSTEMS"), color="#a9b6c7",
            fontsize=10.0, family=MONO, va="baseline")

    title = ax.text(0.055, 0.702, "MLSys", color=CREAM, fontsize=74, fontweight="bold",
                    family=SERIF, va="center", zorder=5)
    renderer = fig.canvas.get_renderer()
    box = title.get_window_extent(renderer).transformed(ax.transData.inverted())
    inf = ax.text(box.x1 + 0.016, 0.702, "Infra", color=CREAM, fontsize=74,
                  fontweight="bold", family=SERIF, va="center", zorder=5)
    ibox = inf.get_window_extent(renderer).transformed(ax.transData.inverted())
    px, py = 0.011, 0.013
    gx, gy = ibox.x0 - px, ibox.y0 - py
    gw, gh = ibox.width + 2 * px, ibox.height + 2 * py
    for lw, alpha in ((7.0, 0.04), (3.2, 0.09), (1.4, 0.26)):
        ax.add_patch(FancyBboxPatch((gx, gy), gw, gh,
                                    boxstyle="round,pad=0,rounding_size=0.009",
                                    facecolor="none", edgecolor=CYAN, lw=lw,
                                    alpha=alpha, zorder=2.0))
    ax.add_patch(FancyBboxPatch((gx, gy), gw, gh,
                                boxstyle="round,pad=0,rounding_size=0.009",
                                facecolor=(0.04, 0.10, 0.18, 0.55), edgecolor="#a7f3d0",
                                lw=0.7, alpha=0.85, zorder=2.1))
    ax.plot([gx + 0.010, gx + gw - 0.010], [gy + gh - 0.002, gy + gh - 0.002],
            color="white", alpha=0.28, lw=0.5, zorder=2.2)

    gradient_rule(ax, 0.055, 0.430, 0.600, [CYAN, GOLD], lw=1.4, alpha=0.85)
    ax.add_patch(Circle((0.055, 0.532), 0.0026, facecolor=GOLD, edgecolor="none", zorder=5))
    draw_tracked(ax, fig, 0.066, 0.525, "面向面试的", 15.5, CJK, "#c9d3e0", 0.010)
    draw_tracked(ax, fig, 0.055, 0.445, "机器学习系统", 38, "STSong", GOLD, 0.016,
                 alpha=0.14, zorder=4, offset=(0.0025, -0.0025))
    end_x = draw_tracked(ax, fig, 0.055, 0.445, "机器学习系统", 38, "STSong",
                         [CREAM, CREAM, CREAM, CREAM, GOLD, GOLD], 0.016)
    ax.plot([0.055, end_x], [0.424, 0.424], color=GOLD, alpha=0.55, lw=0.7, zorder=4)
    ax.add_patch(Polygon([[end_x + 0.009, 0.424], [end_x + 0.013, 0.428],
                          [end_x + 0.017, 0.424], [end_x + 0.013, 0.420]],
                         closed=True, facecolor=GOLD, edgecolor="none", zorder=4))
    ax.plot([0.043, 0.043], [0.435, 0.535], color=GOLD, alpha=0.45, lw=1.0, zorder=4)
    ax.add_patch(Polygon([[0.043, 0.540], [0.046, 0.536], [0.043, 0.532], [0.040, 0.536]],
                         closed=True, facecolor=GOLD, edgecolor="none", zorder=4))
    ax.text(0.055, 0.362, "三门课，一条从硬件到服务的系统栈。", color="#c9d3e0",
            fontsize=11.0, family=CJK, va="baseline")
    ax.text(0.055, 0.324, "18 讲笔记 · 22 条公式 · 10 个实验 · 11 份报告",
            color=MUTED, fontsize=9.5, family=CJK, va="baseline")

    stats = (
        ("18", tracked("LECTURES"), "讲系统笔记", AMBER),
        ("10", tracked("LABS"), "个可复现实验", CYAN),
        ("11", tracked("REPORTS"), "份中文报告", CORAL),
    )
    for i, (number, label, cn, accent) in enumerate(stats):
        sx = 0.055 + i * 0.178
        ax.text(sx, 0.205, number, color=CREAM, fontsize=30, fontweight="bold",
                family=SERIF, va="baseline")
        ax.text(sx + 0.049, 0.205, label, color=accent, fontsize=7.5,
                family=MONO, va="baseline")
        ax.text(sx, 0.162, cn, color=MUTED, fontsize=8.5, family=CJK, va="baseline")
        if i < 2:
            gradient_rule(ax, sx + 0.160, sx + 0.160, 0.188, [accent, MUTED],
                          lw=0.6, alpha=0.25, zorder=3)
            ax.plot([sx + 0.1585, sx + 0.1585], [0.146, 0.224], color=accent,
                    alpha=0.25, lw=0.6, zorder=3)

    ax.text(0.055, 0.045, "SHANELIU04  ·  PERSONAL STUDY REPOSITORY  ·  MIT LICENSE",
            color="#5f7188", fontsize=7.5, family=MONO, va="baseline")
    ax.text(0.945, 0.045, "GITHUB.COM/SHANELIU04/MLSYS-INFRA",
            color="#5f7188", fontsize=7.5, family=MONO, ha="right", va="baseline")

    ax.add_patch(FancyBboxPatch((0.604, 0.074), 0.360, 0.770,
                                boxstyle="round,pad=0,rounding_size=0.012",
                                facecolor="black", edgecolor="none", alpha=0.38, zorder=1))
    ax.add_patch(FancyBboxPatch((0.600, 0.080), 0.360, 0.770,
                                boxstyle="round,pad=0,rounding_size=0.012",
                                facecolor=PANEL, edgecolor="none", alpha=0.62, zorder=1.1))
    for lw, alpha in ((5.0, 0.05), (2.2, 0.10), (0.9, 0.30)):
        ax.add_patch(FancyBboxPatch((0.600, 0.080), 0.360, 0.770,
                                    boxstyle="round,pad=0,rounding_size=0.012",
                                    facecolor="none", edgecolor=BLUE, lw=lw,
                                    alpha=alpha, zorder=1.2))
    ax.add_patch(FancyBboxPatch((0.600, 0.080), 0.360, 0.770,
                                boxstyle="round,pad=0,rounding_size=0.012",
                                facecolor="none", edgecolor="#a7f3d0",
                                lw=0.5, alpha=0.25, zorder=1.3))

    for gx in np.arange(0.620, 0.960, 0.045):
        ax.plot([gx, gx], [0.090, 0.840], color="white", alpha=0.018, lw=0.4, zorder=1.15)
    for gy in np.arange(0.100, 0.850, 0.045):
        ax.plot([0.605, 0.955], [gy, gy], color="white", alpha=0.018, lw=0.4, zorder=1.15)

    ax.text(0.615, 0.800, "COURSE → LAYER", color="#a7f3d0", fontsize=10.0,
            family=MONO, va="baseline", zorder=5)
    ax.text(0.615, 0.770, "三门课程 × 五层系统栈", color="#c9d3e0", fontsize=9.5,
            family=CJK, va="baseline", zorder=5)
    ax.text(0.945, 0.800, "18 MODULES", color=MUTED, fontsize=8.5,
            family=MONO, ha="right", va="baseline", zorder=5)
    ax.plot([0.615, 0.945], [0.748, 0.748], color=GOLD, alpha=0.25, lw=0.6, zorder=3)

    cards = (
        (0.615, "CS149", "L01–L02", AMBER),
        (0.728, "CS229S", "L02 / L05", CYAN),
        (0.841, "15-849", "L03–L05", CORAL),
    )
    for cx, code, layers, accent in cards:
        ax.add_patch(FancyBboxPatch((cx, 0.678), 0.105, 0.058,
                                    boxstyle="round,pad=0,rounding_size=0.006",
                                    facecolor=(0.04, 0.10, 0.18, 0.75),
                                    edgecolor=accent, lw=0.8, alpha=0.95, zorder=3))
        ax.add_patch(Circle((cx + 0.012, 0.722), 0.0040, facecolor=accent,
                            edgecolor="none", zorder=4))
        ax.text(cx + 0.022, 0.714, code, color=accent, fontsize=9.5,
                fontweight="bold", family=MONO, va="baseline", zorder=4)
        ax.text(cx + 0.022, 0.690, layers, color="#c9d3e0", fontsize=8.5,
                family=MONO, va="baseline", zorder=4)

    rows = (
        (0.605, "L01", "硬件与并行", "M01–M06", (("CS149", AMBER),)),
        (0.500, "L02", "单卡算子", "M07–M09", (("CS149", AMBER), ("CS229S", CYAN))),
        (0.395, "L03", "编译与框架", "M10–M11", (("15-849", CORAL),)),
        (0.290, "L04", "训练系统", "M12–M14", (("15-849", CORAL),)),
        (0.185, "L05", "推理与服务", "M15–M18", (("CS229S", CYAN), ("15-849", CORAL))),
    )
    for y, lid, label, modules, pills in rows:
        ax.add_patch(FancyBboxPatch((0.618, y - 0.046), 0.330, 0.085,
                                    boxstyle="round,pad=0,rounding_size=0.006",
                                    facecolor="black", edgecolor="none",
                                    alpha=0.25, zorder=2.1))
        ax.add_patch(FancyBboxPatch((0.615, y - 0.0425), 0.330, 0.085,
                                    boxstyle="round,pad=0,rounding_size=0.006",
                                    facecolor=(0.04, 0.10, 0.18, 0.55),
                                    edgecolor="#7aa2f7", lw=0.4, alpha=0.75, zorder=2.2))
        ax.plot([0.625, 0.935], [y + 0.039, y + 0.039], color="white",
                alpha=0.10, lw=0.4, zorder=2.4)
        ax.plot([0.6155, 0.6155], [y - 0.039, y + 0.039], color=GOLD,
                alpha=0.35, lw=1.2, zorder=2.6)
        ax.plot([0.747, 0.747], [y - 0.038, y + 0.038], color=RULE,
                alpha=0.35, lw=0.4, zorder=2.8)
        ax.text(0.626, y + 0.018, lid, color=GOLD, fontsize=9.0,
                family=MONO, va="baseline", zorder=4)
        ax.text(0.668, y + 0.010, label, color=CREAM, fontsize=11.5,
                fontweight="bold", family=CJK, va="baseline", zorder=4)
        ax.text(0.935, y + 0.018, modules, color=MUTED, fontsize=9.0,
                family=MONO, ha="right", va="baseline", zorder=4)
        px = 0.755
        for code, accent in pills:
            width = {"CS149": 0.062, "CS229S": 0.072, "15-849": 0.066}[code]
            ax.add_patch(FancyBboxPatch((px, y - 0.038), width, 0.032,
                                        boxstyle="round,pad=0,rounding_size=0.008",
                                        facecolor=(0.04, 0.10, 0.18, 0.85),
                                        edgecolor=accent, lw=0.6, alpha=0.95, zorder=3.5))
            ax.add_patch(Circle((px + 0.010, y - 0.022), 0.0025, facecolor=accent,
                                edgecolor="none", zorder=4))
            ax.text(px + 0.017, y - 0.022, code, color=accent, fontsize=8.5,
                    fontweight="bold", family=MONO, va="center", zorder=4)
            px += width + 0.008

    ax.text(0.780, 0.108, "M01 — M18  /  HARDWARE → SERVING", color=MUTED,
            fontsize=8.5, family=MONO, ha="center", va="baseline", zorder=4)

    out = ROOT / "assets" / "mlsys-infra-cover.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=BG_BOTTOM, pil_kwargs={"optimize": True})
    print(out)


if __name__ == "__main__":
    main()
