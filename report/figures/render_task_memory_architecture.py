"""Render documentation figures without importing the model or using a GPU."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parent
INK = "#15263c"
MUTED = "#50627a"
BLUE = ("#eaf3ff", "#3378ba")
ORANGE = ("#fff2e3", "#c57b27")
GREEN = ("#eaf7ee", "#43895a")
PURPLE = ("#f0ebff", "#7860b5")
GRAY = ("#f0f3f7", "#8895a5")
YELLOW = ("#fff8d9", "#aa8c2a")

plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})


def canvas(width, height, title, subtitle):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, width), ylim=(0, height))
    ax.axis("off")
    ax.text(.5, height - .48, title, fontsize=23, fontweight="bold", color=INK, va="top")
    ax.text(.5, height - .98, subtitle, fontsize=11, color=MUTED, va="top")
    return fig, ax


def box(ax, x, y, w, h, title, body, palette, size=11):
    fill, edge = palette
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.025,rounding_size=0.11",
                               linewidth=1.5, edgecolor=edge, facecolor=fill, zorder=3))
    ax.text(x + w / 2, y + h - .24, title, fontsize=size + .6, fontweight="bold",
            color=INK, ha="center", va="top", zorder=4)
    ax.text(x + w / 2, y + h / 2 - .15, body, fontsize=size, color=INK,
            ha="center", va="center", linespacing=1.55, zorder=4)


def arrow(ax, points, color=MUTED, style="-", label=None, label_xy=None):
    for p, q in zip(points[:-2], points[1:-1]):
        ax.plot((p[0], q[0]), (p[1], q[1]), color=color, linewidth=1.65,
                linestyle=style, zorder=2)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                mutation_scale=14, linewidth=1.65, color=color,
                                linestyle=style, shrinkA=0, shrinkB=3, zorder=2))
    if label:
        ax.text(*label_xy, label, color=color, fontsize=10, ha="center", va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 2}, zorder=5)


def save(fig, name):
    for suffix in ("svg", "png"):
        fig.savefig(OUT / f"{name}.{suffix}", dpi=160, facecolor="white")
    plt.close(fig)


def overview():
    fig, ax = canvas(18, 10.5, "PARA-SSR  |  Task-memory planner",
                     "Interaction ON · Camera-only baseline · B = batch size · C = 256 · Forward feature flow")

    box(ax, .5, 7.35, 3.0, 1.35, "History: t - 0.5 s",
        "Camera → BEV\nSame encoders, no_grad", GRAY)
    box(ax, 4.3, 7.35, 3.1, 1.35, "SE(2) history alignment",
        "Translation + relative yaw\nFull-feature grid_sample", BLUE)
    arrow(ax, [(3.52, 8.05), (4.28, 8.05)])
    ax.text(5.85, 7.06, "Pose metadata is detached", fontsize=10, color=MUTED, ha="center",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 2}, zorder=5)

    box(ax, .5, 4.6, 3, 1.45, "Current front cameras",
        "3 views · 768 × 416 each\nResNet-50 + single-level FPN", BLUE)
    box(ax, .5, 2.45, 3, 1.5, "Learned BEV queries",
        "Embedding [5000, 256]\nuse_lidar = false", BLUE, size=10.5)
    box(ax, 4.3, 3.8, 3.1, 2.5, "Shared BEV encoder × 3",
        "Temporal self-attention\nCamera cross-attention\nFFN\n\nBEV [B, 5000, 256]", BLUE)
    arrow(ax, [(3.52, 5.3), (4.28, 5.3)])
    arrow(ax, [(3.52, 3.2), (3.9, 3.2), (3.9, 4.3), (4.28, 4.3)])
    arrow(ax, [(5.85, 7.32), (5.85, 6.33)])
    ax.text(5.85, 3.36, "50 × 100 cells · 0.64 m / cell", ha="center", fontsize=10, color=MUTED)

    box(ax, 8.6, 6.85, 3.7, 1.25, "Detection → motion",
        "Det decoder × 3 → motion decoder\n300 object slots × 6 modes", ORANGE, size=10.6)
    box(ax, 8.6, 5.25, 3.7, 1.22, "Det/motion memory",
        "Mean modes + projections + LN\n[B, 300, 256]", ORANGE)
    box(ax, 8.6, 2.55, 3.7, 1.15, "Map decoder × 3",
        "100 instances × 20 points", GREEN)
    box(ax, 8.6, 3.97, 3.7, 1.03, "Map memory",
        "Mean points + projection + LN\n[B, 100, 256]", GREEN, size=10.6)
    arrow(ax, [(7.42, 5.72), (7.92, 5.72), (7.92, 7.48), (8.58, 7.48)], ORANGE[1])
    arrow(ax, [(7.42, 4.3), (7.92, 4.3), (7.92, 3.12), (8.58, 3.12)], GREEN[1])
    arrow(ax, [(10.45, 6.82), (10.45, 6.5)], ORANGE[1])
    arrow(ax, [(10.45, 3.73), (10.45, 3.94)], GREEN[1])

    box(ax, 13.5, 4.0, 3.9, 3.9, "Planning decoder × 3",
        "Hidden [B, 1, 256]\n\nBEV → parallel det + map\n→ residual sum → FFN\nPre-LN + residual\n\nOnly planning hidden updates\nMemories are reused", PURPLE, size=11.1)
    arrow(ax, [(7.42, 6.05), (7.7, 6.05), (7.7, 8.88), (12.9, 8.88),
               (12.9, 7.26), (13.48, 7.26)], BLUE[1], label="Dense BEV: K / V",
          label_xy=(10.45, 8.88))
    arrow(ax, [(12.32, 5.87), (13.48, 5.87)], ORANGE[1])
    arrow(ax, [(12.32, 4.49), (13.48, 4.49)], GREEN[1])

    box(ax, 8.6, .9, 3.7, 1.2, "Initial planning hidden",
        "Learned query + command fuser\n+ MLP(vx, vy, ax, ay)", YELLOW, size=10.5)
    arrow(ax, [(12.32, 1.52), (12.82, 1.52), (12.82, 3.6), (14.25, 3.6),
               (14.25, 3.98)], YELLOW[1])
    box(ax, 13.5, .9, 3.9, 1.68, "Final LN → trajectory MLP",
        "8 × (Δx, Δy, Δheading)\nCumulative sum → [B, 8, 3]\n4 seconds · 0.5-second intervals", PURPLE, size=10.7)
    arrow(ax, [(15.75, 3.98), (15.75, 2.61)], PURPLE[1])

    ax.text(.5, 1.58, "Task-memory content stays attached to autograd.", fontsize=11, color=INK)
    ax.text(.5, 1.16, "Detached box / polyline positions + scores enter keys only.", fontsize=10.7, color=MUTED)
    ax.text(.5, .56, "No Scene TokenLearner · No planner gate · No det ↔ map decoder dependency",
            fontsize=10.2, color=MUTED)
    save(fig, "17_task_memory_overview")


def planner_layer():
    fig, ax = canvas(18, 10.5, "Inside one planning layer  |  Parallel task attention",
                     "Repeat 3 times with different parameters · Both task branches read the same post-BEV hidden")
    box(ax, .5, 5.0, 2.0, 1.25, "Input hidden h", "[B, 1, 256]", YELLOW)
    box(ax, 3.15, 5.0, 2.85, 1.25, "BEV cross-attention",
        "Query Pre-LN → attention\n+ input residual", BLUE, size=10.3)
    box(ax, 3.15, 7.6, 2.85, 1.12, "BEV K / V",
        "Dense BEV\n[B, 5000, 256]", BLUE, size=10.8)
    arrow(ax, [(4.575, 7.57), (4.575, 6.28)], BLUE[1])
    arrow(ax, [(2.53, 5.625), (3.12, 5.625)])
    arrow(ax, [(6.03, 5.625), (7.85, 5.625)], label="h_bev", label_xy=(6.7, 5.88))

    box(ax, 7.25, 7.45, 3.15, 1.25, "Plan-det cross-attention",
        "Q = det_LN(h_bev) + plan_pos\nOutput: det_update", ORANGE, size=10.3)
    box(ax, 11.2, 7.45, 3.05, 1.25, "Det/motion K / V",
        "Fixed task memory\n[B, 300, 256]", ORANGE, size=10.5)
    arrow(ax, [(6.65, 5.625), (6.65, 8.075), (7.22, 8.075)], ORANGE[1])
    arrow(ax, [(11.17, 8.075), (10.43, 8.075)], ORANGE[1])
    arrow(ax, [(8.825, 7.42), (8.825, 6.28)], ORANGE[1])

    box(ax, 7.25, 3.1, 3.15, 1.25, "Plan-map cross-attention",
        "Q = map_LN(h_bev) + plan_pos\nOutput: map_update", GREEN, size=10.3)
    box(ax, 11.2, 3.1, 3.05, 1.25, "Map K / V",
        "Fixed task memory\n[B, 100, 256]", GREEN, size=10.5)
    arrow(ax, [(6.65, 5.625), (6.65, 3.725), (7.22, 3.725)], GREEN[1])
    arrow(ax, [(11.17, 3.725), (10.43, 3.725)], GREEN[1])
    arrow(ax, [(8.825, 4.38), (8.825, 4.97)], GREEN[1])

    box(ax, 7.9, 5.0, 2.85, 1.25, "Residual sum",
        "h_bev + det_update\n+ map_update", PURPLE, size=10.7)
    box(ax, 12.0, 5.0, 2.6, 1.25, "FFN",
        "Pre-LN → MLP\n+ merged residual", PURPLE, size=10.7)
    arrow(ax, [(10.78, 5.625), (11.97, 5.625)])
    arrow(ax, [(14.63, 5.625), (17.15, 5.625)])
    ax.text(15.95, 5.96, "Updated h", ha="center", color=INK, fontsize=11)
    ax.text(15.95, 5.16, "Next layer", ha="center", color=MUTED, fontsize=10.5)

    box(ax, .5, .45, 8.3, 2.05, "Task attention inputs",
        "Q = branch_query_norm(h_bev) + learned plan position\n"
        "K = memory_norm(memory) + position + confidence\n"
        "V = memory_norm(memory)\n"
        "Separate query norms; neither branch reads the sibling update.", GRAY, size=10.8)
    box(ax, 9.3, .45, 8.1, 2.05, "Content and metadata paths",
        "Latents → pooling / projection → K and V (gradients preserved)\n"
        "Predicted position / score → detach → learned MLP → K only\n"
        "No gate or learned merger; task memories are reused across layers.\n"
        "Parallel branches describe the graph, not GPU kernel scheduling.", PURPLE, size=10.6)
    save(fig, "17_planning_layer")


if __name__ == "__main__":
    overview()
    planner_layer()
