"""Paper figures (vector PDF for ACM sigconf + 300dpi PNG previews), v2.

Design (dataviz method): form first; validated default palette
(blue #2a78d6 / orange #eb6834 / neutral inks); thin marks, recessive axes,
selective direct labels; identity/sign never color-alone (position, marker
shape, glyphs). Numbers transcribed from paper/results_master.md.

F1  headroom forest — right-rail value columns, dev vs confirmatory markers
F2  signature coefficients — 4 panels, confirmatory vs exploratory
F3  identifiability gap — bars + stable-ceiling strip
F4  structure replication — transfer bars + sign matrix
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE = "#2a78d6", "#eb6834"
BLUE_L = "#9ec5f4"
INK, INK2, INK3, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e3e2de"
plt.rcParams.update({
    "font.size": 7.5, "font.family": "DejaVu Sans",
    "axes.edgecolor": "#c9c8c4", "axes.linewidth": 0.6,
    "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK3, "ytick.color": INK2,
    "xtick.labelsize": 6.8, "ytick.labelsize": 7,
    "xtick.major.size": 2.5, "ytick.major.size": 0,
    "xtick.major.width": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
OUT = "paper/figures"
COL_W = 3.35

SETTINGS = ["HotpotQA dev", "HotpotQA lockbox", "2Wiki", "Qwen2.5-14B",
            "MuSiQue"]


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", pad_inches=0.06,
                dpi=300)
    plt.close(fig)
    print(name, "saved")


def despine(ax, keep=("bottom",)):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


# ============ F1: headroom forest with right-rail columns ============
def fig1():
    mean = [0.082, 0.117, 0.142, 0.097, 0.097]
    lo = [0.062, 0.092, 0.117, 0.076, 0.080]
    hi = [0.104, 0.142, 0.168, 0.120, 0.116]
    opp = ["30.1", "29.3", "26.4", "32.1", "21.8"]
    is_dev = [True, False, False, False, False]
    y = np.arange(len(SETTINGS))[::-1]

    fig, ax = plt.subplots(figsize=(COL_W, 1.85))
    ax.axvline(0, color=INK2, lw=0.9, zorder=2)
    for yi in y[::2]:
        ax.axhspan(yi - 0.42, yi + 0.42, color="#f4f3f1", zorder=0, lw=0)
    for yi, m, l, h, dev in zip(y, mean, lo, hi, is_dev):
        ax.plot([l, h], [yi, yi], color=BLUE, lw=1.7,
                solid_capstyle="round", zorder=3)
        for xx in (l, h):
            ax.plot([xx, xx], [yi - 0.10, yi + 0.10], color=BLUE, lw=1.1,
                    zorder=3)
        ax.plot(m, yi, "o", ms=4.8, zorder=4,
                markerfacecolor=("white" if dev else BLUE),
                markeredgecolor=BLUE, markeredgewidth=1.1)
    X1, X2 = 0.222, 0.278
    ax.text(X1, 4.78, "ΔF1", fontsize=6.2, color=INK3, ha="right",
            style="italic")
    ax.text(X2, 4.78, "Opp.", fontsize=6.2, color=INK3, ha="right",
            style="italic")
    for yi, m, o in zip(y, mean, opp):
        ax.text(X1, yi, f"+{m:.3f}", fontsize=6.9, color=INK, ha="right",
                va="center")
        ax.text(X2, yi, f"{o}%", fontsize=6.9, color=INK2, ha="right",
                va="center")
    ax.set_yticks(y, SETTINGS)
    ax.set_ylim(-0.55, 5.1)
    ax.set_xlim(-0.014, 0.29)
    ax.set_xticks([0, 0.05, 0.10, 0.15])
    ax.set_xlabel("Oracle − CE  (Δ answer F1, question-level 95% CI)",
                  fontsize=7.0, labelpad=2)
    for gx in (0.05, 0.10, 0.15):
        ax.axvline(gx, color=GRID, lw=0.5, zorder=1)
    despine(ax)
    save(fig, "fig1_headroom")


# ============ F2: signature coefficients ============
def fig2():
    feats = [
        ("Query alignment", [-2.54, -2.45, -1.91, -2.93, -2.19], "conf"),
        ("Document length", [-3.37, -2.73, -3.82, -2.16, -2.00], "conf"),
        ("Redundancy", [1.06, -1.41, 1.14, -1.79, 1.76], "expl"),
        ("Novelty", [0.50, -0.21, 1.74, -0.29, 1.18], "expl"),
    ]
    markers = ["o", "s", "D", "^", "v"]
    fig, axes = plt.subplots(1, 4, figsize=(COL_W * 2.05, 1.7), sharex=True)
    for ax, (name, vals, kind) in zip(axes, feats):
        conf = kind == "conf"
        ax.axvline(0, color=INK2, lw=0.8, zorder=2)
        yy = np.arange(len(vals))[::-1]
        for yi, v, mk in zip(yy, vals, markers):
            ax.plot([0, v], [yi, yi], color=GRID, lw=1.0, zorder=1)
            ax.plot(v, yi, mk, color=(BLUE if v < 0 else ORANGE), ms=4.4,
                    zorder=3, markeredgecolor="white", markeredgewidth=0.7)
        ax.set_title(f"{name}\n({'confirmatory' if conf else 'exploratory'})",
                     fontsize=7.0, color=(INK if conf else INK3), pad=4)
        ax.set_yticks([])
        ax.set_xlim(-4.5, 2.7)
        ax.set_xticks([-4, -2, 0, 2])
        ax.xaxis.grid(True, color=GRID, lw=0.4, zorder=0)
        ax.set_axisbelow(True)
        despine(ax)
    axes[0].set_yticks(np.arange(5)[::-1], SETTINGS)
    axes[0].tick_params(axis="y", length=0, labelsize=6.8)
    fig.supxlabel("Refit coefficient within CE-conflicting conditions "
                  "(← negative        positive →)", fontsize=7.2, y=-0.07,
                  color=INK2)
    fig.subplots_adjust(wspace=0.14)
    save(fig, "fig2_signatures")


# ============ F3: identifiability gap (단일 패널) ============
def fig3():
    fig, ax = plt.subplots(figsize=(COL_W, 1.6))
    labels = ["Oracle-condition\nrouting",
              "Deployable gate\n(best mixture)",
              "CE-aligned\nexpert only"]
    vals = [76.2, 56.3, 56.7]
    colors = [BLUE, ORANGE, "#6f6e6a"]
    y = np.arange(3)[::-1]
    for yi, v, c in zip(y, vals, colors):
        ax.barh(yi, v, height=0.56, color=c, zorder=2)
        ax.annotate(f"{v:.1f}", (v - 1.8, yi), ha="right", va="center",
                    fontsize=7.6, color="white", fontweight="bold", zorder=4)
    ax.axvline(50, color=INK2, lw=0.8, ls=(0, (3, 2.2)), zorder=3)
    ax.annotate("chance", (50, 2.62), fontsize=6.1, color=INK2, ha="center")
    for xx, y0 in ((76.2, 2.0), (56.3, 1.0)):
        ax.plot([xx, xx], [y0, 1.5], color=INK, lw=0.7, ls=(0, (1.5, 1.5)),
                zorder=3)
    ax.annotate("", xy=(76.2, 1.5), xytext=(56.3, 1.5),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9))
    ax.annotate("identifiability gap\nrecovery ≈ 0", xy=(80.5, 1.5),
                fontsize=6.6, ha="left", va="center", color=INK,
                linespacing=1.35)
    ax.set_yticks(y, labels, fontsize=6.8)
    ax.set_ylim(-0.55, 2.85)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.spines["bottom"].set_bounds(0, 100)
    ax.set_xlabel("Pairwise accuracy on lockbox (%)", fontsize=7.0,
                  labelpad=2)
    despine(ax)
    save(fig, "fig3_identifiability")


# ============ F4: structure replication ============
def fig4():
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(COL_W * 2.05, 1.95), width_ratios=[1.3, 1],
        constrained_layout=True)
    evals = ["HotpotQA\nlockbox", "2Wiki", "Qwen2.5-14B", "MuSiQue"]
    pair_acc = [69.2, 66.0, 66.7, 66.6]
    clus = [(66.4, 61.8, 70.9), (63.3, 58.5, 67.9), (64.9, 60.4, 69.0), None]
    x = np.arange(4)
    ax.bar(x, pair_acc, width=0.56, color=BLUE, zorder=2)
    for xi, v, c in zip(x, pair_acc, clus):
        dx = -0.16 if c else 0.0  # CI 휘스커(중앙)와 분리
        ax.annotate(f"{v:.1f}", (xi + dx, v - 1.2), ha="center", va="top",
                    fontsize=6.8, color="white", fontweight="bold", zorder=5)
    for xi, c in zip(x, clus):
        if c:
            m, l, h = c
            ax.plot([xi, xi], [l, h], color=INK, lw=1.1, zorder=3)
            ax.plot(xi, m, "o", ms=3.6, color="white",
                    markeredgecolor=INK, markeredgewidth=0.9, zorder=4)
    ax.axhline(50, color=INK2, lw=0.8, ls=(0, (3, 2.2)), zorder=1)
    ax.annotate("chance", (-0.48, 50.9), fontsize=6.1, color=INK2)
    ax.set_xticks(x, evals, fontsize=6.7)
    ax.set_ylim(45, 76)
    ax.set_yticks([50, 60, 70])
    ax.set_ylabel("Frozen specialist accuracy (%)", fontsize=7.0)
    ax.tick_params(axis="x", length=0)
    despine(ax, keep=("bottom", "left"))
    ax.set_title("(a) Transfer of the HotpotQA-7B\nutility specialist",
                 fontsize=7.0, pad=4)

    feats = ["Query align.", "Doc. length", "Redundancy", "Novelty"]
    signs = np.array([[-1] * 5, [-1] * 5,
                      [1, -1, 1, -1, 1], [1, -1, 1, -1, 1]])
    for i in range(4):
        for j in range(5):
            s = signs[i, j]
            ax2.add_patch(plt.Rectangle(
                (j + 0.05, 3 - i + 0.05), 0.90, 0.90,
                color=(BLUE if s < 0 else ORANGE),
                alpha=0.92 if i < 2 else 0.55, lw=0))
            ax2.text(j + 0.5, 3 - i + 0.47, "−" if s < 0 else "+",
                     ha="center", va="center", fontsize=9, color="white",
                     fontweight="bold")
    ax2.axhline(2, color="white", lw=2.4, zorder=3)
    ax2.set_xlim(-0.02, 5.02)
    ax2.set_ylim(-0.06, 4.06)
    ax2.set_xticks(np.arange(5) + 0.5,
                   ["dev", "lock", "2Wiki", "14B", "MuSQ"], fontsize=6.4)
    ax2.set_yticks(np.arange(4)[::-1] + 0.5, feats, fontsize=6.7)
    ax2.tick_params(length=0)
    despine(ax2, keep=())
    ax2.set_title("(b) Coefficient sign by setting\n(top two: confirmatory)",
                  fontsize=7.0, pad=4)
    ax2.set_aspect("equal")
    save(fig, "fig4_replication")


if __name__ == "__main__" and "--v2only" not in __import__("sys").argv:
    fig1(); fig2(); fig3(); fig4()
    print("ALL_FIGS_DONE")


# ============ F3 v2: identifiability — (a) ROC + (b) routing ============
def fig3_v2():
    import json
    G = json.load(open("results/lockbox/gate_scores.json"))
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(COL_W, 1.9), width_ratios=[1, 1.12],
        constrained_layout=True)
    # (a) ROC
    ax.plot([0, 1], [0, 1], color=INK3, lw=0.7, ls=(0, (2, 2)), zorder=1)
    curves = [("gbm", "Feature gate", BLUE_L, 0.9),
              ("transformer", "Transformer gate", "#5b9be0", 0.9),
              ("ens", "Frozen ensemble", BLUE, 1.3)]
    aucs = G["auc"]
    for key, label, c, lw in curves:
        r = G[f"roc_{key}"]
        a = aucs["ensemble" if key == "ens" else key]
        ax.plot(r["fpr"], r["tpr"], color=c, lw=lw, zorder=3,
                label=f"{label}  {a:.2f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("False positive rate", fontsize=6.6, labelpad=1.5)
    ax.set_ylabel("True positive rate", fontsize=6.6, labelpad=1.5)
    ax.tick_params(labelsize=6.0, length=2)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=5.5, frameon=True, framealpha=0.92,
              edgecolor="none", handlelength=1.4, borderaxespad=0.15,
              labelspacing=0.25, title="AUC", title_fontsize=5.5)
    ax.text(0.04, 0.965, "Richer gates improve\ndiscrimination only modestly",
            fontsize=5.7, color=INK2, va="top", linespacing=1.25,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                      alpha=0.85))
    ax.set_title("(a) Detecting CE-conflicting\npairs on the lockbox",
                 fontsize=6.8, pad=3)
    despine(ax, keep=("bottom", "left"))
    # (b) routing bars
    labels = ["Oracle-\ncondition\nrouting", "Deployable\ngate (best\nmixture)",
              "CE-aligned\nexpert\nonly"]
    vals = [76.2, 56.3, 56.7]
    cols = [BLUE, ORANGE, "#6f6e6a"]
    x = np.arange(3)
    ax2.bar(x, vals, width=0.56, color=cols, zorder=2)
    for xi, v in zip(x, vals):
        ax2.annotate(f"{v:.1f}", (xi, v + 0.8), ha="center", va="bottom",
                     fontsize=6.5, color=INK, fontweight="bold")
    ax2.axhline(50, color=INK2, lw=0.7, ls=(0, (3, 2.2)), zorder=1)
    ax2.annotate("chance", (-0.42, 50.7), fontsize=5.5, color=INK2, ha="left")
    ax2.annotate("", xy=(0.5, 76.2), xytext=(0.5, 56.3),
                 arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8,
                                 shrinkA=0, shrinkB=0))
    ax2.text(0.58, 66.3, "identifiability\ngap", fontsize=5.7, color=INK,
             va="center", linespacing=1.2)
    ax2.set_xticks(x, labels, fontsize=5.7)
    ax2.set_ylim(44, 84); ax2.set_yticks([50, 60, 70, 80])
    ax2.tick_params(axis="x", length=0, pad=2)
    ax2.tick_params(axis="y", labelsize=6.0)
    ax2.set_ylabel("Pairwise accuracy (%)", fontsize=6.6, labelpad=1.5)
    ax2.set_title("(b) Routing outcome\n(recovery \u2248 \u22120.02)", fontsize=6.8,
                  pad=3)
    despine(ax2, keep=("bottom", "left"))
    save(fig, "fig3_identifiability_v2")


if __name__ == "__main__" and "--v2only" in __import__("sys").argv:
    fig3_v2()
