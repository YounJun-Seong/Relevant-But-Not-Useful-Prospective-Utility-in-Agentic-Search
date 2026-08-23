"""Figure: cost-accuracy of a budgeted prospective lookahead.

(a) Fraction of the full 8-candidate headroom recovered when only the CE top-k
    candidates are rolled out. Solid: Oracle-k, which keeps the best measured
    outcome and is an upper bound because it needs a scored rollout. Dashed:
    the deployable majority-vote rule, which uses no gold label.
(b) Why the cheap rule fails: on opportunity states, the useful candidate's
    continuation answer is usually a MINORITY answer, while the CE candidate's
    is far more often the modal one, so consensus aggregation is
    anti-correlated with utility.

Design: lines for an ordered budget; categorical hues in fixed slot order,
identical across panels so colour follows the setting; rule identity carried by
line style and an explicit legend, never by colour alone; zero drawn as a
neutral reference rule (the CE baseline).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
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
NAMES = [("HotpotQA dev", "HotpotQA dev"),
         ("HotpotQA lockbox", "HotpotQA lockbox"),
         ("2WikiMultiHopQA", "2Wiki"),
         ("Qwen2.5-14B", "Qwen2.5-14B"),
         ("MuSiQue", "MuSiQue")]
KS = [1, 2, 3, 4, 6, 8]
MARKERS = ["o", "s", "^", "D", "v"]
LABEL_GAP = 2.2     # percentage points between end labels


def despine(ax, keep=("bottom", "left")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def main():
    LA = json.load(open("results/lookahead/report.json"))
    DG = json.load(open("results/lookahead/diagnostics.json"))
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.1, 2.5),
                                   gridspec_kw={"width_ratios": [1.45, 1]})

    # ---------------- (a) recovered headroom vs budget ----------------
    axa.axhline(0, color=GRID, lw=0.9, ls=(0, (3, 3)), zorder=1)
    handles = []
    for i, (key, lab) in enumerate(NAMES):
        d = LA[key]
        orc = [100 * d["oracle_k"][str(k)]["recovered_fraction"] for k in KS]
        vot = [100 * d["vote"][str(k)]["recovered_fraction"] for k in KS]
        ln, = axa.plot(KS, orc, color=SERIES[i], lw=1.4, marker=MARKERS[i],
                       ms=3.4, mew=0.8, mec="white", zorder=3)
        axa.plot(KS, vot, color=SERIES[i], lw=1.1, ls=(0, (2.5, 2)), zorder=2,
                 alpha=0.85)
        handles.append((ln, lab))
    axa.set_xticks(KS)
    axa.set_xlabel("Rollout budget $k$ (CE top-$k$ candidates)")
    axa.set_ylabel("Headroom recovered (%)")
    axa.set_ylim(-95, 108)
    axa.set_yticks([-75, -50, -25, 0, 25, 50, 75, 100])
    axa.tick_params(axis="y", length=0)
    axa.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    axa.set_axisbelow(True)
    despine(axa)
    axa.set_title("(a)  Budgeted lookahead", loc="left", fontsize=7.6,
                  color=INK, pad=6)
    style = [plt.Line2D([], [], color=INK3, lw=1.4),
             plt.Line2D([], [], color=INK3, lw=1.1, ls=(0, (2.5, 2)))]
    leg1 = axa.legend(style, ["Oracle-$k$ (upper bound)",
                              "Majority vote (deployable)"],
                      loc="lower left", frameon=False, fontsize=6.3,
                      handlelength=1.8, labelspacing=0.25, borderpad=0.2)
    axa.add_artist(leg1)
    axa.legend([h for h, _ in handles], [t for _, t in handles],
               loc="center right", bbox_to_anchor=(1.005, 0.47), frameon=False,
               fontsize=6.2, handlelength=1.2, labelspacing=0.22,
               borderpad=0.15)

    # ---------------- (b) why consensus fails ----------------
    x = np.array([0, 1])
    ends = []
    for i, (key, lab) in enumerate(NAMES):
        g = DG[key]
        vals = [100 * g["oracle_answer_is_modal"], 100 * g["ce_answer_is_modal"]]
        axb.plot(x, vals, color=SERIES[i], lw=1.2, marker=MARKERS[i], ms=3.6,
                 mew=0.8, mec="white", zorder=3)
        ends.append((vals[1], SERIES[i]))
    ends.sort(key=lambda t: t[0])
    placed = []
    for v, c in ends:
        yy = v if not placed else max(v, placed[-1] + LABEL_GAP)
        placed.append(yy)
        axb.annotate(f"{v:.0f}%", (x[-1], yy), textcoords="offset points",
                     xytext=(7, 0), fontsize=6.3, color=INK2, va="center")
    axb.set_xticks(x)
    axb.set_xticklabels(["Useful\ncandidate", "CE\ncandidate"], fontsize=6.6)
    axb.set_xlim(-0.28, 1.42)
    axb.set_ylim(18, 52)
    axb.set_yticks([20, 30, 40, 50])
    axb.set_ylabel("Answer is the modal answer (%)")
    axb.tick_params(axis="y", length=0)
    axb.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    axb.set_axisbelow(True)
    despine(axb)
    axb.set_title("(b)  Why consensus fails", loc="left", fontsize=7.6,
                  color=INK, pad=6)

    fig.subplots_adjust(wspace=0.34)
    fig.savefig(f"{OUT}/fig3_lookahead.pdf", bbox_inches="tight",
                pad_inches=0.02)
    fig.savefig(f"{OUT}/fig3_lookahead.png", bbox_inches="tight",
                pad_inches=0.06, dpi=300)
    plt.close(fig)
    print("fig3_lookahead saved")


if __name__ == "__main__":
    main()
