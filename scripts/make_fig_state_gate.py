"""Figure 2: prospective identification and recoverability of headroom.

(a) ROC of the state-level gate for Z(s_t)=1[H(s_t)>0], five settings.
(b) ROC-AUC under progressively richer pre-intervention signals, with the two
    fine-tuned neural gates shown as a separate 'learned' region rather than
    folded into the shallow-feature progression.
(c) Budgeted lookahead on the HotpotQA lockbox: how much of the oracle headroom
    is already contained in the CE top-k candidates, and the two gold-free
    aggregation rules that fail to recover it.

Design: lines for ordered progressions and for a curve; categorical hues in
fixed slot order, identical across panels so colour follows the setting; a
legend carries identity so nothing is colour-alone; recessive axes; chance and
the CE baseline drawn as neutral reference rules.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

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
TAGS = [("gate2", "HotpotQA dev", "Hotpot dev"),
        ("lockbox", "HotpotQA lockbox", "Hotpot lockbox"),
        ("wiki2", "2WikiMultiHopQA", "2Wiki"),
        ("xagent", "Qwen2.5-14B", "Qwen-14B"),
        ("musique", "MuSiQue", "MuSiQue")]
SHALLOW = [("ce_margin_only", "CE\nmargin"), ("ce_only", "CE\nstats"),
           ("phi_only", "Cand.\naggr."), ("full", "All\nsignals")]
LEARNED = [("neural_minilm", "MiniLM"), ("neural_deberta", "DeBERTa")]
MARKERS = ["o", "s", "^", "D", "v"]
KS = [1, 2, 3, 4, 6, 8]
GAP_B, GAP_C = 0.0075, 3.0


def despine(ax, keep=("bottom", "left")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def spread(vals, gap):
    """Monotone de-collision for end labels."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0.0] * len(vals)
    prev = None
    for i in order:
        y = vals[i] if prev is None else max(vals[i], prev + gap)
        out[i] = y
        prev = y
    return out


def main():
    z = np.load("results/rq3_gate_ablation/predictions.npz")
    AG = json.load(open("results/rq3_gate_ablation/all_gates.json"))
    AB = json.load(open("results/rq3_gate_ablation/report.json"))["settings"]
    LA = json.load(open("results/lookahead/report.json"))
    fig, (axa, axb, axc) = plt.subplots(
        1, 3, figsize=(7.3, 2.3), gridspec_kw={"width_ratios": [0.95, 1.38, 1.0]})

    # ---------------- (a) ROC ----------------
    axa.plot([0, 1], [0, 1], color=GRID, lw=0.9, ls=(0, (3, 3)), zorder=1)
    axa.text(0.63, 0.55, "chance", color=INK3, fontsize=6.2, rotation=38,
             ha="center", va="center")
    handles = []
    for i, (tag, key, lab) in enumerate(TAGS):
        y, p = z[f"{tag}__y"], z[f"{tag}__full"]
        fpr, tpr, _ = roc_curve(y, p)   # AUC values live in panel (b) and the table
        ln, = axa.plot(fpr, tpr, color=SERIES[i], lw=1.3, zorder=3,
                       solid_capstyle="round")
        handles.append((ln, lab))
    axa.set_xlim(0, 1); axa.set_ylim(0, 1)
    axa.set_xticks([0, 0.5, 1]); axa.set_yticks([0, 0.5, 1])
    axa.set_xlabel("False positive rate")
    axa.set_ylabel("True positive rate")
    axa.tick_params(axis="y", length=0)
    despine(axa)
    axa.set_title("(a)  Identifying headroom states", loc="left",
                  fontsize=7.0, color=INK, pad=5)
    axa.legend([h for h, _ in handles], [t for _, t in handles],
               loc="lower right", frameon=False,
               fontsize=6.0, handlelength=1.1, handletextpad=0.4,
               labelspacing=0.22, borderpad=0.1, borderaxespad=0.25)

    # ---------------- (b) shallow signals vs learned gates ----------------
    xs = np.array([0, 1.4, 2.8, 4.2])
    xl = np.array([5.9, 7.8])
    axb.axhline(0.5, color=GRID, lw=0.9, ls=(0, (3, 3)), zorder=1)
    axb.axvline(4.95, color=GRID, lw=0.8, zorder=1)
    ends = []
    for i, (tag, key, lab) in enumerate(TAGS):
        fs = AB[key]["feature_sets"]
        sv = [fs[k]["auc"] for k, _ in SHALLOW]
        lv = [AG[key][k] for k, _ in LEARNED]
        axb.plot(xs, sv, color=SERIES[i], lw=1.15, marker=MARKERS[i], ms=3.2,
                 mew=0.7, mec="white", zorder=3)
        axb.plot(xl, lv, color=SERIES[i], marker=MARKERS[i], ms=3.4,
                 mew=0.7, mec="white", ls="none", zorder=3)
        ends.append(lv[-1])
    axb.set_xticks(list(xs) + list(xl))
    axb.set_xticklabels([l for _, l in SHALLOW] + [l for _, l in LEARNED],
                        fontsize=6.0)
    axb.set_xlim(-0.5, 8.45)
    axb.set_ylim(0.455, 0.665)
    axb.set_yticks([0.50, 0.55, 0.60, 0.65])
    axb.set_ylabel("ROC-AUC")
    axb.tick_params(axis="y", length=0)
    axb.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    axb.set_axisbelow(True)
    axb.text(2.1, 0.657, "shallow signals", color=INK3, fontsize=6.0,
             ha="center", va="top")
    axb.text(6.85, 0.657, "learned", color=INK3, fontsize=6.0,
             ha="center", va="top")
    axb.text(-0.42, 0.503, "chance", color=INK3, fontsize=6.0, ha="left",
             va="bottom")
    despine(axb)
    axb.set_title("(b)  Pre-intervention signals", loc="left", fontsize=7.0,
                  color=INK, pad=5)

    # ---------------- (c) budgeted lookahead (lockbox) ----------------
    d = LA["HotpotQA lockbox"]
    rec = [100 * d["oracle_k"][str(k)]["recovered_fraction"] for k in KS]
    axc.plot(KS, rec, color=SERIES[1], lw=1.5, marker="s", ms=3.6, mew=0.8,
             mec="white", zorder=3)
    for k, v in zip(KS, rec):
        if k in (2, 3, 4):
            axc.annotate(f"{v:.0f}%", (k, v), textcoords="offset points",
                         xytext=(3, -8), fontsize=6.1, color=INK2)
    axc.set_xticks(KS)
    axc.set_xlabel("Rollout budget $k$ (CE top-$k$)")
    axc.set_ylabel("Oracle headroom contained (%)")
    axc.set_ylim(-4, 108)
    axc.set_yticks([0, 25, 50, 75, 100])
    axc.tick_params(axis="y", length=0)
    axc.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    axc.set_axisbelow(True)
    despine(axc)
    axc.text(8.35, 33, "Gold-free recovery", color=INK2, fontsize=6.1,
             ha="right", va="top")
    axc.text(8.35, 23, "majority vote \u2014 no gain", color=INK3, fontsize=6.0,
             ha="right", va="top")
    axc.text(8.35, 14, "early stop \u2014 no gain", color=INK3, fontsize=6.0,
             ha="right", va="top")
    axc.text(1.15, 104, "HotpotQA lockbox", color=INK3, fontsize=6.1,
             ha="left", va="top")
    axc.set_title("(c)  Where headroom is concentrated", loc="left", fontsize=7.0,
                  color=INK, pad=5)

    fig.subplots_adjust(wspace=0.40)
    fig.savefig(f"{OUT}/fig2_state_gate.pdf", bbox_inches="tight",
                pad_inches=0.02)
    fig.savefig(f"{OUT}/fig2_state_gate.png", bbox_inches="tight",
                pad_inches=0.06, dpi=300)
    plt.close(fig)
    print("fig2_state_gate saved (3 panels)")


if __name__ == "__main__":
    main()
