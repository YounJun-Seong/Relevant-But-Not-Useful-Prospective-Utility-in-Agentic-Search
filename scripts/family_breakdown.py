"""Candidate-family contribution breakdown (post-hoc exploratory, R21 ⑤).

For each candidate family: mean realized utility, and the share of opportunity
states at which the family supplies the oracle pick. Ties at the maximum are
split evenly (fractional credit) because the confirmatory pipeline's tie rule
(first in family enumeration order) is order-biased and would over-credit the
families drawn first. The order-biased share is reported alongside so the
difference is visible.

Emits both a JSON report and LaTeX tables.
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.data_utils import read_jsonl

EPS = 1e-9
ORDER = ["bm25_top", "dense_top", "ce_top", "mid_rel", "novelty",
         "redundancy", "entity_bridge", "random"]
LABEL = {"bm25_top": r"\textbf{bm25\_top}", "dense_top": r"\textbf{dense\_top}",
         "ce_top": r"\textbf{ce\_top}", "mid_rel": r"\textbf{mid\_rel}",
         "novelty": r"\textbf{novelty}", "redundancy": r"\textbf{redundancy}",
         "entity_bridge": r"\textbf{entity\_bridge}",
         "random": r"\textbf{random}"}
STANDARD = {"bm25_top", "dense_top", "ce_top"}
DIRS = [("HotpotQA dev", "results/gate2"), ("HotpotQA lockbox", "results/lockbox"),
        ("2WikiMultiHopQA", "results/wiki2"), ("Qwen2.5-14B", "results/xagent"),
        ("MuSiQue", "results/musique")]


def analyze(gate_dir):
    by_state = defaultdict(list)
    for u in read_jsonl(f"{gate_dir}/utilities.jsonl"):
        by_state[(u["question_id"], u["state_idx"])].append(u)
    meanU, frac, first = defaultdict(list), defaultdict(float), defaultdict(int)
    n_opp = 0
    for rows in by_state.values():
        for r in rows:
            meanU[r["ctype"]].append(r["U_task_raw_f1"])
        mx = max(r["U_task_raw_f1"] for r in rows)
        if mx <= EPS:
            continue
        n_opp += 1
        win = [r for r in rows if abs(r["U_task_raw_f1"] - mx) < EPS]
        for w in win:
            frac[w["ctype"]] += 1.0 / len(win)
        first[max(rows, key=lambda r: r["U_task_raw_f1"])["ctype"]] += 1
    out = {"n_states": len(by_state), "n_opportunity_states": n_opp,
           "families": {}}
    for c in ORDER:
        if not meanU[c]:
            continue
        out["families"][c] = {
            "n_arms": len(meanU[c]),
            "mean_utility": sum(meanU[c]) / len(meanU[c]),
            "win_share_ties_split": frac[c] / n_opp,
            "win_share_first_max": first[c] / n_opp,
        }
    out["standard_share_ties_split"] = sum(
        frac[c] for c in STANDARD) / n_opp
    out["auxiliary_share_ties_split"] = sum(
        frac[c] for c in frac if c not in STANDARD) / n_opp
    return out


def main():
    R = {name: analyze(d) for name, d in DIRS}
    os.makedirs("results/family_breakdown", exist_ok=True)
    with open("results/family_breakdown/report.json", "w") as f:
        json.dump(R, f, indent=2)

    lb = R["HotpotQA lockbox"]
    L = []
    L.append(r"\begin{table}[h]")
    L.append(r"\centering")
    L.append(r"\small")
    L.append(r"\caption{\textbf{Table S\ref{sec:families}.} Candidate-family "
             r"contribution on the HotpotQA lockbox (%d states, %d with a "
             r"positive opportunity). \emph{Mean $U$} averages realized utility "
             r"over all arms of the family. \emph{Oracle share} is the fraction "
             r"of opportunity states at which the family supplies the oracle "
             r"pick, with ties at the maximum split evenly; the last column "
             r"applies the pipeline's order-based tie rule instead and is shown "
             r"only to expose its bias.}"
             % (lb["n_states"], lb["n_opportunity_states"]))
    L.append(r"\label{tab:families}")
    L.append(r"\begin{tabular}{lccc}")
    L.append(r"\toprule")
    L.append(r"\textbf{Family} & \textbf{Mean $U$} & "
             r"\textbf{Oracle share} & \textbf{(order-based)} \\")
    L.append(r"\midrule")
    for c in ORDER:
        if c not in lb["families"]:
            continue
        f = lb["families"][c]
        L.append(f"{LABEL[c]} & {f['mean_utility']:+.4f} & "
                 f"{f['win_share_ties_split']:.1%} & "
                 f"{f['win_share_first_max']:.1%} \\\\".replace("%", r"\%"))
    L.append(r"\midrule")
    L.append(r"Standard (bm25/dense/ce) & --- & "
             f"{lb['standard_share_ties_split']:.1%} & --- \\\\".replace("%", r"\%"))
    L.append(r"Auxiliary (other five) & --- & "
             f"{lb['auxiliary_share_ties_split']:.1%} & --- \\\\".replace("%", r"\%"))
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    L.append("")

    L.append(r"\begin{table}[h]")
    L.append(r"\centering")
    L.append(r"\small")
    L.append(r"\caption{\textbf{Table S\ref{sec:families}b.} Share of the "
             r"measured opportunity supplied by standard retrieval candidates "
             r"versus auxiliary probe families, across all five settings "
             r"(ties split evenly).}")
    L.append(r"\begin{tabular}{lccc}")
    L.append(r"\toprule")
    L.append(r"\textbf{Setting} & \textbf{Opp.\ states} & "
             r"\textbf{Standard} & \textbf{Auxiliary} \\")
    L.append(r"\midrule")
    for name, _ in DIRS:
        r = R[name]
        L.append(f"{name} & {r['n_opportunity_states']} & "
                 f"{r['standard_share_ties_split']:.1%} & "
                 f"{r['auxiliary_share_ties_split']:.1%} \\\\".replace("%", r"\%"))
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")

    tex = "\n".join(L)
    with open("results/family_breakdown/tables.tex", "w") as f:
        f.write(tex + "\n")
    print(tex)
    print("\nFAMILY_BREAKDOWN_DONE")


if __name__ == "__main__":
    main()
