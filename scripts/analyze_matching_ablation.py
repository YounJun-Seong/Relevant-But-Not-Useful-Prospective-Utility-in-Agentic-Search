"""Analysis for the control-matching ablation.

Recomputes the controlled diagnostic U_ctrl = F1(+d) - F1(+d~) under each
matching specification against the identical intervention arms, plus realized
matching quality (rank-bucket agreement, length ratio).
Post-hoc exploratory.
"""
import argparse
import json
import os
import random
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.data_utils import read_jsonl


def boot_ci(per_q, n_boot=10000, seed=7):
    qg = [sum(v) / len(v) for v in per_q.values()]
    rng = random.Random(seed)
    boots = sorted(sum(qg[rng.randrange(len(qg))] for _ in qg) / len(qg)
                   for _ in range(n_boot))
    return {"n_questions": len(qg), "mean": sum(qg) / len(qg),
            "ci95": [boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-dir", default="results/lockbox")
    ap.add_argument("--ablation-dir", default="results/matching_ablation")
    args = ap.parse_args()

    utils = read_jsonl(os.path.join(args.gate_dir, "utilities.jsonl"))
    f1_d, f1_ctrl_base = {}, {}
    for u in utils:
        k = (u["question_id"], u["state_idx"], u["ctype"])
        f1_d[k] = u["arm_d"]["f1"]
        if u.get("arm_ctrl"):
            f1_ctrl_base[k] = u["arm_ctrl"]["f1"]

    rows = read_jsonl(os.path.join(args.ablation_dir, "arm_results.jsonl"))
    by_spec = defaultdict(dict)
    diag = defaultdict(lambda: {"rank_eq": [], "len_ratio": [], "fallback": 0})
    for r in rows:
        k = (r["question_id"], r["state_idx"], r["ctype"])
        by_spec[r["spec"]][k] = r["outcome"]["f1"]
        d = diag[r["spec"]]
        d["rank_eq"].append(bool(r.get("rank_bucket_equal")))
        d["len_ratio"].append(r.get("len_ratio") or 1.0)
        d["fallback"] += int(bool(r.get("fallback")))

    keys_all = set(f1_d) & set(f1_ctrl_base)
    for spec in by_spec:
        keys_all &= set(by_spec[spec])
    keys_all = sorted(keys_all)

    def summarize(get_ctrl, keys):
        per_q = defaultdict(list)
        for k in keys:
            per_q[k[0]].append(f1_d[k] - get_ctrl(k))
        return boot_ci(per_q)

    R = {"n_interventions_compared": len(keys_all),
         "specs": {}}
    R["specs"]["full (confirmatory)"] = summarize(
        lambda k: f1_ctrl_base[k], keys_all)
    for spec in sorted(by_spec):
        R["specs"][spec] = summarize(lambda k, s=spec: by_spec[s][k], keys_all)

    R["matching_quality"] = {}
    base_rank_eq, base_len = [], []
    plan = read_jsonl(os.path.join(args.ablation_dir, "ablation_plan.jsonl"))
    for spec, d in diag.items():
        n = len(d["rank_eq"])
        R["matching_quality"][spec] = {
            "rank_bucket_agreement": sum(d["rank_eq"]) / n,
            "median_length_ratio": statistics.median(d["len_ratio"]),
            "share_length_ratio_within_25pct":
                sum(1 for x in d["len_ratio"] if 0.75 <= x <= 1.25) / n,
            "fallback_rate": d["fallback"] / n,
        }
    out = os.path.join(args.ablation_dir, "MATCHING_ABLATION_REPORT.json")
    with open(out, "w") as f:
        json.dump(R, f, indent=2)
    print(json.dumps(R, indent=2))
    print(f"-> {out}")
    print("MATCHING_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
