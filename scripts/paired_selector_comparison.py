"""Paired selector comparisons (reviewer A + B).

A  Is CE meaningfully better than random selection?  Question-level PAIRED
   bootstrap of CE - Random, in every setting. Random is evaluated in closed
   form as the mean outcome over the state's candidates (the exact expectation
   of a uniform pick), which removes Monte-Carlo noise from the baseline; a
   seeded single draw is reported alongside.

B  Are the residual headrooms under stronger / state-aware relevance selectors
   statistically reliable?  Oracle - selector with CI, plus the PAIRED
   difference selector - CE, which asks whether a stronger selector changes the
   headroom at all.
"""
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.data_utils import read_jsonl

EVAL = [("HotpotQA dev", "gate2", "results/gate2"),
        ("HotpotQA lockbox", "lockbox", "results/lockbox"),
        ("2WikiMultiHopQA", "wiki2", "results/wiki2"),
        ("Qwen2.5-14B", "xagent", "results/xagent"),
        ("MuSiQue", "musique", "results/musique")]
SELS = ["random_expected", "random_draw", "ce", "strong_qt", "strong_state",
        "llm_state", "oracle"]


def boot_paired(per_q, n_boot=10000, seed=7):
    """per_q: question -> list of per-state differences."""
    qg = [float(np.mean(v)) for v in per_q.values()]
    rng = random.Random(seed)
    b = sorted(sum(qg[rng.randrange(len(qg))] for _ in qg) / len(qg)
               for _ in range(n_boot))
    return {"mean": sum(qg) / len(qg),
            "ci95": [b[int(.025 * n_boot)], b[int(.975 * n_boot)]],
            "n_questions": len(qg)}


def main():
    R = {}
    for name, tag, gate_dir in EVAL:
        ce = {r["state_id"]: r["scores"]
              for r in read_jsonl(os.path.join(gate_dir,
                                               "candidate_ce_scores.jsonl"))}
        sp = f"results/strong_reranker/{tag}.jsonl"
        strong = {r["state_id"]: r["scores"] for r in read_jsonl(sp)} \
            if os.path.exists(sp) else {}
        lp = f"results/state_aware_relevance/{tag}.jsonl"
        llm = {r["state_id"]: r["ranking"] for r in read_jsonl(lp)} \
            if os.path.exists(lp) else {}
        by = defaultdict(list)
        for u in read_jsonl(os.path.join(gate_dir, "utilities.jsonl")):
            by[(u["question_id"], u["state_idx"])].append(u)

        y = {s: defaultdict(list) for s in SELS}
        rng = random.Random(7)
        n = 0
        for (qid, sidx), rr in by.items():
            sid = f"{qid}::s{sidx}"
            sc = ce.get(sid)
            if not sc:
                continue
            f1 = {r["doc"]: r["arm_d"]["f1"] for r in rr}
            cand = [d for d in f1 if d in sc]
            if len(cand) < 2:
                continue
            st, rk = strong.get(sid), llm.get(sid)
            if not st or not all(d in st for d in cand):
                continue
            n += 1
            vals = {
                "random_expected": float(np.mean([f1[d] for d in cand])),
                "random_draw": f1[rng.choice(cand)],
                "ce": f1[max(cand, key=lambda d: sc[d])],
                "strong_qt": f1[max(cand, key=lambda d: st[d]["strong_qt"])],
                "strong_state": f1[max(cand,
                                       key=lambda d: st[d]["strong_state"])],
                "oracle": max(f1[d] for d in cand),
            }
            if rk:
                vals["llm_state"] = f1[next((d for d in rk if d in f1),
                                            max(cand, key=lambda d: sc[d]))]
            for s, v in vals.items():
                y[s][qid].append(v)

        entry = {"n_states": n, "abs_f1": {}, "headroom_oracle_minus": {},
                 "paired_vs_ce": {}, "paired_ce_minus_random": None}
        for s in SELS:
            if not y[s]:
                continue
            entry["abs_f1"][s] = boot_paired(y[s])
            entry["headroom_oracle_minus"][s] = boot_paired(
                {q: [a - b for a, b in zip(y["oracle"][q], y[s][q])]
                 for q in y[s]})
            if s not in ("ce", "oracle"):
                entry["paired_vs_ce"][s] = boot_paired(
                    {q: [a - b for a, b in zip(y[s][q], y["ce"][q])]
                     for q in y[s]})
        entry["paired_ce_minus_random"] = boot_paired(
            {q: [a - b for a, b in zip(y["ce"][q], y["random_expected"][q])]
             for q in y["ce"]})
        entry["paired_ce_minus_random_draw"] = boot_paired(
            {q: [a - b for a, b in zip(y["ce"][q], y["random_draw"][q])]
             for q in y["ce"]})
        R[name] = entry
        print(f"[{name}] {n} states", flush=True)

    os.makedirs("results/paired_selectors", exist_ok=True)
    out = "results/paired_selectors/report.json"
    with open(out, "w") as f:
        json.dump(R, f, indent=2)
    print(f"-> {out}")
    print("PAIRED_SELECTORS_DONE")


if __name__ == "__main__":
    main()
