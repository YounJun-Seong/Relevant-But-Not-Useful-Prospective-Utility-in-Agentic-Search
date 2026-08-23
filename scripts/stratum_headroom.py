"""Headroom by sampling stratum (reviewer: prevalence / cherry-picking).

Question-level Oracle-CE separated into the preregistered representative and
failure-enriched strata, so the non-enriched estimate is reported alongside the
pooled one rather than only as an opportunity rate.
"""
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.data_utils import read_jsonl

EPS = 1e-9
EVAL = [("HotpotQA dev", "results/gate2"), ("HotpotQA lockbox", "results/lockbox"),
        ("2WikiMultiHopQA", "results/wiki2"), ("Qwen2.5-14B", "results/xagent"),
        ("MuSiQue", "results/musique")]


def boot(per_q, n_boot=10000, seed=7):
    qg = [float(np.mean(v)) for v in per_q.values()]
    if not qg:
        return None
    rng = random.Random(seed)
    b = sorted(sum(qg[rng.randrange(len(qg))] for _ in qg) / len(qg)
               for _ in range(n_boot))
    return {"mean": sum(qg) / len(qg),
            "ci95": [b[int(.025 * n_boot)], b[int(.975 * n_boot)]],
            "n_questions": len(qg)}


def main():
    R = {}
    for name, gate_dir in EVAL:
        ce = {r["state_id"]: r["scores"]
              for r in read_jsonl(os.path.join(gate_dir,
                                               "candidate_ce_scores.jsonl"))}
        by = defaultdict(list)
        for u in read_jsonl(os.path.join(gate_dir, "utilities.jsonl")):
            by[(u["question_id"], u["state_idx"])].append(u)
        H = defaultdict(lambda: defaultdict(list))
        opp = defaultdict(list)
        for (qid, sidx), rr in by.items():
            sc = ce.get(f"{qid}::s{sidx}")
            if not sc:
                continue
            cand = [r for r in rr if r["doc"] in sc]
            if len(cand) < 2:
                continue
            st = cand[0].get("stratum") or "unknown"
            d_ce = max(cand, key=lambda r: sc[r["doc"]])
            best = max(r["arm_d"]["f1"] for r in cand)
            h = best - d_ce["arm_d"]["f1"]
            H["all"][qid].append(h)
            H[st][qid].append(h)
            opp["all"].append(float(h > EPS))
            opp[st].append(float(h > EPS))
        R[name] = {k: {"headroom": boot(v),
                       "state_failure_rate": float(np.mean(opp[k]))}
                   for k, v in H.items()}
        print(f"[{name}] done", flush=True)
    os.makedirs("results/stratum_headroom", exist_ok=True)
    out = "results/stratum_headroom/report.json"
    with open(out, "w") as f:
        json.dump(R, f, indent=2)
    print(json.dumps(R, indent=2))
    print("STRATUM_DONE")


if __name__ == "__main__":
    main()
