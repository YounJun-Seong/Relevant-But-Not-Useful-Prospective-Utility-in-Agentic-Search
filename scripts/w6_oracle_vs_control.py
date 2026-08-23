"""Controlled PIU for the candidates that matter (reviewer W6).

The reported controlled diagnostic averages over ALL candidates, most of which
are useless, so a near-zero mean is expected by construction. This recomputes
the same quantity separately for the candidate each selector actually picks:

    U_ctrl(d) = Y(s_t, E<t u {d}) - Y(s_t, E<t u {d~})

for d in {oracle pick, CE pick, random pick, anti-oracle pick}, plus the paired
contrast U_ctrl(d_PIU) - U_ctrl(d_CE).

The oracle pick is chosen on the RAW outcome and then evaluated on the
CONTROLLED one, so a selection effect remains; the random and anti-oracle rows
bound it from both sides.
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
    if len(qg) < 2:
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
        sc_all = {r["state_id"]: r["scores"]
                  for r in read_jsonl(os.path.join(gate_dir,
                                                   "candidate_ce_scores.jsonl"))}
        by = defaultdict(list)
        for u in read_jsonl(os.path.join(gate_dir, "utilities.jsonl")):
            by[(u["question_id"], u["state_idx"])].append(u)
        keys = ["oracle", "ce", "random", "anti_oracle"]
        U = {k: defaultdict(list) for k in keys}
        contrast = defaultdict(list)
        rng = random.Random(7)
        n = 0
        for (qid, sidx), rr in by.items():
            sc = sc_all.get(f"{qid}::s{sidx}")
            if not sc:
                continue
            # only candidates whose matched control arm exists
            cand = [r for r in rr
                    if r["doc"] in sc and r.get("U_task_ctrl_f1") is not None]
            if len(cand) < 2:
                continue
            n += 1
            picks = {
                "oracle": max(cand, key=lambda r: r["arm_d"]["f1"]),
                "ce": max(cand, key=lambda r: sc[r["doc"]]),
                "random": rng.choice(cand),
                "anti_oracle": min(cand, key=lambda r: r["arm_d"]["f1"]),
            }
            for k, p in picks.items():
                U[k][qid].append(p["U_task_ctrl_f1"])
            contrast[qid].append(picks["oracle"]["U_task_ctrl_f1"]
                                 - picks["ce"]["U_task_ctrl_f1"])
        entry = {"n_states": n,
                 "U_ctrl": {k: boot(U[k]) for k in keys},
                 "paired_oracle_minus_ce": boot(contrast)}
        R[name] = entry
        o, c = entry["U_ctrl"]["oracle"], entry["U_ctrl"]["ce"]
        print(f"[{name}] oracle {o['mean']:+.4f} | ce {c['mean']:+.4f} | "
              f"contrast {entry['paired_oracle_minus_ce']['mean']:+.4f}",
              flush=True)
    os.makedirs("results/w6_controlled", exist_ok=True)
    out = "results/w6_controlled/report.json"
    with open(out, "w") as f:
        json.dump(R, f, indent=2)
    print(f"-> {out}")
    print("W6_DONE")


if __name__ == "__main__":
    main()
