"""A lightweight prospective lookahead: cost-accuracy trade-off.

PIU says headroom exists; this asks how much of it a budgeted lookahead can
recover. At each state the agent rolls out the CE top-k candidates and then
decides. Two decision rules:

  vote      DEPLOYABLE. Answer by majority over the k continuations' answers,
            ties broken by CE rank. Uses no gold label at any point.
  oracle_k  UPPER BOUND. Keep the best of the k measured outcomes; not
            deployable (needs the gold-scored outcome), it bounds what any
            gold-free rule at budget k could reach.

k=1 is exactly the CE baseline; k=|D_t| is the full oracle for oracle_k.
No new rollouts: every candidate continuation is already stored.
"""
import json
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.data_utils import read_jsonl, f1_score, normalize_answer

EVAL = [("HotpotQA dev", "results/gate2"), ("HotpotQA lockbox", "results/lockbox"),
        ("2WikiMultiHopQA", "results/wiki2"), ("Qwen2.5-14B", "results/xagent"),
        ("MuSiQue", "results/musique")]
KS = [1, 2, 3, 4, 6, 8]


def boot(per_q, n_boot=10000, seed=7):
    qg = [float(np.mean(v)) for v in per_q.values()]
    rng = random.Random(seed)
    b = sorted(sum(qg[rng.randrange(len(qg))] for _ in qg) / len(qg)
               for _ in range(n_boot))
    return {"mean": sum(qg) / len(qg),
            "ci95": [b[int(.025 * n_boot)], b[int(.975 * n_boot)]],
            "n_questions": len(qg)}


def main():
    R = {}
    for name, gate_dir in EVAL:
        with open(os.path.join(gate_dir, "questions.json")) as f:
            qmeta = json.load(f)
        gold = {r["question_id"]: r["answer_gold"]
                for r in read_jsonl(qmeta["pre_path"])}
        sc_all = {r["state_id"]: r["scores"]
                  for r in read_jsonl(os.path.join(gate_dir,
                                                   "candidate_ce_scores.jsonl"))}
        by = defaultdict(list)
        for u in read_jsonl(os.path.join(gate_dir, "utilities.jsonl")):
            by[(u["question_id"], u["state_idx"])].append(u)

        vote = {k: defaultdict(list) for k in KS}
        orak = {k: defaultdict(list) for k in KS}
        n = 0
        for (qid, sidx), rr in by.items():
            sc = sc_all.get(f"{qid}::s{sidx}")
            if not sc:
                continue
            cand = [r for r in rr if r["doc"] in sc]
            if len(cand) < 2:
                continue
            n += 1
            order = sorted(cand, key=lambda r: -sc[r["doc"]])
            g = gold[qid]
            for k in KS:
                top = order[:k]
                # deployable: majority vote over continuation answers
                norm = [normalize_answer(r["arm_d"]["answer"] or "") for r in top]
                cnt = Counter(norm)
                best_n, _ = cnt.most_common(1)[0]
                winner = next(r for r, s in zip(top, norm) if s == best_n)
                vote[k][qid].append(f1_score(winner["arm_d"]["answer"] or "", g))
                # upper bound: best measured outcome among the k
                orak[k][qid].append(max(r["arm_d"]["f1"] for r in top))
        entry = {"n_states": n, "vote": {}, "oracle_k": {}}
        for k in KS:
            entry["vote"][str(k)] = boot(vote[k])
            entry["oracle_k"][str(k)] = boot(orak[k])
        # recovered fraction of the full headroom
        ce = entry["vote"]["1"]["mean"]
        full = entry["oracle_k"][str(KS[-1])]["mean"]
        for k in KS:
            for tag in ("vote", "oracle_k"):
                v = entry[tag][str(k)]["mean"]
                entry[tag][str(k)]["recovered_fraction"] = (
                    (v - ce) / (full - ce) if full > ce else None)
        R[name] = entry
        print(f"[{name}] vote k=2 {entry['vote']['2']['mean']:.4f} "
              f"k=3 {entry['vote']['3']['mean']:.4f} | CE {ce:.4f} "
              f"| oracle {full:.4f}", flush=True)
    os.makedirs("results/lookahead", exist_ok=True)
    out = "results/lookahead/report.json"
    with open(out, "w") as f:
        json.dump(R, f, indent=2)
    print(f"-> {out}")
    print("LOOKAHEAD_DONE")


if __name__ == "__main__":
    main()
