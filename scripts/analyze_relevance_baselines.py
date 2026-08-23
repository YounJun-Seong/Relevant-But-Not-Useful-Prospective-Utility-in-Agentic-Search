"""Do stronger / state-aware relevance selectors close the measured headroom?

Compares every relevance selector we have against the oracle on realized
downstream F1, using the stored intervention outcomes.

  ce             ms-marco-MiniLM-L6-v2 on q_t          (confirmatory baseline)
  strong_qt      bge-reranker-v2-m3 on q_t             (strong, state-blind)
  strong_state   bge-reranker-v2-m3 on q0+q_t+seen     (strong, state-aware)
  llm_state      Qwen2.5-7B listwise, state-aware      (if available)
  random / oracle                                       (reference points)
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
EVAL = [("HotpotQA dev", "gate2", "results/gate2"),
        ("HotpotQA lockbox", "lockbox", "results/lockbox"),
        ("2WikiMultiHopQA", "wiki2", "results/wiki2"),
        ("Qwen2.5-14B", "xagent", "results/xagent"),
        ("MuSiQue", "musique", "results/musique")]


def boot(per_q, n_boot=10000, seed=7):
    qg = [float(np.mean(v)) for v in per_q.values()]
    rng = random.Random(seed)
    b = sorted(sum(qg[rng.randrange(len(qg))] for _ in qg) / len(qg)
               for _ in range(n_boot))
    return {"mean": sum(qg) / len(qg),
            "ci95": [b[int(.025 * n_boot)], b[int(.975 * n_boot)]]}


def main():
    R = {}
    for name, tag, gate_dir in EVAL:
        ce = {r["state_id"]: r["scores"]
              for r in read_jsonl(os.path.join(gate_dir,
                                               "candidate_ce_scores.jsonl"))}
        strong_path = f"results/strong_reranker/{tag}.jsonl"
        strong = {r["state_id"]: r["scores"]
                  for r in read_jsonl(strong_path)} if os.path.exists(strong_path) else {}
        llm_path = f"results/state_aware_relevance/{tag}.jsonl"
        llm = {r["state_id"]: r["ranking"]
               for r in read_jsonl(llm_path)} if os.path.exists(llm_path) else {}
        by = defaultdict(list)
        for u in read_jsonl(os.path.join(gate_dir, "utilities.jsonl")):
            by[(u["question_id"], u["state_idx"])].append(u)
        sels = ["random", "ce", "strong_qt", "strong_state", "llm_state", "oracle"]
        absf = {s: defaultdict(list) for s in sels}
        head = {s: defaultdict(list) for s in sels}
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
            st = strong.get(sid)
            rk = llm.get(sid)
            picks = {"random": rng.choice(cand),
                     "ce": max(cand, key=lambda d: sc[d]),
                     "oracle": max(cand, key=lambda d: f1[d])}
            if st and all(d in st for d in cand):
                picks["strong_qt"] = max(cand, key=lambda d: st[d]["strong_qt"])
                picks["strong_state"] = max(cand,
                                            key=lambda d: st[d]["strong_state"])
            if rk:
                picks["llm_state"] = next((d for d in rk if d in f1), picks["ce"])
            n += 1
            best = max(f1[d] for d in cand)
            for s, d in picks.items():
                absf[s][qid].append(f1[d])
                head[s][qid].append(best - f1[d])
        R[name] = {"n_states": n,
                   "abs_f1": {s: boot(v) for s, v in absf.items() if v},
                   "headroom_oracle_minus": {s: boot(v) for s, v in head.items() if v}}
        print(f"[{name}] done", flush=True)
    os.makedirs("results/relevance_baselines", exist_ok=True)
    out = "results/relevance_baselines/report.json"
    with open(out, "w") as f:
        json.dump(R, f, indent=2)
    print(f"-> {out}")
    print("RELEVANCE_BASELINES_DONE")


if __name__ == "__main__":
    main()
