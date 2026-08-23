"""Does a state-aware relevance ranker close the measured headroom? (W4)

Compares selectors on realized downstream F1:
  CE            query-local cross-encoder (confirmatory baseline)
  StateAware    LLM listwise ranker given q0, q_t and consumed evidence
  Oracle        argmax_d Y(d)

The headroom that matters is Oracle - StateAware: if it collapses, PIU measures
a limit of query-local relevance; if it persists, the limit is not merely
state-blindness.
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
    qg = [sum(v) / len(v) for v in per_q.values()]
    rng = random.Random(seed)
    b = sorted(sum(qg[rng.randrange(len(qg))] for _ in qg) / len(qg)
               for _ in range(n_boot))
    return {"mean": sum(qg) / len(qg),
            "ci95": [b[int(.025 * n_boot)], b[int(.975 * n_boot)]]}


def main():
    R = {}
    for name, tag, gate_dir in EVAL:
        sa_path = f"results/state_aware_relevance/{tag}.jsonl"
        if not os.path.exists(sa_path):
            print(f"[{name}] missing {sa_path}, skipped")
            continue
        sa = {r["state_id"]: r["ranking"] for r in read_jsonl(sa_path)}
        ce = {r["state_id"]: r["scores"]
              for r in read_jsonl(os.path.join(gate_dir,
                                               "candidate_ce_scores.jsonl"))}
        by = defaultdict(list)
        for u in read_jsonl(os.path.join(gate_dir, "utilities.jsonl")):
            by[(u["question_id"], u["state_idx"])].append(u)
        absf = {k: defaultdict(list) for k in ("ce", "state_aware", "oracle")}
        head = {k: defaultdict(list) for k in ("ce", "state_aware")}
        agree, opp_sa, n = [], [], 0
        for (qid, sidx), rr in by.items():
            sid = f"{qid}::s{sidx}"
            sc, rank = ce.get(sid), sa.get(sid)
            if not sc or not rank:
                continue
            f1 = {r["doc"]: r["arm_d"]["f1"] for r in rr}
            cand = [d for d in f1 if d in sc]
            if len(cand) < 2:
                continue
            n += 1
            d_ce = max(cand, key=lambda d: sc[d])
            d_sa = next((d for d in rank if d in f1), d_ce)
            d_or = max(cand, key=lambda d: f1[d])
            agree.append(float(d_sa == d_ce))
            opp_sa.append(float(f1[d_or] - f1[d_sa] > EPS))
            absf["ce"][qid].append(f1[d_ce])
            absf["state_aware"][qid].append(f1[d_sa])
            absf["oracle"][qid].append(f1[d_or])
            head["ce"][qid].append(f1[d_or] - f1[d_ce])
            head["state_aware"][qid].append(f1[d_or] - f1[d_sa])
        R[name] = {
            "n_states": n,
            "top1_agreement_with_ce": float(np.mean(agree)),
            "state_level_failure_rate_vs_state_aware": float(np.mean(opp_sa)),
            "abs_f1": {k: boot(v) for k, v in absf.items()},
            "headroom_oracle_minus": {k: boot(v) for k, v in head.items()},
        }
        print(f"[{name}] done", flush=True)
    os.makedirs("results/state_aware_relevance", exist_ok=True)
    out = "results/state_aware_relevance/REPORT.json"
    with open(out, "w") as f:
        json.dump(R, f, indent=2)
    print(json.dumps(R, indent=2))
    print("STATE_AWARE_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
