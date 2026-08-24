"""Post-hoc cross-family replication evaluation (microsoft/phi-4).

Same estimators as the confirmatory settings: question-level paired bootstrap
of Oracle-CE downstream F1; Opp.-Base and Opp.-CE rates; exact-expectation
random selector; and, where BGE scores on D_t exist, the Oracle-selector
headroom for the strong and state-aware BGE selectors. Representative
(non-enriched) HotpotQA questions only.
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.data_utils import read_jsonl

EPS = 1e-9


def boot(per_q, n_boot=10000, seed=7):
    qg = [float(np.mean(v)) for v in per_q.values()]
    rng = random.Random(seed)
    b = sorted(sum(qg[rng.randrange(len(qg))] for _ in qg) / len(qg)
               for _ in range(n_boot))
    return {"mean": sum(qg) / len(qg),
            "ci95": [b[int(.025 * n_boot)], b[int(.975 * n_boot)]],
            "n_questions": len(qg)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-dir", default="results/xfamily_phi4")
    ap.add_argument("--tag", default="xfamily_phi4")
    args = ap.parse_args()
    gd = args.gate_dir
    with open(os.path.join(gd, "questions.json")) as f:
        qmeta = json.load(f)
    pre = read_jsonl(qmeta["pre_path"])
    base_em = float(np.mean([r["outcome"]["em"] for r in pre]))
    base_f1 = float(np.mean([r["outcome"]["f1"] for r in pre]))
    strata = {s["question_id"]: s["stratum"] for s in qmeta["selected"]}
    ce = {r["state_id"]: r["scores"]
          for r in read_jsonl(os.path.join(gd, "candidate_ce_scores.jsonl"))}
    bp = f"results/strong_reranker/{args.tag}.jsonl"
    bge = {r["state_id"]: r["scores"] for r in read_jsonl(bp)} \
        if os.path.exists(bp) else {}
    by = defaultdict(list)
    for u in read_jsonl(os.path.join(gd, "utilities.jsonl")):
        by[(u["question_id"], u["state_idx"])].append(u)

    sels = ["random", "ce", "bge_qt", "bge_state", "oracle"]
    absf = {s: defaultdict(list) for s in sels}
    head = {s: defaultdict(list) for s in sels}
    opp_base, opp_ce, n = [], [], 0
    for (qid, sidx), rr in by.items():
        sid = f"{qid}::s{sidx}"
        sc = ce.get(sid)
        if not sc:
            continue
        cand = [r for r in rr if r["doc"] in sc]
        if len(cand) < 2:
            continue
        n += 1
        f1 = {r["doc"]: r["arm_d"]["f1"] for r in cand}
        best = max(f1.values())
        d_ce = max(f1, key=lambda d: sc[d])
        picks = {"random": None, "ce": d_ce, "oracle": max(f1, key=f1.get)}
        st = bge.get(sid)
        if st and all(d in st for d in f1):
            picks["bge_qt"] = max(f1, key=lambda d: st[d]["strong_qt"])
            picks["bge_state"] = max(f1, key=lambda d: st[d]["strong_state"])
        opp_base.append(float(max(r["U_task_raw_f1"] for r in cand) > EPS))
        opp_ce.append(float(best - f1[d_ce] > EPS))
        for s, d in picks.items():
            y = float(np.mean(list(f1.values()))) if s == "random" else f1[d]
            absf[s][qid].append(y)
            head[s][qid].append(best - y)
    R = {
        "model": "microsoft/phi-4", "n_states": n,
        "n_questions": len({q for q, _ in by}),
        "strata": dict(zip(*np.unique(list(strata.values()), return_counts=True))),
        "base_em_presample": base_em, "base_f1_presample": base_f1,
        "Opp_base": float(np.mean(opp_base)), "Opp_CE": float(np.mean(opp_ce)),
        "abs_f1": {s: boot(v) for s, v in absf.items() if v},
        "headroom_oracle_minus": {s: boot(v) for s, v in head.items() if v and s != "oracle"},
        "paired_ce_minus_random": boot({q: [a - b for a, b in zip(absf["ce"][q], absf["random"][q])]
                                        for q in absf["ce"]}),
    }
    for s in ("bge_qt", "bge_state"):
        if absf[s]:
            R[f"paired_{s}_minus_ce"] = boot({q: [a - b for a, b in zip(absf[s][q], absf["ce"][q])]
                                              for q in absf[s]})
    R["strata"] = {str(k): int(v) for k, v in R["strata"].items()}
    with open(os.path.join(gd, "XFAMILY_REPORT.json"), "w") as f:
        json.dump(R, f, indent=2)
    print(json.dumps(R, indent=2))
    print("XFAMILY_EVAL_DONE")


if __name__ == "__main__":
    main()
