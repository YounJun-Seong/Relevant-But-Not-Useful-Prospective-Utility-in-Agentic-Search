"""Independent relevance quality of each selector (reviewer (a)).

Downstream F1 alone cannot show that BGE is a *stronger relevance model* than
the MiniLM cross-encoder. This measures relevance quality directly, against the
dataset's annotated supporting documents, on the retained intervention set
D_t: gold hit@1 and nDCG.

States whose candidate set contains no gold document are excluded from hit@1
and nDCG (the ranking task is undefined there) and counted separately.
Paired question-level bootstrap for selector-vs-CE differences.
"""
import json
import math
import os
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.data_utils import read_jsonl, norm_title

EVAL = [("HotpotQA dev", "gate2", "results/gate2"),
        ("HotpotQA lockbox", "lockbox", "results/lockbox"),
        ("2WikiMultiHopQA", "wiki2", "results/wiki2"),
        ("Qwen2.5-14B", "xagent", "results/xagent"),
        ("MuSiQue", "musique", "results/musique")]
SELS = ["random", "ce", "strong_qt", "strong_state", "llm_state"]


def ndcg(order, gold, k=None):
    k = k or len(order)
    dcg = sum((1.0 if order[i] in gold else 0.0) / math.log2(i + 2)
              for i in range(min(k, len(order))))
    n_rel = min(len(gold & set(order)), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
    return dcg / idcg if idcg > 0 else None


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
    for name, tag, gate_dir in EVAL:
        with open(os.path.join(gate_dir, "questions.json")) as f:
            qmeta = json.load(f)
        gold_of = {r["question_id"]: {norm_title(t) for t in r["gold_titles"]}
                   for r in read_jsonl(qmeta["pre_path"])}
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

        hit = {s: defaultdict(list) for s in SELS}
        nd = {s: defaultdict(list) for s in SELS}
        rng = random.Random(7)
        n_states = n_with_gold = 0
        for (qid, sidx), rr in by.items():
            sid = f"{qid}::s{sidx}"
            sc, st, rk = ce.get(sid), strong.get(sid), llm.get(sid)
            if not sc:
                continue
            cand = [r["doc"] for r in rr if r["doc"] in sc]
            if len(cand) < 2:
                continue
            n_states += 1
            gold = gold_of[qid]
            present = {d for d in cand if norm_title(d) in gold}
            if not present:
                continue
            n_with_gold += 1
            orders = {
                "random": rng.sample(cand, len(cand)),
                "ce": sorted(cand, key=lambda d: -sc[d]),
            }
            if st and all(d in st for d in cand):
                orders["strong_qt"] = sorted(
                    cand, key=lambda d: -st[d]["strong_qt"])
                orders["strong_state"] = sorted(
                    cand, key=lambda d: -st[d]["strong_state"])
            if rk:
                seen = [d for d in rk if d in cand]
                orders["llm_state"] = seen + [d for d in cand if d not in seen]
            for s, order in orders.items():
                hit[s][qid].append(float(order[0] in present))
                v = ndcg(order, present)
                if v is not None:
                    nd[s][qid].append(v)

        entry = {"n_states": n_states, "n_states_with_gold_in_Dt": n_with_gold,
                 "gold_coverage": n_with_gold / max(n_states, 1),
                 "hit_at_1": {}, "ndcg": {}, "paired_vs_ce": {}}
        for s in SELS:
            if not hit[s]:
                continue
            entry["hit_at_1"][s] = boot(hit[s])
            entry["ndcg"][s] = boot(nd[s])
            if s != "ce":
                entry["paired_vs_ce"][s] = {
                    "hit_at_1": boot({q: [a - b for a, b in
                                          zip(hit[s][q], hit["ce"][q])]
                                      for q in hit[s]}),
                    "ndcg": boot({q: [a - b for a, b in
                                      zip(nd[s][q], nd["ce"][q])]
                                  for q in nd[s]}),
                }
        R[name] = entry
        print(f"[{name}] {n_with_gold}/{n_states} states with gold in D_t",
              flush=True)

    os.makedirs("results/selector_relevance", exist_ok=True)
    out = "results/selector_relevance/report_Dt.json"
    with open(out, "w") as f:
        json.dump(R, f, indent=2)
    print(f"-> {out}")
    print("SELECTOR_RELEVANCE_DONE")


if __name__ == "__main__":
    main()
