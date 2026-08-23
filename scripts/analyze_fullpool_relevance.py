"""Gold hit@1 / nDCG@10 on the full pool C_t, per selector (reviewer (a))."""
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
SELS = ["random", "ce", "bge_qt", "bge_state"]


def ndcg_at_k(order, gold, k=10):
    dcg = sum((1.0 if order[i] in gold else 0.0) / math.log2(i + 2)
              for i in range(min(k, len(order))))
    n_rel = min(len(gold), k)
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
        path = f"results/selector_relevance/fullpool_{tag}.jsonl"
        if not os.path.exists(path):
            print(f"[{name}] missing, skipped")
            continue
        with open(os.path.join(gate_dir, "questions.json")) as f:
            qmeta = json.load(f)
        gold_of = {r["question_id"]: {norm_title(t) for t in r["gold_titles"]}
                   for r in read_jsonl(qmeta["pre_path"])}
        hit = {s: defaultdict(list) for s in SELS}
        nd = {s: defaultdict(list) for s in SELS}
        rng = random.Random(7)
        n = n_gold = 0
        sizes = []
        for r in read_jsonl(path):
            n += 1
            sizes.append(r["pool_size"])
            titles = r["titles"]
            gold = {t for t in titles if norm_title(t) in gold_of[r["question_id"]]}
            if not gold:
                continue
            n_gold += 1
            orders = {
                "random": rng.sample(titles, len(titles)),
                "ce": [titles[i] for i in np.argsort(-np.array(r["ce"]))],
                "bge_qt": [titles[i] for i in np.argsort(-np.array(r["bge_qt"]))],
                "bge_state": [titles[i]
                              for i in np.argsort(-np.array(r["bge_state"]))],
            }
            for s, order in orders.items():
                hit[s][r["question_id"]].append(float(order[0] in gold))
                v = ndcg_at_k(order, gold, 10)
                if v is not None:
                    nd[s][r["question_id"]].append(v)
        entry = {"n_states": n, "n_states_with_gold_in_pool": n_gold,
                 "gold_coverage": n_gold / max(n, 1),
                 "pool_size_mean": float(np.mean(sizes)),
                 "pool_size_sd": float(np.std(sizes)),
                 "hit_at_1": {}, "ndcg_at_10": {}, "paired_vs_ce": {}}
        for s in SELS:
            entry["hit_at_1"][s] = boot(hit[s])
            entry["ndcg_at_10"][s] = boot(nd[s])
            if s != "ce":
                entry["paired_vs_ce"][s] = {
                    "hit_at_1": boot({q: [a - b for a, b in
                                          zip(hit[s][q], hit["ce"][q])]
                                      for q in hit[s]}),
                    "ndcg_at_10": boot({q: [a - b for a, b in
                                            zip(nd[s][q], nd["ce"][q])]
                                        for q in nd[s]}),
                }
        R[name] = entry
        print(f"[{name}] {n_gold}/{n} states with gold in pool", flush=True)
    out = "results/selector_relevance/fullpool_report.json"
    with open(out, "w") as f:
        json.dump(R, f, indent=2)
    print(f"-> {out}")
    print("FULLPOOL_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
