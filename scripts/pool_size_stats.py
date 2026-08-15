"""Recompute per-state candidate-pool sizes (BM25∪Dense union before CE
ranking) for paper Method placeholders — pool size was not logged at run time
(ce_rankings.jsonl truncates to top-20). Reconstruction is deterministic:
build_pool_v2 uses no RNG."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.data_utils import read_jsonl
from piu.intervene import state_points, seen_titles_upto
from piu.intervene2 import build_pool_v2
from piu.retriever import BM25Retriever
from piu.dense import E5DenseRetriever

SETTINGS = {"gate2": "results/gate2", "lockbox": "results/lockbox"}


def main():
    retriever = BM25Retriever("data/bm25_index")
    dense = E5DenseRetriever("data/e5_embeddings.fp16.npy", retriever.titles)
    out = {}
    for name, dirp in SETTINGS.items():
        with open(f"{dirp}/questions.json") as f:
            meta = json.load(f)
        base_rows = {r["question_id"]: r for r in read_jsonl(meta["pre_path"])}
        wanted = {}  # state_id -> True (states that actually entered analysis)
        for r in read_jsonl(f"{dirp}/ce_rankings.jsonl"):
            wanted[r["state_id"]] = True
        sizes = []
        for s in meta["selected"]:
            rec = base_rows[s["question_id"]]
            for sidx in state_points(rec):
                sid = f"{rec['question_id']}::s{sidx}"
                if sid not in wanted:
                    continue
                seen = seen_titles_upto(rec, sidx)
                qt = rec["events"][sidx].get("query") or rec["question"]
                pool = build_pool_v2(retriever, dense, rec["question"], qt, seen)
                sizes.append(len(pool))
        a = np.array(sizes)
        out[name] = {"n_states": int(len(a)), "mean": float(a.mean()),
                     "sd": float(a.std(ddof=1)), "median": float(np.median(a)),
                     "min": int(a.min()), "max": int(a.max())}
        print(name, out[name], flush=True)
    with open("results/pool_size_stats.json", "w") as f:
        json.dump(out, f, indent=2)
    print("POOL_STATS_DONE")


if __name__ == "__main__":
    main()
