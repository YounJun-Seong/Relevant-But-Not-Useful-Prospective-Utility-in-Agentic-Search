"""Exact CE scores for every evaluated candidate (fixes the top-20 truncation).

ce_rankings.jsonl stores only the CE top-20 of the ~109-document pool, so any
candidate outside it is indistinguishable ('unranked'). Pair-level CE
comparisons that fall back to a sentinel rank therefore treat 'CE has no stored
opinion' as 'CE disagrees'. This script re-scores the (q_t, candidate) pairs
with the identical cross-encoder, giving a total order over the candidate set.
Deterministic: same model, same inputs as the original run.
"""
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.data_utils import read_jsonl, write_jsonl

DIRS = {"gate2": "results/gate2", "gate3": "results/gate3",
        "lockbox": "results/lockbox", "wiki2": "results/wiki2",
        "xagent": "results/xagent", "musique": "results/musique"}


def main():
    from piu.retriever import BM25Retriever
    from piu.dense import CEScorer
    retriever = BM25Retriever("data/bm25_index")
    ce = CEScorer()
    for name, gate_dir in DIRS.items():
        out_path = os.path.join(gate_dir, "candidate_ce_scores.jsonl")
        if os.path.exists(out_path):
            print(f"[{name}] cached", flush=True)
            continue
        with open(os.path.join(gate_dir, "questions.json")) as f:
            qmeta = json.load(f)
        base_pre = {r["question_id"]: r for r in read_jsonl(qmeta["pre_path"])}
        by_state = defaultdict(list)
        for u in read_jsonl(os.path.join(gate_dir, "utilities.jsonl")):
            by_state[(u["question_id"], u["state_idx"])].append(u)
        rows, t0 = [], time.time()
        for (qid, sidx), rr in by_state.items():
            rec = base_pre[qid]
            qt = rec["events"][sidx].get("query") or rec["question"]
            docs, titles = [], []
            for r in rr:
                d = retriever.doc_by_title(r["doc"])
                if d is None:
                    continue
                titles.append(r["doc"])
                docs.append(d["title"] + ". " + d["text"])
            if not docs:
                continue
            scores = ce.score(qt, docs)
            rows.append({"state_id": f"{qid}::s{sidx}", "question_id": qid,
                         "state_idx": sidx, "query": qt,
                         "scores": dict(zip(titles, scores))})
        write_jsonl(out_path, rows)
        print(f"[{name}] {len(rows)} states in {time.time()-t0:.0f}s -> {out_path}",
              flush=True)
    print("CE_SCORES_DONE")


if __name__ == "__main__":
    main()
