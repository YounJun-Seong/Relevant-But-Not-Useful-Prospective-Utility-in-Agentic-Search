"""Strong and state-aware relevance baselines (reviewer W1 baseline weakness / W4).

The confirmatory relevance baseline is ms-marco-MiniLM-L6-v2 scored against the
current query only, so two objections follow: the reranker is weak, and it is
state-blind. Both are tested with a strong open reranker
(BAAI/bge-reranker-v2-m3, 568M) under two query formulations:

  strong_qt      the current query alone           (strong but state-blind)
  strong_state   original question + current query + already-consumed titles
                 (strong and state-aware)

Scoring only; no agent rollouts are needed because every candidate's
intervention outcome is already stored.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.data_utils import read_jsonl, write_jsonl

DIRS = {"gate2": "results/gate2", "lockbox": "results/lockbox",
        "wiki2": "results/wiki2", "xagent": "results/xagent",
        "musique": "results/musique"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--index", default="data/bm25_index")
    ap.add_argument("--tags", default="lockbox,gate2,wiki2,xagent,musique")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--doc-chars", type=int, default=600)
    ap.add_argument("--max-evidence", type=int, default=8)
    ap.add_argument("--out", default="results/strong_reranker")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from piu.retriever import BM25Retriever
    retriever = BM25Retriever(args.index)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, torch_dtype=torch.float16).to("cuda").eval()

    def score(pairs):
        out = []
        with torch.inference_mode():
            for i in range(0, len(pairs), args.batch):
                blk = pairs[i:i + args.batch]
                enc = tok([a for a, _ in blk], [b for _, b in blk],
                          padding=True, truncation=True, max_length=512,
                          return_tensors="pt").to("cuda")
                out.extend(model(**enc).logits.squeeze(-1).float().cpu().tolist())
        return out

    for tag in args.tags.split(","):
        gate_dir = DIRS[tag]
        out_path = os.path.join(args.out, f"{tag}.jsonl")
        if os.path.exists(out_path):
            print(f"[{tag}] cached", flush=True)
            continue
        with open(os.path.join(gate_dir, "questions.json")) as f:
            qmeta = json.load(f)
        base_pre = {r["question_id"]: r for r in read_jsonl(qmeta["pre_path"])}
        by = defaultdict(list)
        for u in read_jsonl(os.path.join(gate_dir, "utilities.jsonl")):
            by[(u["question_id"], u["state_idx"])].append(u)
        rows, t0 = [], time.time()
        qt_pairs, st_pairs, keys = [], [], []
        for (qid, sidx), rr in by.items():
            rec = base_pre[qid]
            qt = rec["events"][sidx].get("query") or rec["question"]
            seen = []
            for e in rec["events"][: sidx + 1]:
                if e["type"] == "info":
                    seen.extend(e.get("titles", []))
            q_state = (f"{rec['question']} [current query] {qt} "
                       f"[already seen] {'; '.join(seen[-args.max_evidence:])}")
            for r in rr:
                d = retriever.doc_by_title(r["doc"])
                if d is None:
                    continue
                passage = d["title"] + ". " + d["text"][:args.doc_chars]
                qt_pairs.append((qt, passage))
                st_pairs.append((q_state, passage))
                keys.append((qid, sidx, r["doc"]))
        print(f"[{tag}] scoring {len(keys)} candidates", flush=True)
        s_qt = score(qt_pairs)
        s_st = score(st_pairs)
        agg = defaultdict(dict)
        for (qid, sidx, doc), a, b in zip(keys, s_qt, s_st):
            agg[(qid, sidx)][doc] = {"strong_qt": a, "strong_state": b}
        for (qid, sidx), d in agg.items():
            rows.append({"state_id": f"{qid}::s{sidx}", "question_id": qid,
                         "state_idx": sidx, "scores": d})
        write_jsonl(out_path, rows)
        print(f"[{tag}] {len(rows)} states in {time.time()-t0:.0f}s", flush=True)
    print("STRONG_RERANKER_DONE")


if __name__ == "__main__":
    main()
