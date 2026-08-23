"""Relevance quality of each selector on the FULL candidate pool C_t (reviewer (a)).

On the eight retained candidates the ranking task saturates, so downstream F1
alone cannot establish that BGE is a stronger relevance model than the MiniLM
cross-encoder. Here every selector ranks the entire pool C_t (~109 unseen
documents per state, reconstructed exactly as at intervention-planning time)
and is scored against the dataset's annotated supporting documents.

Reported: gold hit@1 and nDCG@10, with paired question-level bootstrap for the
selector-vs-CE difference. States whose pool contains no gold document are
excluded from the ranking metrics and counted separately.

Dense queries are batched across states; the brute-force E5 pass over 5.2M
embeddings then amortises over many queries instead of running per state.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.data_utils import read_jsonl, write_jsonl, norm_title
from piu.intervene import seen_titles_upto

TAGS = {"gate2": "results/gate2", "lockbox": "results/lockbox",
        "wiki2": "results/wiki2", "xagent": "results/xagent",
        "musique": "results/musique"}


def ndcg_at_k(order, gold, k=10):
    dcg = sum((1.0 if order[i] in gold else 0.0) / math.log2(i + 2)
              for i in range(min(k, len(order))))
    n_rel = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
    return dcg / idcg if idcg > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="lockbox,gate2,wiki2,xagent,musique")
    ap.add_argument("--index", default="data/bm25_index")
    ap.add_argument("--emb", default="data/e5_embeddings.fp16.npy")
    ap.add_argument("--dense-batch", type=int, default=128)
    ap.add_argument("--ce-batch", type=int, default=128)
    ap.add_argument("--bge-batch", type=int, default=64)
    ap.add_argument("--doc-chars", type=int, default=500,
                    help="BGE passage truncation (chars); D_t scoring used 600")
    ap.add_argument("--ce-full-text", action="store_true",
                    help="score CE on the full passage (tokenizer-truncated), "
                         "matching candidate_ce_scores.py exactly")
    ap.add_argument("--max-evidence", type=int, default=8)
    ap.add_argument("--out", default="results/selector_relevance")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from piu.retriever import BM25Retriever
    from piu.dense import E5DenseRetriever, CEScorer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    retriever = BM25Retriever(args.index)
    dense = E5DenseRetriever(args.emb, retriever.titles)
    ce = CEScorer()
    btok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
    bmodel = AutoModelForSequenceClassification.from_pretrained(
        "BAAI/bge-reranker-v2-m3", torch_dtype=torch.float16).to("cuda").eval()

    def bge(pairs, batch):
        out = []
        with torch.inference_mode():
            for i in range(0, len(pairs), batch):
                blk = pairs[i:i + batch]
                enc = btok([a for a, _ in blk], [b for _, b in blk],
                           padding=True, truncation=True, max_length=512,
                           return_tensors="pt").to("cuda")
                out.extend(bmodel(**enc).logits.squeeze(-1)
                           .float().cpu().tolist())
        return out

    for tag in args.tags.split(","):
        gate_dir = TAGS[tag]
        out_path = os.path.join(args.out, f"fullpool_{tag}.jsonl")
        if os.path.exists(out_path):
            print(f"[{tag}] cached", flush=True)
            continue
        t0 = time.time()
        with open(os.path.join(gate_dir, "questions.json")) as f:
            qmeta = json.load(f)
        base_pre = {r["question_id"]: r for r in read_jsonl(qmeta["pre_path"])}
        states = sorted({(u["question_id"], u["state_idx"])
                         for u in read_jsonl(os.path.join(gate_dir,
                                                          "utilities.jsonl"))})
        # ---- phase 1: per-state queries, exactly as build_pool_v2 ----
        specs = []
        for qid, sidx in states:
            rec = base_pre[qid]
            qt = rec["events"][sidx].get("query") or rec["question"]
            seen = seen_titles_upto(rec, sidx)
            specs.append({"qid": qid, "sidx": sidx, "qt": qt,
                          "q0": rec["question"], "seen": seen,
                          "bm": [qt, rec["question"]] + list(seen[:6])})
        # ---- phase 2: BM25 (batched) ----
        flat, span = [], []
        for s in specs:
            span.append((len(flat), len(s["bm"])))
            flat.extend(s["bm"])
        bm_res = []
        for i in range(0, len(flat), 512):
            bm_res.extend(retriever.search_batch(flat[i:i + 512], k=20))
        print(f"[{tag}] BM25 {len(flat)} queries {time.time()-t0:.0f}s",
              flush=True)
        # ---- phase 3: dense (batched across states) ----
        dq = []
        for s in specs:
            dq.extend([s["qt"], s["q0"]])
        de_res = []
        for i in range(0, len(dq), args.dense_batch):
            de_res.extend(dense.search_batch(dq[i:i + args.dense_batch], k=20))
            if (i // args.dense_batch) % 5 == 0:
                print(f"  dense {i}/{len(dq)} ({time.time()-t0:.0f}s)",
                      flush=True)
        print(f"[{tag}] dense done {time.time()-t0:.0f}s", flush=True)
        # ---- phase 4: assemble pools ----
        rows = []
        for si, s in enumerate(specs):
            start, nbm = span[si]
            seen_set = set(s["seen"])
            pool = {}
            for r in bm_res[start:start + nbm]:
                for d in r:
                    if d["title"] not in seen_set:
                        pool.setdefault(d["title"], d["idx"])
            for r in de_res[2 * si:2 * si + 2]:
                for d in r:
                    if d["title"] not in seen_set:
                        pool.setdefault(d["title"], d["idx"])
            s["pool"] = list(pool)
        sizes = [len(s["pool"]) for s in specs]
        print(f"[{tag}] pools mean {np.mean(sizes):.1f} sd {np.std(sizes):.1f}",
              flush=True)
        # ---- phase 5: score every pool document ----
        for si, s in enumerate(specs):
            rec = base_pre[s["qid"]]
            docs, docs_ce, titles = [], [], []
            for t in s["pool"]:
                d = retriever.doc_by_title(t)
                if d is None:
                    continue
                titles.append(t)
                docs.append(d["title"] + ". " + d["text"][:args.doc_chars])
                docs_ce.append(d["title"] + ". " + d["text"]
                               if args.ce_full_text else docs[-1])
            if len(titles) < 2:
                continue
            q_state = (f"{rec['question']} [current query] {s['qt']} "
                       f"[already seen] "
                       f"{'; '.join(s['seen'][-args.max_evidence:])}")
            ce_s = ce.score(s["qt"], docs_ce)
            bq = bge([(s["qt"], d) for d in docs], args.bge_batch)
            bs = bge([(q_state, d) for d in docs], args.bge_batch)
            rows.append({"state_id": f"{s['qid']}::s{s['sidx']}",
                         "question_id": s["qid"], "state_idx": s["sidx"],
                         "pool_size": len(titles), "titles": titles,
                         "ce": ce_s, "bge_qt": bq, "bge_state": bs})
            if si % 50 == 0:
                print(f"  score {si}/{len(specs)} ({time.time()-t0:.0f}s)",
                      flush=True)
        write_jsonl(out_path, rows)
        print(f"[{tag}] {len(rows)} states in {time.time()-t0:.0f}s -> {out_path}",
              flush=True)
    print("FULLPOOL_SCORING_DONE")


if __name__ == "__main__":
    main()
