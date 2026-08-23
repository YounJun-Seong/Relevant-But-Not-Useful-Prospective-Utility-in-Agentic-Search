"""State-aware relevance baseline (reviewer W4).

The confirmatory relevance baseline scores (q_t, d) with a query-local
cross-encoder, so a reviewer can ask whether the measured headroom reflects a
limit of relevance in general or only of *state-blind* relevance. This baseline
gives an LLM the original question, the current query, and the evidence already
consumed, and asks it to rank the unseen candidates by relevance to the current
unresolved information need.

It is deliberately a RELEVANCE model, not a utility model: the prompt forbids
predicting downstream answer accuracy or simulating future search. Greedy
decoding, one listwise call per state.
"""
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.data_utils import read_jsonl, write_jsonl

PROMPT = """You are ranking documents for an ongoing multi-step search.

Original question:
{q0}

Current search query:
{qt}

Evidence already observed:
{evidence}

Candidate documents (not yet seen by the agent):
{candidates}

Rank ALL candidates by how relevant they are to the agent's CURRENT unresolved
information need.
Do not predict downstream answer accuracy.
Do not simulate future search.
Respond with only the candidate numbers, most relevant first, comma-separated.
"""


def parse_ranking(text, n):
    nums = [int(x) for x in re.findall(r"\d+", text)]
    seen, out = set(), []
    for x in nums:
        if 1 <= x <= n and x not in seen:
            seen.add(x)
            out.append(x)
    for x in range(1, n + 1):
        if x not in seen:
            out.append(x)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-dirs", default="results/lockbox")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--index", default="data/bm25_index")
    ap.add_argument("--gpu-mem", type=float, default=0.40)
    ap.add_argument("--snippet", type=int, default=300)
    ap.add_argument("--ev-snippet", type=int, default=160)
    ap.add_argument("--max-evidence", type=int, default=6)
    ap.add_argument("--out", default="results/state_aware_relevance")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from piu.retriever import BM25Retriever
    from piu.state_delta import state_context
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    retriever = BM25Retriever(args.index)
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=16384, seed=42)
    tok = AutoTokenizer.from_pretrained(args.model)
    sp = SamplingParams(temperature=0.0, max_tokens=96)

    for gate_dir in args.gate_dirs.split(","):
        tag = os.path.basename(gate_dir.rstrip("/"))
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

        prompts, keys, t0 = [], [], time.time()
        for (qid, sidx), rr in by.items():
            rec = base_pre[qid]
            qt, ev_texts, _, _ = state_context(rec, sidx, retriever)
            seen = []
            for e in rec["events"][: sidx + 1]:
                if e["type"] == "info":
                    seen.extend(e.get("titles", []))
            ev = []
            for t in seen[-args.max_evidence:]:
                d = retriever.doc_by_title(t)
                ev.append(f"- {t}: {d['text'][:args.ev_snippet]}" if d else f"- {t}")
            cands, titles = [], []
            for k, r in enumerate(rr):
                d = retriever.doc_by_title(r["doc"])
                if d is None:
                    continue
                titles.append(r["doc"])
                cands.append(f"({len(titles)}) {d['title']}: "
                             f"{d['text'][:args.snippet]}")
            if len(titles) < 2:
                continue
            msg = PROMPT.format(q0=rec["question"], qt=qt,
                                evidence="\n".join(ev) if ev else "(none)",
                                candidates="\n\n".join(cands))
            prompts.append(tok.apply_chat_template(
                [{"role": "user", "content": msg}], tokenize=False,
                add_generation_prompt=True))
            keys.append((qid, sidx, titles))
        print(f"[{tag}] {len(prompts)} states", flush=True)
        outs = llm.generate(prompts, sp, use_tqdm=False)
        rows = []
        for (qid, sidx, titles), o in zip(keys, outs):
            text = o.outputs[0].text
            order = parse_ranking(text, len(titles))
            rows.append({"state_id": f"{qid}::s{sidx}", "question_id": qid,
                         "state_idx": sidx,
                         "ranking": [titles[i - 1] for i in order],
                         "raw": text[:200]})
        write_jsonl(out_path, rows)
        print(f"[{tag}] done in {time.time()-t0:.0f}s -> {out_path}", flush=True)
    print("STATE_AWARE_DONE")


if __name__ == "__main__":
    main()
