"""Intervene on the stronger selectors' FULL-POOL argmax when it is not retained.

Reviewer: BGE's relevance advantage is shown on C_t, but interventions exist
only for D_t, and BGE's full-pool top document is retained in D_t for only a
minority of states. This closes the gap with one new arm per (state, selector):
the full-pool argmax of BGE (q_t) and of BGE (state-aware) is injected at the
frozen state and the SAME frozen greedy policy is resumed, exactly as for the
retained candidates. States whose argmax is already in D_t reuse the stored
outcome.

Outcome: each selector's realized downstream F1 when it is allowed to pick
from the full pool, and the oracle over the expanded set D_t U {new arms}.
Post-hoc exploratory; the RQ1 endpoint is untouched.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.agent import AgentRunner, Episode
from piu.data_utils import read_jsonl, write_jsonl
from piu.intervene import make_arm_events
from piu.utility import arm_outcome

TAGS = {"gate2": "results/gate2", "lockbox": "results/lockbox",
        "wiki2": "results/wiki2", "xagent": "results/xagent",
        "musique": "results/musique"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="lockbox,gate2,wiki2,xagent,musique")
    ap.add_argument("--fullpool-dir", default="results/selector_relevance")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--index", default="data/bm25_index")
    ap.add_argument("--gpu-mem", type=float, default=0.40)
    ap.add_argument("--out", default="results/choice_space")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from piu.retriever import BM25Retriever
    from vllm import LLM
    from transformers import AutoTokenizer
    retriever = BM25Retriever(args.index)

    # ---- plan: one arm per (state, selector) whose full-pool argmax is new
    plans = {}
    for tag in args.tags.split(","):
        gate_dir = TAGS[tag]
        with open(os.path.join(gate_dir, "questions.json")) as f:
            qmeta = json.load(f)
        base_pre = {r["question_id"]: r for r in read_jsonl(qmeta["pre_path"])}
        Dt = {}
        for u in read_jsonl(os.path.join(gate_dir, "utilities.jsonl")):
            Dt.setdefault(f"{u['question_id']}::s{u['state_idx']}", set()).add(u["doc"])
        plan, seen = [], set()
        for r in read_jsonl(os.path.join(args.fullpool_dir, f"fullpool_{tag}.jsonl")):
            sid = r["state_id"]
            if sid not in Dt:
                continue
            for sel in ("bge_qt", "bge_state"):
                top = r["titles"][int(np.argmax(r[sel]))]
                if top in Dt[sid] or (sid, top) in seen:
                    continue
                seen.add((sid, top))
                rec = base_pre[r["question_id"]]
                plan.append({"arm_id": f"{sid}::fullpool::{top}",
                             "question_id": r["question_id"],
                             "state_idx": r["state_idx"], "doc": top,
                             "selectors": [s for s in ("bge_qt", "bge_state")
                                           if r["titles"][int(np.argmax(r[s]))] == top],
                             "events": make_arm_events(rec, r["state_idx"],
                                                       injected_titles=[top])})
        plans[tag] = (plan, base_pre)
        print(f"[{tag}] {len(plan)} new arms to roll out", flush=True)

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=16384, seed=42)
    tok = AutoTokenizer.from_pretrained(args.model)
    runner = AgentRunner(llm, tok, retriever)       # frozen greedy policy

    for tag, (plan, base_pre) in plans.items():
        out_path = os.path.join(args.out, f"fullpool_arms_{tag}.jsonl")
        if os.path.exists(out_path):
            print(f"[{tag}] cached", flush=True)
            continue
        t0 = time.time()
        eps = [Episode(p["question_id"], base_pre[p["question_id"]]["question"],
                       prefix_events=p["events"], meta={"arm_id": p["arm_id"]})
               for p in plan]
        runner.run(eps)
        rows = []
        for ep, p in zip(eps, plan):
            rec = ep.to_record()
            for k in ("arm_id", "state_idx", "doc", "selectors"):
                rec[k] = p[k]
            rec["outcome"] = arm_outcome(
                rec, base_pre[p["question_id"]]["answer_gold"])
            rows.append(rec)
        write_jsonl(out_path, rows)
        print(f"[{tag}] {len(rows)} arms in {time.time()-t0:.0f}s -> {out_path}",
              flush=True)
    print("EXPAND_CHOICE_SPACE_DONE")


if __name__ == "__main__":
    main()
