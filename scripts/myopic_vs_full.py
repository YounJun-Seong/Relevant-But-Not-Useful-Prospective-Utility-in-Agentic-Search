"""Myopic utility vs full PIU on the HotpotQA lockbox (post-hoc exploratory).

Same frozen state, same candidate, same frozen greedy policy. Two conditions:
  full    inject d, resume the normal search continuation  (stored arms)
  myopic  inject d, FORBID further retrieval, answer immediately
The myopic condition reuses the harness's forced-answer mechanism: a forced
information block tells the agent the search limit is reached, so the only
legal action is a final answer. If the agent searches anyway the episode ends
with an empty answer, exactly as in the confirmatory protocol.

Reports: (state,candidate)-level correlation of the two utilities, per-state
winner agreement and rank correlation, oracle headroom under each condition,
and -- the practically relevant number -- the full-PIU headroom captured when
the candidate is SELECTED by myopic utility.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.agent import AgentRunner, Episode
from piu.data_utils import read_jsonl, write_jsonl
from piu.intervene import make_arm_events
from piu.utility import arm_outcome


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-dir", default="results/lockbox")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--index", default="data/bm25_index")
    ap.add_argument("--gpu-mem", type=float, default=0.40)
    ap.add_argument("--out", default="results/myopic")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "myopic_arms.jsonl")
    if os.path.exists(out_path):
        print("cached:", out_path)
        print("MYOPIC_DONE")
        return

    from piu.retriever import BM25Retriever
    from vllm import LLM
    from transformers import AutoTokenizer
    retriever = BM25Retriever(args.index)
    with open(os.path.join(args.gate_dir, "questions.json")) as f:
        qmeta = json.load(f)
    base_pre = {r["question_id"]: r for r in read_jsonl(qmeta["pre_path"])}
    plan = [p for p in read_jsonl(os.path.join(args.gate_dir, "arm_plan.jsonl"))
            if p["arm_type"] in ("base", "I2_add")]

    eps, meta = [], []
    for p in plan:
        rec = base_pre[p["question_id"]]
        events = make_arm_events(rec, p["state_idx"],
                                 injected_titles=[p["doc"]] if p["doc"] else ())
        # forbid retrieval: the next observation is the forced-answer notice
        events.append({"type": "info", "query": None, "titles": [],
                       "forced": True})
        ep = Episode(p["question_id"], rec["question"], prefix_events=events,
                     meta={"arm_id": p["arm_id"]})
        ep.forced = True
        eps.append(ep)
        meta.append(p)
    print(f"[setup] {len(eps)} myopic arms ({sum(1 for p in plan if p['arm_type']=='base')} base, "
          f"{sum(1 for p in plan if p['arm_type']=='I2_add')} interventions)", flush=True)

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=16384, seed=42)
    tok = AutoTokenizer.from_pretrained(args.model)
    runner = AgentRunner(llm, tok, retriever)
    t0 = time.time()
    runner.run(eps)
    rows = []
    for ep, p in zip(eps, meta):
        rec = ep.to_record()
        for k in ("arm_id", "arm_type", "ctype", "state_idx", "doc"):
            rec[k] = p.get(k)
        rec["outcome"] = arm_outcome(rec, base_pre[p["question_id"]]["answer_gold"])
        rows.append(rec)
    write_jsonl(out_path, rows)
    print(f"[arms] {len(rows)} in {time.time()-t0:.0f}s -> {out_path}", flush=True)
    print("MYOPIC_DONE")


if __name__ == "__main__":
    main()
