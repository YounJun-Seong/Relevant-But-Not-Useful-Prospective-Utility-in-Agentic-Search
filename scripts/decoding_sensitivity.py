"""Decoding-stochasticity sensitivity (post-hoc exploratory, R21 review response).

Re-runs the lockbox base and I2_add arms under SAMPLED decoding (temperature
T>0) with R independent replicates, and recomputes the H1 endpoints
(opportunity rate, Oracle-CE headroom) plus greedy-vs-sampled label agreement.

Frozen code is NOT modified: the AgentRunner is constructed exactly as in the
confirmatory runs and only its SamplingParams object is replaced afterwards, so
the prompt, stop sequences, parsing, retrieval, and budget logic are identical.
Controls (I3_ctrl) are not re-run: confirmatory endpoints are defined on raw
intervention outcomes.

EXPLORATORY: the lockbox was opened once under R16; this analysis introduces no
new model and no new hypothesis test, and is reported as a robustness check.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.agent import AgentRunner, Episode
from piu.data_utils import read_jsonl, write_jsonl
from piu.utility import arm_outcome


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-dir", default="results/lockbox")
    ap.add_argument("--out", default="results/decoding_sensitivity")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--index", default="data/bm25_index")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=101)
    ap.add_argument("--gpu-mem", type=float, default=0.40)
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--docs-per-search", type=int, default=3)
    ap.add_argument("--limit-states", type=int, default=0,
                    help="pilot mode: only the first N states (0 = all)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    from piu.retriever import BM25Retriever
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    retriever = BM25Retriever(args.index)
    with open(os.path.join(args.gate_dir, "questions.json")) as f:
        qmeta = json.load(f)
    base_pre = {r["question_id"]: r for r in read_jsonl(qmeta["pre_path"])}
    plan = read_jsonl(os.path.join(args.gate_dir, "arm_plan.jsonl"))
    plan = [p for p in plan if p["arm_type"] in ("base", "I2_add")]
    if args.limit_states:
        states = []
        for p in plan:
            k = (p["question_id"], p["state_idx"])
            if k not in states:
                states.append(k)
        keep = set(states[: args.limit_states])
        plan = [p for p in plan if (p["question_id"], p["state_idx"]) in keep]
    n_states = len({(p["question_id"], p["state_idx"]) for p in plan})
    print(f"[setup] {len(plan)} arms over {n_states} states, "
          f"T={args.temperature}, R={args.replicates}", flush=True)

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=16384, seed=42)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    runner = AgentRunner(llm, tokenizer, retriever, max_turns=args.max_turns,
                         docs_per_search=args.docs_per_search)

    for rep in range(args.replicates):
        out_path = os.path.join(
            args.out, f"arm_results_T{args.temperature}_r{rep + 1}.jsonl")
        if os.path.exists(out_path):
            print(f"[rep {rep+1}] cached -> {out_path}", flush=True)
            continue
        seed = args.seed_base + rep
        # only the sampling policy changes; everything else is the frozen setup
        runner.sampling = SamplingParams(
            temperature=args.temperature, top_p=args.top_p, seed=seed,
            max_tokens=768, stop=["</search>", "</answer>", "<information>"],
            include_stop_str_in_output=True)
        t0 = time.time()
        eps = [Episode(p["question_id"], base_pre[p["question_id"]]["question"],
                       prefix_events=p["events"], meta={"arm_id": p["arm_id"]})
               for p in plan]
        runner.run(eps)
        rows = []
        for ep, p in zip(eps, plan):
            rec = ep.to_record()
            for k in ("arm_id", "arm_type", "ctype", "state_idx", "doc"):
                rec[k] = p.get(k)
            gold = base_pre[p["question_id"]]["answer_gold"]
            rec["outcome"] = arm_outcome(rec, gold)
            rows.append(rec)
        write_jsonl(out_path, rows)
        print(f"[rep {rep+1}] seed {seed}: {len(rows)} arms in "
              f"{time.time()-t0:.0f}s -> {out_path}", flush=True)

    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print("DECODING_ARMS_DONE", flush=True)


if __name__ == "__main__":
    main()
