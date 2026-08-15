"""Gate features (R12-⑤): per-state uncertainty observables.

Agent uncertainty — from the frozen state prefix, probe the agent with a
short greedy continuation and record:
  probe_action      would it search (0) or answer (1) right now?
  probe_conf        mean token logprob of the continuation
  probe_ans_conf    mean token logprob inside <answer>..</answer> if any

Retrieval uncertainty — CE scores of the state's 8 candidates:
  ce_top_margin     top1 − top2 (relevance saturation: big = clear winner)
  ce_std, ce_mean   spread of the candidate pool's relevance

All features are observable at serving time (no labels involved).
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.data_utils import read_jsonl, write_jsonl
from piu.agent import AgentRunner, Episode
from piu.intervene import make_arm_events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--index", default="data/bm25_index")
    ap.add_argument("--gpu-mem", type=float, default=0.5)
    args = ap.parse_args()

    from piu.retriever import BM25Retriever
    from piu.dense import CEScorer
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    retriever = BM25Retriever(args.index)
    with open(os.path.join(args.gate_dir, "questions.json")) as f:
        qmeta = json.load(f)
    base_pre = {r["question_id"]: r for r in read_jsonl(qmeta["pre_path"])}
    ce_rankings = read_jsonl(os.path.join(args.gate_dir, "ce_rankings.jsonl"))

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=16384, seed=42)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    runner = AgentRunner(llm, tokenizer, retriever)
    sp = SamplingParams(temperature=0.0, max_tokens=96, logprobs=0,
                       stop=["</search>", "</answer>"],
                       include_stop_str_in_output=True)

    # --- agent probe: one continuation per state ---
    prompts, keys = [], []
    for r in ce_rankings:
        qid, sidx = r["state_id"].split("::s")
        sidx = int(sidx)
        rec = base_pre[qid]
        ep = Episode(qid, rec["question"],
                     prefix_events=make_arm_events(rec, sidx))
        prompts.append(runner.base_prompt(rec["question"])
                       + ep.assistant_text(retriever))
        keys.append((qid, sidx))
    outs = llm.generate(prompts, sp, use_tqdm=True)

    ce = CEScorer()
    rows = []
    for (qid, sidx), out, r in zip(keys, outs, ce_rankings):
        gen = out.outputs[0]
        text = gen.text
        lps = [next(iter(t.values())).logprob for t in gen.logprobs] \
            if gen.logprobs else []
        probe_conf = sum(lps) / max(len(lps), 1)
        is_answer = "</answer>" in text
        # CE score distribution over this state's candidates
        rec = base_pre[qid]
        qt = None
        for e in rec["events"][: sidx + 1]:
            if e["type"] == "info" and e.get("query"):
                qt = e["query"]
        qt = qt or rec["question"]
        cand_titles = list(r["candidates"].values())
        passages = []
        for t in cand_titles:
            d = retriever.doc_by_title(t)
            passages.append((t + ". " + d["text"]) if d else t)
        scores = ce.score(qt, passages) if passages else []
        ss = sorted(scores, reverse=True)
        n = len(ss)
        mean = sum(ss) / max(n, 1)
        std = math.sqrt(sum((x - mean) ** 2 for x in ss) / max(n, 1))
        rows.append({
            "state_id": r["state_id"],
            "probe_action_answer": int(is_answer),
            "probe_conf": probe_conf,
            "probe_len": len(lps),
            "ce_top_margin": (ss[0] - ss[1]) if n >= 2 else 0.0,
            "ce_std": std, "ce_mean": mean,
        })
    write_jsonl(os.path.join(args.gate_dir, "gate_features.jsonl"), rows)
    print(f"GATE_FEATURES_DONE {len(rows)} states -> "
          f"{args.gate_dir}/gate_features.jsonl")


if __name__ == "__main__":
    main()
