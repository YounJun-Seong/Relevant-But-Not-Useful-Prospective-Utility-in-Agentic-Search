"""Control-matching ablation (post-hoc exploratory, R21 review response).

The confirmatory matched control draws a distractor from BM25 top-100 under q_t
subject to: same rank bucket (rank//20), length within an escalating tolerance
(10/25/50/100%), non-gold, title-Jaccard <= 0.3, and not already used.
This script re-draws controls under relaxed specifications and re-runs ONLY the
control arms under the frozen greedy policy, so the controlled diagnostic
U_ctrl = F1(+d) - F1(+d~) can be recomputed against the identical +d arms.

Specs:
  no_rank_bucket   drop the rank-bucket constraint
  no_length        drop the length-tolerance constraint
  neither          drop both (non-gold + title-overlap only)

Frozen code is not modified; the matching variant lives here.
Fallback pool is the BM25 top-100 list rather than the original union pool
(which would require the dense index); fallbacks are rare and logged.
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.agent import AgentRunner, Episode
from piu.data_utils import read_jsonl, write_jsonl
from piu.intervene import _tokens, jaccard, make_arm_events
from piu.utility import arm_outcome

SPECS = ("no_rank_bucket", "no_length", "neither")


def draw_variant(spec, wide, cand_title, cand_len, cand_title_tok, gold,
                 excluded, rng):
    """Matched-distractor draw with one or both constraints removed."""
    cand_rank = next((i for i, d in enumerate(wide)
                      if d["title"] == cand_title), None)
    cband = None if cand_rank is None else cand_rank // 20
    use_rank = spec == "no_length"          # rank kept only in no_length
    use_len = spec == "no_rank_bucket"      # length kept only in no_rank_bucket
    tols = (0.10, 0.25, 0.50, 1.00) if use_len else (None,)
    for tol in tols:
        options = []
        for i, d in enumerate(wide):
            if d["title"] == cand_title or d["title"] in excluded:
                continue
            if d["title"].lower() in gold:
                continue
            if jaccard(_tokens(d["title"]), cand_title_tok) > 0.3:
                continue
            if use_len:
                dlen = max(len(d["text"].split()), 1)
                if abs(dlen - cand_len) / cand_len > tol:
                    continue
            if use_rank and cband is not None and (i // 20) != cband:
                continue
            options.append((d["title"], i, max(len(d["text"].split()), 1)))
        if options:
            return rng.choice(options), False
    rest = [(d["title"], i, max(len(d["text"].split()), 1))
            for i, d in enumerate(wide)
            if d["title"] not in excluded and d["title"] != cand_title
            and d["title"].lower() not in gold]
    return (rng.choice(rest) if rest else None), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-dir", default="results/lockbox")
    ap.add_argument("--out", default="results/matching_ablation")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--index", default="data/bm25_index")
    ap.add_argument("--gpu-mem", type=float, default=0.40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-states", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    from piu.retriever import BM25Retriever
    from vllm import LLM
    from transformers import AutoTokenizer

    retriever = BM25Retriever(args.index)
    with open(os.path.join(args.gate_dir, "questions.json")) as f:
        qmeta = json.load(f)
    base_pre = {r["question_id"]: r for r in read_jsonl(qmeta["pre_path"])}
    plan = [p for p in read_jsonl(os.path.join(args.gate_dir, "arm_plan.jsonl"))
            if p["arm_type"] == "I2_add"]

    by_state = {}
    for p in plan:
        by_state.setdefault((p["question_id"], p["state_idx"]), []).append(p)
    states = sorted(by_state)
    if args.limit_states:
        states = states[: args.limit_states]
    print(f"[setup] {len(states)} states, "
          f"{sum(len(by_state[s]) for s in states)} candidates", flush=True)

    plan_path = os.path.join(args.out, "ablation_plan.jsonl")
    if os.path.exists(plan_path):
        ab_plan = read_jsonl(plan_path)
        print(f"[plan] cached: {len(ab_plan)} arms", flush=True)
    else:
        ab_plan, t0 = [], time.time()
        for si, st in enumerate(states):
            qid, sidx = st
            rec = base_pre[qid]
            qt = rec["events"][sidx].get("query") or rec["question"]
            wide = retriever.search(qt, k=100)
            gold = {g.lower() for g in rec["gold_titles"]}
            cands = by_state[st]
            cand_titles = {p["doc"] for p in cands}
            for spec in SPECS:
                rng = random.Random(f"{args.seed}:{spec}:{qid}:{sidx}")
                excluded = set(cand_titles)
                for p in cands:
                    doc = retriever.doc_by_title(p["doc"])
                    if doc is None:
                        continue
                    clen = max(len(doc["text"].split()), 1)
                    crank = next((i for i, d in enumerate(wide)
                                  if d["title"] == p["doc"]), None)
                    got, fb = draw_variant(
                        spec, wide, p["doc"], clen, _tokens(p["doc"]), gold,
                        excluded, rng)
                    if got is None:
                        continue
                    dt, drank, dlen = got
                    excluded.add(dt)
                    ab_plan.append({
                        "arm_id": f"{qid}::s{sidx}::{spec}::{p['ctype']}",
                        "question_id": qid, "state_idx": sidx, "spec": spec,
                        "ctype": p["ctype"], "matched_to": p["doc"], "doc": dt,
                        "fallback": fb,
                        "cand_rank": crank, "ctrl_rank": drank,
                        "rank_bucket_equal": (crank is not None
                                              and crank // 20 == drank // 20),
                        "len_ratio": dlen / clen,
                        "events": make_arm_events(rec, sidx,
                                                  injected_titles=[dt]),
                    })
            if si % 100 == 0:
                print(f"  plan {si}/{len(states)} ({time.time()-t0:.0f}s)",
                      flush=True)
        write_jsonl(plan_path, ab_plan)
        print(f"[plan] {len(ab_plan)} arms in {time.time()-t0:.0f}s", flush=True)

    out_path = os.path.join(args.out, "arm_results.jsonl")
    if os.path.exists(out_path):
        print(f"[arms] cached -> {out_path}", flush=True)
    else:
        llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
                  max_model_len=16384, seed=42)
        tok = AutoTokenizer.from_pretrained(args.model)
        runner = AgentRunner(llm, tok, retriever)  # frozen greedy policy
        t0 = time.time()
        eps = [Episode(p["question_id"], base_pre[p["question_id"]]["question"],
                       prefix_events=p["events"], meta={"arm_id": p["arm_id"]})
               for p in ab_plan]
        runner.run(eps)
        rows = []
        for ep, p in zip(eps, ab_plan):
            rec = ep.to_record()
            for k in ("arm_id", "spec", "ctype", "state_idx", "doc",
                      "matched_to", "fallback", "rank_bucket_equal",
                      "len_ratio"):
                rec[k] = p.get(k)
            rec["outcome"] = arm_outcome(
                rec, base_pre[p["question_id"]]["answer_gold"])
            rows.append(rec)
        write_jsonl(out_path, rows)
        print(f"[arms] {len(rows)} arms in {time.time()-t0:.0f}s", flush=True)

    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print("MATCHING_ABLATION_DONE", flush=True)


if __name__ == "__main__":
    main()
