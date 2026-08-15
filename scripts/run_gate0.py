"""Gate 0: pipeline validation + cost measurement on N HotpotQA questions.

Phases (cached to results/<run>/ so the run is restartable):
  A. base trajectories (greedy agent rollouts)
  B. interventions — per state: base replay, I2 (+d) x5 candidate types,
     I3 (+matched distractor) per candidate, I1 (omission) x1,
     reliability duplicates on a subset of +d arms
  C. utilities (training vs diagnostic axes) + cost report

No go/no-go judgement at Gate 0 (per master doc) — outputs are pipeline
health metrics and a cost table extrapolated to Gate 1/2/3 sizes.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from piu.data_utils import load_hotpot_dev, write_jsonl, read_jsonl
from piu.agent import AgentRunner, Episode
from piu.intervene import (
    state_points, seen_titles_upto, build_pool, pick_candidates,
    match_distractor, make_arm_events,
)
from piu.utility import compute_utilities, arm_outcome

PROMPT_VERSION = "v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--index", default="data/bm25_index")
    ap.add_argument("--n-questions", type=int, default=30)
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--docs-per-search", type=int, default=3)
    ap.add_argument("--reliability-frac", type=float, default=0.10)
    ap.add_argument("--reliability-replays", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/gate0")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    config = vars(args).copy()
    config["prompt_version"] = PROMPT_VERSION
    config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    config["config_hash"] = config_hash
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    from piu.retriever import BM25Retriever
    retriever = BM25Retriever(args.index)
    print(f"[gate0] index loaded: {len(retriever.titles)} docs", flush=True)

    questions = load_hotpot_dev(n=args.n_questions, seed=args.seed)
    qmap = {q["question_id"]: q for q in questions}

    from vllm import LLM
    from transformers import AutoTokenizer
    t0 = time.time()
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=16384, enforce_eager=False, seed=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    runner = AgentRunner(llm, tokenizer, retriever, max_turns=args.max_turns,
                         docs_per_search=args.docs_per_search)
    load_time = time.time() - t0
    timings = {"model_load_sec": load_time}

    # ---------------- Phase A: base trajectories ----------------
    base_path = os.path.join(args.out, "base_trajectories.jsonl")
    if os.path.exists(base_path):
        base_records = {r["question_id"]: r for r in read_jsonl(base_path)}
        print(f"[gate0] phase A cached: {len(base_records)}", flush=True)
    else:
        t0 = time.time()
        eps = [Episode(q["question_id"], q["question"], meta={"phase": "A"})
               for q in questions]
        runner.run(eps)
        timings["phase_A_sec"] = time.time() - t0
        base_records = {}
        rows = []
        for ep in eps:
            rec = ep.to_record()
            q = qmap[ep.question_id]
            rec["outcome"] = arm_outcome(rec, q["answer"])
            base_records[ep.question_id] = rec
            rows.append(rec)
        write_jsonl(base_path, rows)
        print(f"[gate0] phase A done in {timings['phase_A_sec']:.0f}s", flush=True)

    # ---------------- Phase B: build arms ----------------
    plan_path = os.path.join(args.out, "arm_plan.jsonl")
    if os.path.exists(plan_path):
        plan = read_jsonl(plan_path)
        print(f"[gate0] arm plan cached: {len(plan)}", flush=True)
    else:
        t0 = time.time()
        plan = []  # one row per arm: prefix events + tags
        for qid, rec in base_records.items():
            q = qmap[qid]
            for sidx in state_points(rec):
                seen = seen_titles_upto(rec, sidx)
                seen_texts = [retriever.doc_by_title(t)["text"] for t in seen
                              if retriever.doc_by_title(t)]
                info = rec["events"][sidx]
                qt = info.get("query") or q["question"]
                pool, _ = build_pool(retriever, q["question"], qt, seen)
                cands = pick_candidates(pool, seen_texts, rng, q["gold_titles"])
                if not cands:
                    continue
                state_id = f"{qid}::s{sidx}"
                plan.append({"arm_id": f"{state_id}::base", "question_id": qid,
                             "state_idx": sidx, "arm_type": "base",
                             "ctype": None, "doc": None,
                             "events": make_arm_events(rec, sidx)})
                # I1 omission of the top served doc (sanity / Bridge Evidence repro)
                if info.get("titles"):
                    plan.append({"arm_id": f"{state_id}::omit", "question_id": qid,
                                 "state_idx": sidx, "arm_type": "I1_omit",
                                 "ctype": None, "doc": info["titles"][0],
                                 "events": make_arm_events(
                                     rec, sidx, removed_titles=[info["titles"][0]])})
                used = set(cands.values())
                for ctype, title in cands.items():
                    plan.append({"arm_id": f"{state_id}::d::{ctype}",
                                 "question_id": qid, "state_idx": sidx,
                                 "arm_type": "I2_add", "ctype": ctype, "doc": title,
                                 "events": make_arm_events(rec, sidx,
                                                          injected_titles=[title])})
                    dtil = match_distractor(retriever, title, qt, pool,
                                            q["gold_titles"], used, rng)
                    if dtil:
                        used.add(dtil)
                        plan.append({"arm_id": f"{state_id}::ctrl::{ctype}",
                                     "question_id": qid, "state_idx": sidx,
                                     "arm_type": "I3_ctrl", "ctype": ctype,
                                     "doc": dtil, "matched_to": title,
                                     "events": make_arm_events(
                                         rec, sidx, injected_titles=[dtil])})
        # reliability duplicates on a subset of I2 arms
        i2 = [p for p in plan if p["arm_type"] == "I2_add"]
        rel = rng.sample(i2, max(1, int(len(i2) * args.reliability_frac)))
        for p in rel:
            for r in range(args.reliability_replays):
                dup = {k: v for k, v in p.items()}
                dup["arm_id"] = p["arm_id"] + f"::rep{r + 1}"
                dup["arm_type"] = "I2_add_rep"
                plan.append(dup)
        write_jsonl(plan_path, plan)
        timings["phase_B_plan_sec"] = time.time() - t0
        print(f"[gate0] arm plan: {len(plan)} arms", flush=True)

    arms_path = os.path.join(args.out, "arm_results.jsonl")
    if os.path.exists(arms_path):
        arm_results = {r["arm_id"]: r for r in read_jsonl(arms_path)}
        print(f"[gate0] phase B cached: {len(arm_results)}", flush=True)
    else:
        t0 = time.time()
        eps = []
        for p in plan:
            ep = Episode(p["question_id"], qmap[p["question_id"]]["question"],
                         prefix_events=p["events"],
                         meta={"arm_id": p["arm_id"], "arm_type": p["arm_type"],
                               "ctype": p["ctype"], "state_idx": p["state_idx"],
                               "doc": p["doc"]})
            eps.append(ep)
        runner.run(eps)
        timings["phase_B_sec"] = time.time() - t0
        arm_results = {}
        rows = []
        for ep in eps:
            rec = ep.to_record()
            rec["arm_id"] = ep.meta["arm_id"]
            rec["arm_type"] = ep.meta["arm_type"]
            rec["ctype"] = ep.meta["ctype"]
            rec["state_idx"] = ep.meta["state_idx"]
            rec["doc"] = ep.meta["doc"]
            arm_results[rec["arm_id"]] = rec
            rows.append(rec)
        write_jsonl(arms_path, rows)
        print(f"[gate0] phase B done in {timings['phase_B_sec']:.0f}s "
              f"({len(eps)} arms)", flush=True)

    # ---------------- Phase C: utilities + cost report ----------------
    util_rows = []
    for p in plan:
        if p["arm_type"] != "I2_add":
            continue
        qid = p["question_id"]
        q = qmap[qid]
        sidx = p["state_idx"]
        state_id = f"{qid}::s{sidx}"
        base = arm_results.get(f"{state_id}::base")
        arm_d = arm_results.get(p["arm_id"])
        ctrl = arm_results.get(f"{state_id}::ctrl::{p['ctype']}")
        if not base or not arm_d:
            continue
        u = compute_utilities(retriever, base, arm_d, ctrl, sidx,
                              q["answer"], q["gold_titles"])
        u.update({"arm_id": p["arm_id"], "question_id": qid, "state_idx": sidx,
                  "ctype": p["ctype"], "doc": p["doc"],
                  "doc_is_gold": p["doc"] in q["gold_titles"],
                  "model_id": args.model, "prompt_version": PROMPT_VERSION,
                  "config_hash": config_hash})
        util_rows.append(u)
    write_jsonl(os.path.join(args.out, "utilities.jsonl"), util_rows)

    # omission utilities (I1 sanity)
    omit_rows = []
    for p in plan:
        if p["arm_type"] != "I1_omit":
            continue
        qid = p["question_id"]
        state_id = f"{qid}::s{p['state_idx']}"
        base = arm_results.get(f"{state_id}::base")
        omit = arm_results.get(p["arm_id"])
        if not base or not omit:
            continue
        yb = arm_outcome(base, qmap[qid]["answer"])
        yo = arm_outcome(omit, qmap[qid]["answer"])
        omit_rows.append({"arm_id": p["arm_id"], "question_id": qid,
                          "state_idx": p["state_idx"], "doc": p["doc"],
                          "doc_is_gold": p["doc"] in qmap[qid]["gold_titles"],
                          "ctu_f1": yb["f1"] - yo["f1"],
                          "ctu_em": yb["em"] - yo["em"]})
    write_jsonl(os.path.join(args.out, "omission_utilities.jsonl"), omit_rows)

    # replay consistency: base replay vs original trajectory
    consistency = []
    for qid, rec in base_records.items():
        for sidx in state_points(rec):
            b = arm_results.get(f"{qid}::s{sidx}::base")
            if b:
                consistency.append(float((b["answer"] or "") == (rec["answer"] or "")))
    # reliability: duplicated I2 arms agree with the original I2 arm?
    rel_agree = []
    for p in plan:
        if p["arm_type"] != "I2_add_rep":
            continue
        orig_id = p["arm_id"].rsplit("::rep", 1)[0]
        a, b = arm_results.get(orig_id), arm_results.get(p["arm_id"])
        if a and b:
            rel_agree.append(float((a["answer"] or "") == (b["answer"] or "")))

    n_states = sum(len(state_points(r)) for r in base_records.values())
    total_tokens = (sum(r["gen_tokens"] for r in base_records.values())
                    + sum(r["gen_tokens"] for r in arm_results.values()))
    n_arms = len(arm_results)
    dEM0 = [u for u in util_rows if u["U_task_raw_em"] == 0.0]
    cost = {
        "n_questions": len(base_records),
        "n_states": n_states,
        "states_per_question": n_states / max(len(base_records), 1),
        "n_arms": n_arms,
        "arms_per_question": n_arms / max(len(base_records), 1),
        "total_gen_tokens": total_tokens,
        "tokens_per_rollout": total_tokens / max(n_arms + len(base_records), 1),
        "timings_sec": timings,
        "base_em": sum(r["outcome"]["em"] for r in base_records.values()) / max(len(base_records), 1),
        "base_f1": sum(r["outcome"]["f1"] for r in base_records.values()) / max(len(base_records), 1),
        "replay_consistency": (sum(consistency) / len(consistency)) if consistency else None,
        "reliability_answer_agreement": (sum(rel_agree) / len(rel_agree)) if rel_agree else None,
        "frac_dEM_zero": len(dEM0) / max(len(util_rows), 1),
        "n_utility_rows": len(util_rows),
    }
    with open(os.path.join(args.out, "cost_report.json"), "w") as f:
        json.dump(cost, f, indent=2)
    print(json.dumps(cost, indent=2), flush=True)
    print("[gate0] DONE", flush=True)


if __name__ == "__main__":
    main()
