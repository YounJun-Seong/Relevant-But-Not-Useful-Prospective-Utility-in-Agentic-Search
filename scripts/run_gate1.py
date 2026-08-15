"""Gate 1 pipeline, phase-per-process to avoid GPU memory conflicts:

  --phase base    vLLM: base trajectories on n_pre questions (default 300)
  --phase select  CPU:  stratified 50 representative + 50 base-failure enriched
  --phase plan    GPU (E5+CE, no vLLM): candidate pools, 8 types, dual controls
  --phase arms    vLLM: run all replay arms
  --phase utils   CPU:  utilities (training + diagnostic axes)

Main condition keeps Gate 0 search settings (docs=3, max_turns=4) per R8-6.
Stress test (diagnostic only) runs with --stress: docs=2, max_turns=6,
hard/base-failure questions, separate out dir.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from piu.data_utils import load_hotpot_dev, read_jsonl, write_jsonl
from piu.agent import AgentRunner, Episode
from piu.intervene import state_points, seen_titles_upto, make_arm_events
from piu.utility import compute_utilities, arm_outcome

PROMPT_VERSION = "v1"


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["base", "select", "plan", "arms", "utils"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--index", default="data/bm25_index")
    ap.add_argument("--emb", default="data/e5_embeddings.fp16.npy")
    ap.add_argument("--out", default="results/gate1")
    ap.add_argument("--dataset", default="hotpot", choices=["hotpot", "2wiki", "musique"])
    ap.add_argument("--split", default="validation",
                    help="HotpotQA split for the question pool (train for "
                         "Gate 3 label generation; strict question-level "
                         "separation from dev-based evaluation)")
    ap.add_argument("--skip-diag", action="store_true",
                    help="utils phase: skip U_search nDCG scoring (labels only)")
    ap.add_argument("--n-pre", type=int, default=300)
    ap.add_argument("--pre-offset", type=int, default=0,
                    help="skip first K rows of the seeded shuffle (lockbox)")
    ap.add_argument("--pre-tag", default="",
                    help="extra tag in the pre-pool filename (e.g. agent name)")
    ap.add_argument("--n-rep", type=int, default=50)
    ap.add_argument("--n-fail", type=int, default=50)
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--docs-per-search", type=int, default=3)
    ap.add_argument("--ctrl2-frac", type=float, default=0.30)
    ap.add_argument("--reliability-frac", type=float, default=0.20)
    ap.add_argument("--reliability-replays", type=int, default=2)
    ap.add_argument("--reliability-by-state", action="store_true",
                    help="R10: sample whole states (all candidates replayed) "
                         "so 3x3 pair comparisons exist")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu-mem", type=float, default=0.25)
    ap.add_argument("--arm-types", default="",
                    help="comma list: run only these arm types (crossfit)")
    ap.add_argument("--arms-suffix", default="",
                    help="suffix for arm_results file (crossfit set A/B)")
    ap.add_argument("--stress", action="store_true",
                    help="diagnostic stress test: docs=2, max_turns=6, hard qs")
    return ap.parse_args()


def apply_stress(args):
    if args.stress:
        args.docs_per_search = 2
        args.max_turns = 6
        args.out = "results/gate1_stress"
        args.n_rep = 0
        args.n_fail = 30
    return args


def make_runner(args, retriever):
    from vllm import LLM
    from transformers import AutoTokenizer

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=16384, seed=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    return AgentRunner(llm, tokenizer, retriever, max_turns=args.max_turns,
                       docs_per_search=args.docs_per_search)


def load_retriever(args):
    from piu.retriever import BM25Retriever
    return BM25Retriever(args.index)


def phase_base(args):
    """Base trajectories over the pre-pool (shared by main and stress)."""
    ds_tag = "" if args.dataset == "hotpot" else f"{args.dataset}_"
    if args.pre_tag:
        ds_tag = f"{args.pre_tag}_{ds_tag}"
    pre_path = os.path.join("results", f"base_{ds_tag}{args.split}_pre{args.n_pre}"
                            f"_t{args.max_turns}_d{args.docs_per_search}.jsonl")
    if os.path.exists(pre_path):
        print(f"cached: {pre_path}")
        print("BASE_DONE (cached)")
        return
    retriever = load_retriever(args)
    if args.dataset == "2wiki":
        from piu.data_utils import load_2wiki
        questions = load_2wiki(n=args.n_pre, seed=args.seed, split=args.split,
                               offset=args.pre_offset,
                               corpus_titles=set(retriever.titles))
    elif args.dataset == "musique":
        from piu.data_utils import load_musique
        questions = load_musique(n=args.n_pre, seed=args.seed, split=args.split,
                                 offset=args.pre_offset,
                                 corpus_titles=set(retriever.titles))
    else:
        questions = load_hotpot_dev(n=args.n_pre, seed=args.seed,
                                    split=args.split, offset=args.pre_offset)
    runner = make_runner(args, retriever)
    t0 = time.time()
    eps = [Episode(q["question_id"], q["question"]) for q in questions]
    runner.run(eps)
    rows = []
    for ep, q in zip(eps, questions):
        rec = ep.to_record()
        rec["outcome"] = arm_outcome(rec, q["answer"])
        rec["gold_titles"] = q["gold_titles"]
        rec["answer_gold"] = q["answer"]
        rec["level"] = q["level"]
        rows.append(rec)
    write_jsonl(pre_path, rows)
    em = sum(r["outcome"]["em"] for r in rows) / len(rows)
    print(f"BASE_DONE {len(rows)} qs, EM {em:.3f}, {time.time()-t0:.0f}s -> {pre_path}")


def phase_select(args):
    os.makedirs(args.out, exist_ok=True)
    ds_tag = "" if args.dataset == "hotpot" else f"{args.dataset}_"
    if args.pre_tag:
        ds_tag = f"{args.pre_tag}_{ds_tag}"
    pre_path = os.path.join("results", f"base_{ds_tag}{args.split}_pre{args.n_pre}"
                            f"_t{args.max_turns}_d{args.docs_per_search}.jsonl")
    rows = read_jsonl(pre_path)
    rng = random.Random(args.seed)
    fails = [r for r in rows if r["outcome"]["em"] == 0.0]
    fail_sel = rng.sample(fails, min(args.n_fail, len(fails)))
    fail_ids = {r["question_id"] for r in fail_sel}
    rest = [r for r in rows if r["question_id"] not in fail_ids]
    rep_sel = rng.sample(rest, min(args.n_rep, len(rest)))
    sel = [{"question_id": r["question_id"], "stratum": "fail_enriched"}
           for r in fail_sel]
    sel += [{"question_id": r["question_id"], "stratum": "representative"}
            for r in rep_sel]
    with open(os.path.join(args.out, "questions.json"), "w") as f:
        json.dump({"selected": sel, "pre_path": pre_path,
                   "n_fail_pool": len(fails)}, f, indent=2)
    print(f"SELECT_DONE rep={len(rep_sel)} fail={len(fail_sel)} "
          f"(fail pool {len(fails)}/{len(rows)})")


def _load_selected(args):
    with open(os.path.join(args.out, "questions.json")) as f:
        meta = json.load(f)
    base_rows = {r["question_id"]: r for r in read_jsonl(meta["pre_path"])}
    stratum = {s["question_id"]: s["stratum"] for s in meta["selected"]}
    return {qid: base_rows[qid] for qid in stratum}, stratum


def phase_plan(args):
    if os.path.exists(os.path.join(args.out, "arm_plan.jsonl")):
        print("PLAN_DONE (cached)")
        return
    from piu.dense import E5DenseRetriever, CEScorer
    from piu.intervene2 import build_pool_v2, pick_candidates_v2, match_distractors

    retriever = load_retriever(args)
    dense = E5DenseRetriever(args.emb, retriever.titles)
    ce = CEScorer()
    base_records, stratum = _load_selected(args)
    rng = random.Random(args.seed)
    plan, ce_rankings = [], []
    t0 = time.time()
    for qi, (qid, rec) in enumerate(base_records.items()):
        for sidx in state_points(rec):
            seen = seen_titles_upto(rec, sidx)
            seen_texts = [retriever.doc_by_title(t)["text"] for t in seen
                          if retriever.doc_by_title(t)]
            info = rec["events"][sidx]
            qt = info.get("query") or rec["question"]
            pool = build_pool_v2(retriever, dense, rec["question"], qt, seen)
            cands, ce_rank = pick_candidates_v2(pool, ce, qt, seen_texts, rng)
            if not cands:
                continue
            wide = retriever.search(qt, k=100)  # shared by all distractor picks
            state_id = f"{qid}::s{sidx}"
            ce_rankings.append({"state_id": state_id, "ce_ranking": ce_rank[:20],
                                "candidates": cands})
            plan.append({"arm_id": f"{state_id}::base", "question_id": qid,
                         "state_idx": sidx, "arm_type": "base", "ctype": None,
                         "doc": None, "events": make_arm_events(rec, sidx)})
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
                n_ctrl = 2 if rng.random() < args.ctrl2_frac else 1
                dts = match_distractors(retriever, title, qt, pool,
                                        rec["gold_titles"], used, rng,
                                        n=n_ctrl, wide=wide)
                for ci, dt in enumerate(dts):
                    used.add(dt)
                    suffix = "" if ci == 0 else f"{ci+1}"
                    plan.append({"arm_id": f"{state_id}::ctrl{suffix}::{ctype}",
                                 "question_id": qid, "state_idx": sidx,
                                 "arm_type": "I3_ctrl" if ci == 0 else "I3_ctrl2",
                                 "ctype": ctype, "doc": dt, "matched_to": title,
                                 "events": make_arm_events(
                                     rec, sidx, injected_titles=[dt])})
        if qi % 20 == 0:
            print(f"plan {qi}/{len(base_records)} ({time.time()-t0:.0f}s)", flush=True)
    i2 = [p for p in plan if p["arm_type"] == "I2_add"]
    if args.reliability_by_state:
        state_ids = sorted({(p["question_id"], p["state_idx"]) for p in i2})
        rel_states = set(rng.sample(
            state_ids, max(1, int(len(state_ids) * args.reliability_frac))))
        rel = [p for p in i2
               if (p["question_id"], p["state_idx"]) in rel_states]
    else:
        rel = rng.sample(i2, max(1, int(len(i2) * args.reliability_frac)))
    for p in rel:
        for r in range(args.reliability_replays):
            dup = dict(p)
            dup["arm_id"] = p["arm_id"] + f"::rep{r+1}"
            dup["arm_type"] = "I2_add_rep"
            plan.append(dup)
    write_jsonl(os.path.join(args.out, "arm_plan.jsonl"), plan)
    write_jsonl(os.path.join(args.out, "ce_rankings.jsonl"), ce_rankings)
    print(f"PLAN_DONE {len(plan)} arms, {len(ce_rankings)} states, "
          f"{time.time()-t0:.0f}s")


def phase_arms(args):
    retriever = load_retriever(args)
    base_records, _ = _load_selected(args)
    plan = read_jsonl(os.path.join(args.out, "arm_plan.jsonl"))
    if args.arm_types:
        allowed = set(args.arm_types.split(","))
        plan = [p for p in plan if p["arm_type"] in allowed]
        print(f"[arms] filtered to {sorted(allowed)}: {len(plan)} arms")
    runner = make_runner(args, retriever)
    t0 = time.time()
    eps = []
    for p in plan:
        eps.append(Episode(p["question_id"],
                           base_records[p["question_id"]]["question"],
                           prefix_events=p["events"],
                           meta={"arm_id": p["arm_id"]}))
    runner.run(eps)
    rows = []
    for ep, p in zip(eps, plan):
        rec = ep.to_record()
        for k in ("arm_id", "arm_type", "ctype", "state_idx", "doc"):
            rec[k] = p.get(k)
        rec["matched_to"] = p.get("matched_to")
        rows.append(rec)
    write_jsonl(os.path.join(args.out,
                             f"arm_results{args.arms_suffix}.jsonl"), rows)
    print(f"ARMS_DONE {len(rows)} arms in {time.time()-t0:.0f}s")
    with open(os.path.join(args.out, "timings.json"), "w") as f:
        json.dump({"phase_arms_sec": time.time() - t0, "n_arms": len(rows)}, f)


def phase_utils(args):
    retriever = load_retriever(args)
    base_records, stratum = _load_selected(args)
    plan = read_jsonl(os.path.join(args.out, "arm_plan.jsonl"))
    arm_results = {r["arm_id"]: r
                   for r in read_jsonl(os.path.join(args.out, "arm_results.jsonl"))}
    config = vars(args).copy()
    config["prompt_version"] = PROMPT_VERSION
    config_hash = hashlib.md5(
        json.dumps({k: v for k, v in config.items() if k != "phase"},
                   sort_keys=True).encode()).hexdigest()[:10]
    util_rows = []
    for p in plan:
        if p["arm_type"] != "I2_add":
            continue
        qid, sidx = p["question_id"], p["state_idx"]
        rec = base_records[qid]
        state_id = f"{qid}::s{sidx}"
        base = arm_results.get(f"{state_id}::base")
        arm_d = arm_results.get(p["arm_id"])
        ctrl = arm_results.get(f"{state_id}::ctrl::{p['ctype']}")
        ctrl2 = arm_results.get(f"{state_id}::ctrl2::{p['ctype']}")
        if not base or not arm_d:
            continue
        u = compute_utilities(retriever, base, arm_d, ctrl, sidx,
                              rec["answer_gold"], rec["gold_titles"],
                              compute_ndcg=not args.skip_diag)
        if ctrl2:
            yd = u["arm_d"]
            yc2 = arm_outcome(ctrl2, rec["answer_gold"])
            u["U_task_ctrl2_f1"] = yd["f1"] - yc2["f1"]
            u["U_task_ctrl2_em"] = yd["em"] - yc2["em"]
        u.update({"arm_id": p["arm_id"], "question_id": qid, "state_idx": sidx,
                  "ctype": p["ctype"], "doc": p["doc"],
                  "doc_is_gold": p["doc"] in rec["gold_titles"],
                  "stratum": stratum[qid], "model_id": args.model,
                  "prompt_version": PROMPT_VERSION, "config_hash": config_hash})
        util_rows.append(u)
    write_jsonl(os.path.join(args.out, "utilities.jsonl"), util_rows)

    omit_rows = []
    for p in plan:
        if p["arm_type"] != "I1_omit":
            continue
        qid = p["question_id"]
        rec = base_records[qid]
        state_id = f"{qid}::s{p['state_idx']}"
        base, omit = arm_results.get(f"{state_id}::base"), arm_results.get(p["arm_id"])
        if not base or not omit:
            continue
        yb, yo = arm_outcome(base, rec["answer_gold"]), arm_outcome(omit, rec["answer_gold"])
        omit_rows.append({"arm_id": p["arm_id"], "question_id": qid,
                          "state_idx": p["state_idx"], "doc": p["doc"],
                          "doc_is_gold": p["doc"] in rec["gold_titles"],
                          "stratum": stratum[qid],
                          "ctu_f1": yb["f1"] - yo["f1"],
                          "ctu_em": yb["em"] - yo["em"]})
    write_jsonl(os.path.join(args.out, "omission_utilities.jsonl"), omit_rows)
    print(f"UTILS_DONE {len(util_rows)} interventions, {len(omit_rows)} omissions")


if __name__ == "__main__":
    args = apply_stress(get_args())
    os.makedirs(args.out, exist_ok=True)
    {"base": phase_base, "select": phase_select, "plan": phase_plan,
     "arms": phase_arms, "utils": phase_utils}[args.phase](args)
