"""Build pairwise preference labels (R9) from a gate results dir and report
label statistics + Pairwise Utility Consistency."""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.data_utils import read_jsonl, write_jsonl, f1_score
from piu.pairwise import build_pairs, replay_consistency

OUT = sys.argv[1] if len(sys.argv) > 1 else "results/gate1"
TAU = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1

utils = read_jsonl(os.path.join(OUT, "utilities.jsonl"))
plan = read_jsonl(os.path.join(OUT, "arm_plan.jsonl"))
arm_results = {r["arm_id"]: r for r in read_jsonl(os.path.join(OUT, "arm_results.jsonl"))}
with open(os.path.join(OUT, "questions.json")) as f:
    qmeta = json.load(f)
base_pre = {r["question_id"]: r for r in read_jsonl(qmeta["pre_path"])}

# annotate replay arms with F1 vs gold answer
for p in plan:
    if p["arm_type"] in ("I2_add", "I2_add_rep"):
        r = arm_results.get(p["arm_id"])
        if r is not None and "_f1" not in r:
            gold = base_pre[p["question_id"]]["answer_gold"]
            r["_f1"] = f1_score(r["answer"] or "", gold)

pairs = build_pairs(utils, tau=TAU)
decided = [p for p in pairs if not p["tie"]]
write_jsonl(os.path.join(OUT, f"pairs_tau{TAU}.jsonl"), pairs)

n_checked, n_consistent = replay_consistency(pairs, plan, arm_results, tau=TAU)

print(f"pairs total: {len(pairs)} | decided: {len(decided)} "
      f"({len(decided)/max(len(pairs),1):.1%}) | tie: {len(pairs)-len(decided)}")
print(f"decided margin 분포: "
      f"{Counter(round(abs(p['margin']),1) for p in decided).most_common(6)}")
strata = Counter(p["stratum"] for p in decided)
print(f"decided by stratum: {dict(strata)}")
win = Counter()
for p in decided:
    win[p["ctype_i"] if p["pref"] == "i" else p["ctype_j"]] += 1
print(f"승자 ctype 분포: {dict(win.most_common())}")
print(f"Pairwise Utility Consistency (양쪽 3x replay 존재 pair): "
      f"{n_consistent}/{n_checked}"
      + (f" = {n_consistent/n_checked:.1%}" if n_checked else " (해당 없음)"))
