"""R21-① cross-fitted Oracle evaluation — run once.

Two independent arm sets (A/B, separate engine instances). Selection on one
set, evaluation on the other:
  Gain_A→B = U_B(argmax_d U_A(d)) − U_B(d_CE);  symmetric;  Gain_CF = mean.
Question-level paired bootstrap CI. Naive (within-set) gains reported for
comparison — cross-fitted is the headline.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.data_utils import read_jsonl, f1_score

D = "results/crossfit"
EPS = 1e-9


def main():
    assert not os.path.exists(f"{D}/CROSSFIT_REPORT.json"), "already evaluated"
    plan = read_jsonl(f"{D}/arm_plan.jsonl")
    ce_rank = {r["state_id"]: {t: i for i, t in enumerate(r["ce_ranking"])}
               for r in read_jsonl(f"{D}/ce_rankings.jsonl")}
    with open(f"{D}/questions.json") as f:
        qmeta = json.load(f)
    gold = {r["question_id"]: r["answer_gold"]
            for r in read_jsonl(qmeta["pre_path"])}

    def load_set(suffix):
        res = {r["arm_id"]: r
               for r in read_jsonl(f"{D}/arm_results{suffix}.jsonl")}
        out = {}
        for arm_id, r in res.items():
            out[arm_id] = f1_score(r["answer"] or "", gold[r["question_id"]])
        return out

    fa, fb = load_set("_A"), load_set("_B")

    by_state = defaultdict(list)
    for p in plan:
        if p["arm_type"] == "I2_add":
            by_state[(p["question_id"], p["state_idx"])].append(p)

    def gains(sel, ev):
        """sel/ev: arm_id→f1 dicts. Returns per-question mean lists."""
        per_q_cf, per_q_naive = defaultdict(list), defaultdict(list)
        for (qid, sidx), rows in by_state.items():
            sid = f"{qid}::s{sidx}"
            order = ce_rank.get(sid)
            base_sel = sel.get(f"{sid}::base")
            base_ev = ev.get(f"{sid}::base")
            if order is None or base_sel is None or base_ev is None:
                continue
            rows2 = [p for p in rows
                     if p["arm_id"] in sel and p["arm_id"] in ev]
            if not rows2:
                continue
            ce_pick = min(rows2, key=lambda p: order.get(p["doc"], 10**6))
            star_sel = max(rows2, key=lambda p: sel[p["arm_id"]] - base_sel)
            per_q_cf[qid].append(ev[star_sel["arm_id"]] - ev[ce_pick["arm_id"]])
            star_ev = max(rows2, key=lambda p: ev[p["arm_id"]] - base_ev)
            per_q_naive[qid].append(ev[star_ev["arm_id"]] - ev[ce_pick["arm_id"]])
        return per_q_cf, per_q_naive

    ab_cf, b_naive = gains(fa, fb)
    ba_cf, a_naive = gains(fb, fa)

    def summarize(per_q_list):
        # per_q_list: list of dicts → 질문별 평균 gain을 합산 평균
        merged = defaultdict(list)
        for d in per_q_list:
            for q, v in d.items():
                merged[q].extend(v)
        qg = [float(np.mean(v)) for v in merged.values()]
        rng = np.random.default_rng(7)
        boots = [np.mean([qg[j] for j in rng.integers(0, len(qg), len(qg))])
                 for _ in range(10000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return {"mean": float(np.mean(qg)), "ci95": [float(lo), float(hi)],
                "n_questions": len(qg)}

    R = {
        "crossfit_gain": summarize([ab_cf, ba_cf]),
        "naive_gain_within_set": summarize([a_naive, b_naive]),
        "gain_A_to_B": summarize([ab_cf]),
        "gain_B_to_A": summarize([ba_cf]),
        "PASS_ci_excludes_zero": None,
    }
    R["PASS_ci_excludes_zero"] = bool(R["crossfit_gain"]["ci95"][0] > 0)
    with open(f"{D}/CROSSFIT_REPORT.json", "w") as f:
        json.dump(R, f, indent=2)
    print(json.dumps(R, indent=2))
    print("CROSSFIT_EVAL_DONE")


if __name__ == "__main__":
    main()
