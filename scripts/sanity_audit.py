"""Day-1 sanity audit (reviewer W1/W2).

Checks whether the pair-level CE-conflict label makes the utility-prediction
task circular, and reconstructs the reported routing numbers from the logs.

W1: on pairs labelled CE-conflicting, "prefer the lower-CE document" is correct
    by construction -- unless CE has no opinion (both documents outside the
    stored CE top-20), in which case the label is contaminated rather than
    circular. Both effects are quantified here.
W2: recomputes ceiling / aligned-only baseline / gated mixture on the exact
    pair universe used by lockbox_eval.py.
"""
import json
import os
import sys
from collections import defaultdict

import joblib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from piu.data_utils import read_jsonl

EPS = 1e-9
UNRANKED = 10 ** 6
DIRS = [("HotpotQA dev", "results/gate2"), ("HotpotQA lockbox", "results/lockbox"),
        ("2WikiMultiHopQA", "results/wiki2"), ("Qwen2.5-14B", "results/xagent"),
        ("MuSiQue", "results/musique")]


def pair_universe(gate_dir, retriever, tau=0.1):
    """Reproduces phi_baseline.load_phi_pairs ordering, with CE-rank details."""
    pairs = [p for p in read_jsonl(os.path.join(gate_dir, f"pairs_tau{tau}.jsonl"))
             if not p["tie"]]
    ce_rank = {r["state_id"]: {t: i for i, t in enumerate(r["ce_ranking"])}
               for r in read_jsonl(os.path.join(gate_dir, "ce_rankings.jsonl"))}
    rows = []
    for p in pairs:
        if retriever.doc_by_title(p["doc_i"]) is None:
            continue
        if retriever.doc_by_title(p["doc_j"]) is None:
            continue
        pref_i = p["pref"] == "i"
        order = ce_rank.get(f"{p['question_id']}::s{p['state_idx']}", {})
        rp = order.get(p["doc_i"] if pref_i else p["doc_j"], UNRANKED)
        ro = order.get(p["doc_j"] if pref_i else p["doc_i"], UNRANKED)
        rows.append({
            "question_id": p["question_id"], "state_idx": p["state_idx"],
            "ce_agrees": rp < ro,          # exactly as in load_phi_pairs
            "rank_pref": rp, "rank_other": ro,
            "both_unranked": rp == UNRANKED and ro == UNRANKED,
            "ce_has_opinion": rp != ro,
            "abs_margin": abs(p["margin"]),
        })
    return rows


def trivial_rules(rows):
    """Accuracy of CE-following and CE-inverting rules.

    Target is always 'the first element is preferred' (rows are oriented that
    way), so: follow-CE is correct iff CE ranks the preferred doc higher;
    invert-CE is correct iff CE ranks it lower. Rank ties (incl. both-unranked)
    are counted as chance (0.5) for both rules.
    """
    def acc(subset, invert):
        if not subset:
            return None
        s = 0.0
        for r in subset:
            if r["rank_pref"] == r["rank_other"]:
                s += 0.5
            else:
                pref_higher = r["rank_pref"] < r["rank_other"]
                s += float(pref_higher != invert)
        return s / len(subset)
    agr = [r for r in rows if r["ce_agrees"]]
    dis = [r for r in rows if not r["ce_agrees"]]
    return {
        "follow_ce": {"all": acc(rows, False), "aligned": acc(agr, False),
                      "conflicting": acc(dis, False)},
        "invert_ce": {"all": acc(rows, True), "aligned": acc(agr, True),
                      "conflicting": acc(dis, True)},
    }


def main():
    from piu.retriever import BM25Retriever
    from phi_baseline import load_phi_pairs
    retriever = BM25Retriever("data/bm25_index")
    ea = joblib.load("results/frozen/specialist_aligned.joblib")
    ed = joblib.load("results/frozen/specialist_conflicting.joblib")

    R = {}
    for name, gate_dir in DIRS:
        rows = pair_universe(gate_dir, retriever)
        X, meta = load_phi_pairs(gate_dir, retriever)
        assert len(rows) == len(X), (name, len(rows), len(X))
        assert all(a["ce_agrees"] == b["ce_agrees"] for a, b in zip(rows, meta))

        agr = np.array([i for i, r in enumerate(rows) if r["ce_agrees"]])
        dis = np.array([i for i, r in enumerate(rows) if not r["ce_agrees"]])
        acc_a = float((ea.decision_function(X[agr]) > 0).mean())
        acc_d = float((ed.decision_function(X[dis]) > 0).mean())
        ceiling = (len(agr) * acc_a + len(dis) * acc_d) / len(X)
        baseline = float((ea.decision_function(X) > 0).mean())
        triv = trivial_rules(rows)
        triv_ceiling = (len(agr) * triv["follow_ce"]["aligned"]
                        + len(dis) * triv["invert_ce"]["conflicting"]) / len(X)

        # state-level headroom
        by_state = defaultdict(list)
        for u in read_jsonl(os.path.join(gate_dir, "utilities.jsonl")):
            by_state[(u["question_id"], u["state_idx"])].append(u)
        ce_rank = {r["state_id"]: {t: i for i, t in enumerate(r["ce_ranking"])}
                   for r in read_jsonl(os.path.join(gate_dir,
                                                    "ce_rankings.jsonl"))}
        H = []
        for (qid, sidx), rr in by_state.items():
            order = ce_rank.get(f"{qid}::s{sidx}")
            if not order:
                continue
            ce_pick = min(rr, key=lambda r: order.get(r["doc"], UNRANKED))
            pi_pick = max(rr, key=lambda r: r["U_task_raw_f1"])
            H.append(pi_pick["arm_d"]["f1"] - ce_pick["arm_d"]["f1"])
        H = np.array(H)

        R[name] = {
            "A_pair_definition": {
                "decided_pairs": len(rows),
                "ce_aligned": int(len(agr)), "ce_conflicting": int(len(dis)),
                "conflicting_share": len(dis) / len(rows),
                "both_docs_outside_ce_top20": sum(r["both_unranked"]
                                                  for r in rows),
                "share_of_conflicting_that_is_both_unranked":
                    sum(1 for r in rows
                        if not r["ce_agrees"] and r["both_unranked"]) / len(dis),
                "ce_has_no_opinion(rank tie)": sum(1 for r in rows
                                                   if not r["ce_has_opinion"]),
            },
            "B_pairwise_accuracy": {
                "follow_ce": triv["follow_ce"],
                "invert_ce": triv["invert_ce"],
                "frozen_aligned_specialist_on_aligned": acc_a,
                "frozen_conflicting_specialist_on_conflicting": acc_d,
            },
            "C_routing": {
                "reported_ceiling(oracle_condition, specialists)": ceiling,
                "trivial_oracle_condition(follow/invert CE)": triv_ceiling,
                "aligned_only_baseline": baseline,
            },
            "D_state_level": {
                "n_states": int(len(H)),
                "H_gt_0": int((H > EPS).sum()),
                "H_ge_0.25": int((H >= 0.25).sum()),
                "rate_H_gt_0": float((H > EPS).mean()),
                "rate_H_ge_0.25": float((H >= 0.25).mean()),
                "mean_H": float(H.mean()),
            },
        }
        print(f"[{name}] done", flush=True)

    os.makedirs("results/sanity_audit", exist_ok=True)
    out = "results/sanity_audit/report.json"
    with open(out, "w") as f:
        json.dump(R, f, indent=2)
    print(json.dumps(R, indent=2))
    print(f"-> {out}")
    print("SANITY_AUDIT_DONE")


if __name__ == "__main__":
    main()
