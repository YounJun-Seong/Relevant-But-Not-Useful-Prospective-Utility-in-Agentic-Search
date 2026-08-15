"""R17 2Wiki cross-dataset replication evaluation — run once.

Confirmatory: H1' (opportunity>0, Oracle−CE CI excludes 0),
H2a' (frozen HotpotQA conflicting-specialist beats chance on 2Wiki
conflicting pairs), directions sign(query_align)<0 AND sign(doc_len)<0.
Exploratory (labeled): redundancy/novelty signs, frozen-gate transfer.
"""
import json
import os
import random
import sys
from collections import defaultdict

import joblib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from piu.data_utils import read_jsonl
from piu.state_delta import PHI_KEYS

W = "results/wiki2"
EPS = 1e-9


def main():
    from piu.retriever import BM25Retriever
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from scipy.stats import binomtest
    from phi_baseline import load_phi_pairs

    assert not os.path.exists(f"{W}/WIKI2_REPORT.json"), \
        "2wiki replication already evaluated — refusing to re-run (R17)"
    retriever = BM25Retriever("data/bm25_index")
    R = {}

    utils = read_jsonl(f"{W}/utilities.jsonl")
    ce_rank = {r["state_id"]: r for r in read_jsonl(f"{W}/ce_rankings.jsonl")}
    by_state = defaultdict(list)
    for u in utils:
        by_state[(u["question_id"], u["state_idx"])].append(u)
    opp = sum(1 for rows in by_state.values()
              if max(r["U_task_raw_f1"] for r in rows) > EPS)
    R["H1_opportunity"] = {"decisive": opp, "states": len(by_state),
                           "rate": opp / len(by_state)}
    per_q = defaultdict(list)
    for (qid, sidx), rows in by_state.items():
        cr = ce_rank.get(f"{qid}::s{sidx}")
        if not cr:
            continue
        order = {t: i for i, t in enumerate(cr["ce_ranking"])}
        ce_pick = min(rows, key=lambda r: order.get(r["doc"], 10**6))
        pi_pick = max(rows, key=lambda r: r["U_task_raw_f1"])
        per_q[qid].append(pi_pick["arm_d"]["f1"] - ce_pick["arm_d"]["f1"])
    qg = [sum(v) / len(v) for v in per_q.values()]
    rng = random.Random(7)
    boots = sorted(
        sum(qg[rng.randrange(len(qg))] for _ in qg) / len(qg)
        for _ in range(10000))
    lo, hi = boots[250], boots[9750]
    R["H1_oracle_vs_ce"] = {"n_questions": len(qg), "mean": sum(qg) / len(qg),
                            "ci95": [lo, hi], "excludes_zero": lo > 0}
    R["H1_PASS"] = bool(opp > 0 and lo > 0)

    Xw, mw = load_phi_pairs(W, retriever)
    dis = np.array([i for i, m in enumerate(mw) if not m["ce_agrees"]])
    agr = np.array([i for i, m in enumerate(mw) if m["ce_agrees"]])
    R["n_pairs"] = {"total": len(Xw), "conflicting": len(dis),
                    "aligned": len(agr)}
    ed = joblib.load("results/frozen/specialist_conflicting.joblib")
    ea = joblib.load("results/frozen/specialist_aligned.joblib")
    pred_d = ed.decision_function(Xw[dis]) > 0
    bt = binomtest(int(pred_d.sum()), len(pred_d), 0.5, alternative="greater")
    R["H2a_specialist_transfer"] = {"acc": float(pred_d.mean()),
                                    "n": len(pred_d),
                                    "binomial_p": bt.pvalue}
    R["H2a_PASS"] = bool(bt.pvalue < 0.05)

    Xd = Xw[dis]
    clf = LogisticRegression(max_iter=1000, fit_intercept=False).fit(
        np.vstack([Xd, -Xd]),
        np.concatenate([np.ones(len(Xd)), np.zeros(len(Xd))]))
    coefs = dict(zip(PHI_KEYS, clf.coef_[0]))
    rng2 = np.random.default_rng(7)
    signs = []
    for _ in range(500):
        bidx = rng2.choice(dis, size=len(dis), replace=True)
        Xb = Xw[bidx]
        cb = LogisticRegression(max_iter=500, fit_intercept=False).fit(
            np.vstack([Xb, -Xb]),
            np.concatenate([np.ones(len(Xb)), np.zeros(len(Xb))]))
        signs.append(np.sign(cb.coef_[0]))
    signs = np.array(signs)
    def entry(k):
        j = PHI_KEYS.index(k)
        return {"coef": float(coefs[k]),
                "bootstrap_sign_stability":
                    float((signs[:, j] == np.sign(coefs[k])).mean())}
    R["confirmatory_directions"] = {
        "query_align_neg": bool(coefs["query_align"] < 0),
        "doc_len_neg": bool(coefs["doc_len"] < 0),
        "query_align": entry("query_align"), "doc_len": entry("doc_len"),
    }
    R["directions_PASS"] = bool(coefs["query_align"] < 0 and coefs["doc_len"] < 0)
    R["exploratory_signature"] = {k: entry(k) for k in ("redundancy", "novelty",
                                                        "entity_gain",
                                                        "gap_coverage")}

    # exploratory: frozen specialists ceiling + aligned-only baseline
    acc_aa = float((ea.decision_function(Xw[agr]) > 0).mean())
    acc_dd = float(pred_d.mean())
    R["exploratory_ceiling"] = {
        "aligned_expert_on_aligned": acc_aa,
        "conflicting_expert_on_conflicting": acc_dd,
        "ceiling": (len(agr) * acc_aa + len(dis) * acc_dd) / len(Xw),
        "aligned_only_baseline": float((ea.decision_function(Xw) > 0).mean()),
    }

    with open(f"{W}/WIKI2_REPORT.json", "w") as f:
        json.dump(R, f, indent=2)
    print(json.dumps(R, indent=2))
    print("WIKI2_EVAL_DONE")


if __name__ == "__main__":
    main()
