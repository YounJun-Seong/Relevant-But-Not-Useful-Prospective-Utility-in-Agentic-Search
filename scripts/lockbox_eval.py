"""R16 lockbox evaluation — OPEN EXACTLY ONCE.

Loads only frozen artifacts (results/frozen/*, results/pred_A_early_lowlr) and
the lockbox data (results/lockbox/*), computes the pre-registered endpoints:

  H1  Mismatch replicates: opportunity > 0; Oracle−CE question-level paired
      bootstrap 95% CI excludes 0.
  H2  Conditional structure replicates: (a) frozen conflicting-specialist
      beats chance on lockbox CE-conflicting pairs (binomial p<0.05);
      (b) >=3/4 directional coefficient replication among
      {query_align<0, doc_len<0, redundancy>0, novelty>0} (lockbox refit).
  H3  Prospective identifiability remains limited: frozen gate mixture
      recovers <25% of the lockbox ceiling headroom.
"""
import itertools
import json
import os
import random
import sys
from collections import defaultdict

import joblib
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from piu.data_utils import read_jsonl
from piu.state_delta import PHI_KEYS

LOCK = "results/lockbox"
EPS = 1e-9


def main():
    from piu.retriever import BM25Retriever
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from scipy.stats import binomtest
    from phi_baseline import load_phi_pairs
    from gate_A import pair_lists, build_rows_frozen

    assert not os.path.exists(f"{LOCK}/LOCKBOX_REPORT.json"), \
        "lockbox already opened — refusing to re-open (R16)"
    retriever = BM25Retriever("data/bm25_index")
    R = {}

    # ---------------- H1 ----------------
    utils = read_jsonl(f"{LOCK}/utilities.jsonl")
    ce_rank = {r["state_id"]: r for r in read_jsonl(f"{LOCK}/ce_rankings.jsonl")}
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
    R["H1_oracle_vs_ce"] = {"n_questions": len(qg),
                            "mean": sum(qg) / len(qg),
                            "ci95": [lo, hi], "excludes_zero": lo > 0}
    R["H1_PASS"] = bool(opp > 0 and lo > 0)

    # ---------------- H2 ----------------
    Xl, ml = load_phi_pairs(LOCK, retriever)
    dis = np.array([i for i, m in enumerate(ml) if not m["ce_agrees"]])
    agr = np.array([i for i, m in enumerate(ml) if m["ce_agrees"]])
    ed = joblib.load("results/frozen/specialist_conflicting.joblib")
    ea = joblib.load("results/frozen/specialist_aligned.joblib")
    pred_d = ed.decision_function(Xl[dis]) > 0
    k = int(pred_d.sum())
    bt = binomtest(k, len(pred_d), 0.5, alternative="greater")
    R["H2_specialist"] = {"acc": float(pred_d.mean()), "n": len(pred_d),
                          "binomial_p": bt.pvalue}
    # directional coefficient replication (lockbox refit on conflicting pairs)
    Xd = Xl[dis]
    Xs = np.vstack([Xd, -Xd])
    y = np.concatenate([np.ones(len(Xd)), np.zeros(len(Xd))])
    clf = LogisticRegression(max_iter=1000, fit_intercept=False).fit(Xs, y)
    coefs = dict(zip(PHI_KEYS, clf.coef_[0]))
    expected = {"query_align": -1, "doc_len": -1, "redundancy": 1, "novelty": 1}
    hits = {k2: bool(np.sign(coefs[k2]) == s) for k2, s in expected.items()}
    # bootstrap sign stability
    stab = {}
    rng2 = np.random.default_rng(7)
    signs = []
    for _ in range(500):
        bidx = rng2.choice(dis, size=len(dis), replace=True)
        Xb = Xl[bidx]
        cb = LogisticRegression(max_iter=500, fit_intercept=False).fit(
            np.vstack([Xb, -Xb]),
            np.concatenate([np.ones(len(Xb)), np.zeros(len(Xb))]))
        signs.append(np.sign(cb.coef_[0]))
    signs = np.array(signs)
    for k2 in expected:
        j = PHI_KEYS.index(k2)
        stab[k2] = float((signs[:, j] == np.sign(coefs[k2])).mean())
    R["H2_coefficients"] = {k2: {"coef": float(coefs[k2]),
                                 "direction_replicates": hits[k2],
                                 "bootstrap_sign_stability": stab[k2]}
                            for k2 in expected}
    n_rep = sum(hits.values())
    R["H2_PASS"] = bool(bt.pvalue < 0.05 and n_rep >= 3)

    # ---------------- H3 ----------------
    s_l = np.load(f"{LOCK}/semantic_cache.npz")
    GF = {}
    for r in read_jsonl(f"{LOCK}/gate_features.jsonl"):
        qid, sidx = r["state_id"].split("::s")
        GF[(qid, int(sidx))] = [r["probe_action_answer"], r["probe_conf"],
                                r["probe_len"] / 96.0, r["ce_top_margin"],
                                r["ce_std"], r["ce_mean"]]
    states_l = [(p["question_id"], p["state_idx"])
                for p in pair_lists(LOCK, retriever)]
    S = np.array([GF.get(kk, [0] * 6) for kk in states_l], np.float32)
    dpsi = np.abs(s_l["psi_pref"] - s_l["psi_other"])
    Rmat = np.hstack([np.abs(Xl), dpsi, s_l["state"], S])
    yl = np.array([m["ce_agrees"] for m in ml])
    gbm = joblib.load("results/frozen/gate_gbm.joblib")
    p_gbm = gbm.predict_proba(Rmat)[:, 1]

    from transformers import (AutoModelForSequenceClassification, AutoTokenizer)
    model_name = "cross-encoder/ms-marco-MiniLM-L6-v2"
    tok = AutoTokenizer.from_pretrained(model_name)
    net = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, ignore_mismatched_sizes=True)
    net.load_state_dict(torch.load("results/frozen/gate_transformer.pt"))
    net = net.cuda().eval()
    rows_l = build_rows_frozen(LOCK, retriever)
    preds = []
    with torch.no_grad():
        for bi in range(0, len(rows_l), 64):
            block = rows_l[bi:bi + 64]
            enc = tok([r["text_a"] for r in block],
                      [r["text_b"] for r in block], padding=True,
                      truncation=True, max_length=384,
                      return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                preds.extend(torch.sigmoid(net(**enc).logits.squeeze(-1))
                             .float().cpu().tolist())
    p_tf = np.array(preds)
    p_ens = (p_gbm + p_tf) / 2

    acc_aa = float((ea.decision_function(Xl[agr]) > 0).mean())
    acc_dd = float((ed.decision_function(Xl[dis]) > 0).mean())
    ceiling = (len(agr) * acc_aa + len(dis) * acc_dd) / len(Xl)
    baseline = float((ea.decision_function(Xl) > 0).mean())  # aligned-only
    mixes = {}
    for name, pg in [("gbm", p_gbm), ("transformer", p_tf), ("ens", p_ens)]:
        score = pg * ea.decision_function(Xl) + (1 - pg) * ed.decision_function(Xl)
        pred = score > 0
        mixes[name] = {"overall": float(pred.mean()),
                       "conflicting": float(pred[dis].mean()),
                       "aligned": float(pred[agr].mean())}
    best_mix = max(m["overall"] for m in mixes.values())
    recovery = (best_mix - baseline) / max(ceiling - baseline, 1e-9)
    R["H3"] = {
        "gate_auc": {"gbm": float(roc_auc_score(yl, p_gbm)),
                     "transformer": float(roc_auc_score(yl, p_tf)),
                     "ensemble": float(roc_auc_score(yl, p_ens))},
        "ceiling": ceiling, "baseline_aligned_only": baseline,
        "mixtures": mixes, "best_mixture": best_mix,
        "recovery_ratio": float(recovery),
    }
    R["H3_limited_confirmed"] = bool(recovery < 0.25)

    with open(f"{LOCK}/LOCKBOX_REPORT.json", "w") as f:
        json.dump(R, f, indent=2)
    print(json.dumps(R, indent=2))
    print("LOCKBOX_OPENED_ONCE")


if __name__ == "__main__":
    main()
