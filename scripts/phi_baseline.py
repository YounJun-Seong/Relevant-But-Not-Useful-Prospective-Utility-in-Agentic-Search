"""φ-only diagnostic (R12): logistic regression on explicit state-delta
features, pairwise (φ_i − φ_j → pref). If this alone reaches ~58-60% with
above-chance CE-disagreement accuracy, the signal the encoder missed lives in
explicit state-transition properties → hybrid model is the path."""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.data_utils import read_jsonl
from piu.state_delta import phi_vector, state_context, PHI_KEYS


def load_phi_pairs(gate_dir, retriever, tau=0.1):
    pairs = [p for p in read_jsonl(os.path.join(gate_dir, f"pairs_tau{tau}.jsonl"))
             if not p["tie"]]
    with open(os.path.join(gate_dir, "questions.json")) as f:
        qmeta = json.load(f)
    base_pre = {r["question_id"]: r for r in read_jsonl(qmeta["pre_path"])}
    ce_rank = {r["state_id"]: {t: i for i, t in enumerate(r["ce_ranking"])}
               for r in read_jsonl(os.path.join(gate_dir, "ce_rankings.jsonl"))}
    ctx_cache, doc_cache = {}, {}
    X, y_meta = [], []
    for p in pairs:
        qid, sidx = p["question_id"], p["state_idx"]
        key = (qid, sidx)
        if key not in ctx_cache:
            ctx_cache[key] = state_context(base_pre[qid], sidx, retriever)
        qt, ev, turn, nev = ctx_cache[key]
        q0 = base_pre[qid]["question"]
        for t in (p["doc_i"], p["doc_j"]):
            if (key, t) not in doc_cache:
                d = retriever.doc_by_title(t)
                doc_cache[(key, t)] = phi_vector(q0, qt, ev, d["text"], turn, nev) \
                    if d else None
        vi, vj = doc_cache[(key, p["doc_i"])], doc_cache[(key, p["doc_j"])]
        if vi is None or vj is None:
            continue
        pref_i = p["pref"] == "i"
        v_pref, v_other = (vi, vj) if pref_i else (vj, vi)
        order = ce_rank.get(f"{qid}::s{sidx}", {})
        rp = order.get(p["doc_i"] if pref_i else p["doc_j"], 10**6)
        ro = order.get(p["doc_j"] if pref_i else p["doc_i"], 10**6)
        X.append(np.array(v_pref) - np.array(v_other))
        y_meta.append({"question_id": qid, "ce_agrees": rp < ro,
                       "abs_margin": abs(p["margin"])})
    return np.array(X), y_meta


def main():
    from piu.retriever import BM25Retriever
    from sklearn.linear_model import LogisticRegression

    retriever = BM25Retriever("data/bm25_index")
    Xtr, mtr = load_phi_pairs("results/gate3", retriever)
    Xte, mte = load_phi_pairs("results/gate2", retriever)
    print(f"train {len(Xtr)} pairs, test {len(Xte)} pairs", flush=True)

    # antisymmetric training: (Δφ → 1) and (−Δφ → 0)
    X = np.vstack([Xtr, -Xtr])
    y = np.concatenate([np.ones(len(Xtr)), np.zeros(len(Xtr))])
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(X, y)

    pred = clf.decision_function(Xte) > 0
    acc = pred.mean()
    dis = [i for i, m in enumerate(mte) if not m["ce_agrees"]]
    dis_acc = pred[dis].mean() if dis else None
    strong = [i for i, m in enumerate(mte) if m["abs_margin"] >= 0.99]
    strong_acc = pred[strong].mean() if strong else None

    coefs = dict(zip(PHI_KEYS, clf.coef_[0].round(3).tolist()))
    report = {
        "phi_only_test_acc": float(acc),
        "phi_only_ce_disagreement_acc": float(dis_acc) if dis_acc is not None else None,
        "n_ce_disagreement": len(dis),
        "phi_only_strong_margin_acc": float(strong_acc) if strong_acc is not None else None,
        "coefficients(pref-vs-other)": coefs,
    }
    os.makedirs("results/phi_baseline", exist_ok=True)
    with open("results/phi_baseline/report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print("PHI_DONE")


if __name__ == "__main__":
    main()
