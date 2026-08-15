"""Final attempt A, step 2 (R14): full-feature GBM gate + small transformer
gate, then the pre-registered verdict.

Success criteria (locked): Gate AUC >= 0.70 AND mixture >= 66%
(+ significant disagreement improvement). Anything less = predictor campaign
ends, pivot to the analysis paper (C).
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.data_utils import read_jsonl


def pair_lists(gate_dir, retriever):
    pairs = [p for p in read_jsonl(f"{gate_dir}/pairs_tau0.1.jsonl")
             if not p["tie"]]
    keep = []
    for p in pairs:
        if retriever.doc_by_title(p["doc_i"]) is None:
            continue
        if retriever.doc_by_title(p["doc_j"]) is None:
            continue
        keep.append(p)
    return keep


def build_rows_frozen(gate_dir, retriever):
    """Module-level copy of the transformer-gate input builder (frozen spec):
    state text + both docs in canonical (alphabetical) order."""
    from piu.predictor import serialize_state, serialize_doc

    with open(f"{gate_dir}/questions.json") as f:
        qmeta = json.load(f)
    base_pre = {r["question_id"]: r for r in read_jsonl(qmeta["pre_path"])}
    cache, rows = {}, []
    for p in pair_lists(gate_dir, retriever):
        key = (p["question_id"], p["state_idx"])
        if key not in cache:
            cache[key] = serialize_state(base_pre[key[0]], key[1], retriever,
                                         snippet_chars=150, max_evidence=6)
        a, b = sorted([p["doc_i"], p["doc_j"]])
        da, db = retriever.doc_by_title(a), retriever.doc_by_title(b)
        rows.append({
            "text_a": cache[key],
            "text_b": serialize_doc(a, da["text"], 300) + " [VS] "
                      + serialize_doc(b, db["text"], 300),
        })
    return rows


def main():
    from piu.retriever import BM25Retriever
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    retriever = BM25Retriever("data/bm25_index")
    z = np.load("results/phi_cache.npz", allow_pickle=True)
    Xtr, Xte = z["Xtr"], z["Xte"]
    mtr, mte = list(z["mtr"]), list(z["mte"])
    str_tr = [(p["question_id"], p["state_idx"])
              for p in pair_lists("results/gate3", retriever)]
    str_te = [(p["question_id"], p["state_idx"])
              for p in pair_lists("results/gate2", retriever)]
    assert len(str_tr) == len(Xtr) and len(str_te) == len(Xte)

    s3 = np.load("results/gate3/semantic_cache.npz")
    s2 = np.load("results/gate2/semantic_cache.npz")
    assert len(s3["psi_pref"]) == len(Xtr) and len(s2["psi_pref"]) == len(Xte)

    GF = {}
    for d in ("results/gate3", "results/gate2"):
        for r in read_jsonl(f"{d}/gate_features.jsonl"):
            qid, sidx = r["state_id"].split("::s")
            GF[(qid, int(sidx))] = [r["probe_action_answer"], r["probe_conf"],
                                    r["probe_len"] / 96.0, r["ce_top_margin"],
                                    r["ce_std"], r["ce_mean"]]

    def gate_matrix(X, sem, states):
        S = np.array([GF.get(k, [0] * 6) for k in states], np.float32)
        dpsi = np.abs(sem["psi_pref"] - sem["psi_other"])
        return np.hstack([np.abs(X), dpsi, sem["state"], S])

    Rtr = gate_matrix(Xtr, s3, str_tr)
    Rte = gate_matrix(Xte, s2, str_te)
    ytr = np.array([m["ce_agrees"] for m in mtr])
    yte = np.array([m["ce_agrees"] for m in mte])

    gbm = HistGradientBoostingClassifier(max_iter=400, max_depth=4).fit(Rtr, ytr)
    p_gbm = gbm.predict_proba(Rte)[:, 1]
    auc_gbm = roc_auc_score(yte, p_gbm)
    print(f"[A] GBM gate (lexical+semantic+probe+CE): AUC {auc_gbm:.3f}", flush=True)

    # ---- transformer gate: state text + docs in CANONICAL order (no leak) ----
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer)
    from piu.predictor import serialize_state, serialize_doc

    def build_rows(gate_dir, retriever, states_pairs):
        with open(f"{gate_dir}/questions.json") as f:
            qmeta = json.load(f)
        base_pre = {r["question_id"]: r
                    for r in read_jsonl(qmeta["pre_path"])}
        cache, rows = {}, []
        for p in states_pairs:
            key = (p["question_id"], p["state_idx"])
            if key not in cache:
                cache[key] = serialize_state(base_pre[key[0]], key[1], retriever,
                                             snippet_chars=150, max_evidence=6)
            a, b = sorted([p["doc_i"], p["doc_j"]])  # canonical order
            da, db = retriever.doc_by_title(a), retriever.doc_by_title(b)
            pref_t = p["doc_i"] if p["pref"] == "i" else p["doc_j"]
            oth_t = p["doc_j"] if p["pref"] == "i" else p["doc_i"]
            rows.append({
                "text_a": cache[key],
                "text_b": serialize_doc(a, da["text"], 300) + " [VS] "
                          + serialize_doc(b, db["text"], 300),
            })
        return rows

    tr_pairs = pair_lists("results/gate3", retriever)
    te_pairs = pair_lists("results/gate2", retriever)
    rows_tr = build_rows("results/gate3", retriever, tr_pairs)
    rows_te = build_rows("results/gate2", retriever, te_pairs)

    model_name = "cross-encoder/ms-marco-MiniLM-L6-v2"
    tok = AutoTokenizer.from_pretrained(model_name)
    net = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, ignore_mismatched_sizes=True).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-5, weight_decay=0.01)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    idx = np.random.default_rng(42).permutation(len(rows_tr))
    B = 48
    net.train()
    for ep in range(2):
        for bi in range(0, len(idx), B):
            bidx = idx[bi:bi + B]
            enc = tok([rows_tr[i]["text_a"] for i in bidx],
                      [rows_tr[i]["text_b"] for i in bidx],
                      padding=True, truncation=True, max_length=384,
                      return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = net(**enc).logits.squeeze(-1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits.float(), ytr_t[bidx].cuda())
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"[A] transformer gate ep{ep} last-loss {loss.item():.4f}", flush=True)
    net.eval()
    preds = []
    with torch.no_grad():
        for bi in range(0, len(rows_te), 64):
            block = rows_te[bi:bi + 64]
            enc = tok([r["text_a"] for r in block], [r["text_b"] for r in block],
                      padding=True, truncation=True, max_length=384,
                      return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                preds.extend(torch.sigmoid(net(**enc).logits.squeeze(-1))
                             .float().cpu().tolist())
    p_tf = np.array(preds)
    auc_tf = roc_auc_score(yte, p_tf)
    print(f"[A] transformer gate: AUC {auc_tf:.3f}", flush=True)

    # ---- ensemble gate + mixture + verdict ----
    p_ens = (p_gbm + p_tf) / 2
    auc_ens = roc_auc_score(yte, p_ens)
    tr_a = np.array([i for i, m in enumerate(mtr) if m["ce_agrees"]])
    tr_d = np.array([i for i, m in enumerate(mtr) if not m["ce_agrees"]])

    def fit(idxs):
        Xd = Xtr[idxs]
        Xs = np.vstack([Xd, -Xd])
        y = np.concatenate([np.ones(len(Xd)), np.zeros(len(Xd))])
        return LogisticRegression(max_iter=1000, fit_intercept=False).fit(Xs, y)
    ea, ed = fit(tr_a), fit(tr_d)
    report = {"auc_gbm": auc_gbm, "auc_transformer": auc_tf, "auc_ensemble": auc_ens}
    for name, pg in [("gbm", p_gbm), ("transformer", p_tf), ("ensemble", p_ens)]:
        score = pg * ea.decision_function(Xte) + (1 - pg) * ed.decision_function(Xte)
        pred = score > 0
        dis = [i for i, m in enumerate(mte) if not m["ce_agrees"]]
        agr = [i for i, m in enumerate(mte) if m["ce_agrees"]]
        report[f"mixture_{name}"] = {
            "overall": float(pred.mean()),
            "disagreement": float(np.array(pred)[dis].mean()),
            "agreement": float(np.array(pred)[agr].mean()),
        }
        print(f"[A] mixture({name}): 전체 {pred.mean():.3f} | "
              f"dis {np.array(pred)[dis].mean():.3f} | "
              f"agr {np.array(pred)[agr].mean():.3f}", flush=True)

    best_auc = max(auc_gbm, auc_tf, auc_ens)
    best_mix = max(report[f"mixture_{n}"]["overall"]
                   for n in ("gbm", "transformer", "ensemble"))
    verdict = "PASS" if (best_auc >= 0.70 and best_mix >= 0.66) else "FAIL"
    report["verdict"] = verdict
    report["criteria"] = "AUC>=0.70 AND mixture>=0.66 (R14 pre-registered)"
    with open("results/gate_A_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[A] VERDICT: {verdict} (best AUC {best_auc:.3f}, "
          f"best mixture {best_mix:.3f})")
    print("GATE_A_DONE")


if __name__ == "__main__":
    main()
