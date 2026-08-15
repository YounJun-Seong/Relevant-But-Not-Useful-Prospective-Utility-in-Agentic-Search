"""R16: train-and-save the frozen analysis artifacts (seeded) BEFORE the
lockbox is generated. These exact files are the only models the lockbox
evaluation may load.

Saves to results/frozen/:
  specialist_aligned.joblib / specialist_conflicting.joblib  (φ logistic)
  gate_gbm.joblib                                            (full-feature GBM)
  gate_transformer.pt                                        (MiniLM pair gate)
  (global predictor: results/pred_A_early_lowlr/best.pt, already frozen)
"""
import json
import os
import sys

import joblib
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from piu.data_utils import read_jsonl

OUT = "results/frozen"
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(42)
np.random.seed(42)


def main():
    from piu.retriever import BM25Retriever
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from gate_A import pair_lists

    retriever = BM25Retriever("data/bm25_index")
    z = np.load("results/phi_cache.npz", allow_pickle=True)
    Xtr, Xte = z["Xtr"], z["Xte"]
    mtr, mte = list(z["mtr"]), list(z["mte"])
    s3 = np.load("results/gate3/semantic_cache.npz")
    s2 = np.load("results/gate2/semantic_cache.npz")
    str_tr = [(p["question_id"], p["state_idx"])
              for p in pair_lists("results/gate3", retriever)]
    str_te = [(p["question_id"], p["state_idx"])
              for p in pair_lists("results/gate2", retriever)]
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

    ytr = np.array([m["ce_agrees"] for m in mtr])
    yte = np.array([m["ce_agrees"] for m in mte])

    # specialists
    tr_a = np.array([i for i, m in enumerate(mtr) if m["ce_agrees"]])
    tr_d = np.array([i for i, m in enumerate(mtr) if not m["ce_agrees"]])

    def fit(idxs):
        Xd = Xtr[idxs]
        Xs = np.vstack([Xd, -Xd])
        y = np.concatenate([np.ones(len(Xd)), np.zeros(len(Xd))])
        return LogisticRegression(max_iter=1000, fit_intercept=False).fit(Xs, y)
    ea, ed = fit(tr_a), fit(tr_d)
    joblib.dump(ea, f"{OUT}/specialist_aligned.joblib")
    joblib.dump(ed, f"{OUT}/specialist_conflicting.joblib")

    # GBM gate
    Rtr = gate_matrix(Xtr, s3, str_tr)
    Rte = gate_matrix(Xte, s2, str_te)
    gbm = HistGradientBoostingClassifier(max_iter=400, max_depth=4,
                                         random_state=42).fit(Rtr, ytr)
    joblib.dump(gbm, f"{OUT}/gate_gbm.joblib")
    auc_gbm = roc_auc_score(yte, gbm.predict_proba(Rte)[:, 1])

    # transformer gate (seeded rerun of gate_A step 2)
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer)
    from gate_A import build_rows_frozen

    rows_tr = build_rows_frozen("results/gate3", retriever)
    rows_te = build_rows_frozen("results/gate2", retriever)
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
    net.eval()
    torch.save(net.state_dict(), f"{OUT}/gate_transformer.pt")
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
    auc_tf = roc_auc_score(yte, np.array(preds))

    meta = {"dev_auc_gbm": float(auc_gbm), "dev_auc_transformer": float(auc_tf),
            "note": "frozen artifacts for lockbox; dev numbers of THESE files"}
    with open(f"{OUT}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print("FREEZE_DONE")


if __name__ == "__main__":
    main()
