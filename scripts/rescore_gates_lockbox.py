"""Re-score the FROZEN gates on the (already opened) lockbox to save per-pair
scores for an ROC figure. This is a pure re-scoring of frozen artifacts:
identical inputs/models as lockbox_eval.py; the LOCKBOX_REPORT is not
touched. Asserts AUCs match the sealed report."""
import json
import os
import sys

import joblib
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from piu.data_utils import read_jsonl
from piu.retriever import BM25Retriever
from phi_baseline import load_phi_pairs
from gate_A import pair_lists, build_rows_frozen
from sklearn.metrics import roc_auc_score, roc_curve

LOCK = "results/lockbox"


def main():
    retriever = BM25Retriever("data/bm25_index")
    Xl, ml = load_phi_pairs(LOCK, retriever)
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

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
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

    sealed = json.load(open(f"{LOCK}/LOCKBOX_REPORT.json"))["H3"]["gate_auc"]
    aucs = {"gbm": float(roc_auc_score(yl, p_gbm)),
            "transformer": float(roc_auc_score(yl, p_tf)),
            "ensemble": float(roc_auc_score(yl, p_ens))}
    print("sealed:", sealed)
    print("rescored:", aucs)
    for k in aucs:
        assert abs(aucs[k] - sealed[k]) < 5e-3, f"AUC mismatch for {k}"
    out = {"y_ce_agrees": yl.astype(int).tolist(), "p_gbm": p_gbm.tolist(),
           "p_transformer": p_tf.tolist(), "p_ensemble": p_ens.tolist(),
           "auc": aucs}
    # ROC curves (positive class = CE-conflicting = 1 - y)
    for name, p in [("gbm", p_gbm), ("transformer", p_tf), ("ens", p_ens)]:
        fpr, tpr, _ = roc_curve(1 - yl, 1 - p)
        out[f"roc_{name}"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    with open(f"{LOCK}/gate_scores.json", "w") as f:
        json.dump(out, f)
    print("RESCORE_DONE", len(yl))


if __name__ == "__main__":
    main()
