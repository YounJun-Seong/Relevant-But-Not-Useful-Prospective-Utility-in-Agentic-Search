"""Feature ablation and imbalance-aware metrics for the state-level gate (C).

Question: is the modest AUC an artefact of class imbalance, and does the rich
feature set add anything over relevance confidence alone?

Feature sets
  ce_margin_only   the single CE top1-top2 margin (relevance confidence)
  ce_only          CE max / mean / std / margin
  phi_only         candidate-set aggregates of the interpretable features
  full             everything, plus the agent probe where available

Metrics
  AUC              insensitive to prevalence
  AP               average precision, whose random baseline IS the prevalence
  precision@prevalence and its lift over the prevalence
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.state_delta import PHI_KEYS
from rq3_state_gate import EVAL, load, state_rows

GROUPS = {
    "ce_margin_only": ["ce_margin"],
    "ce_only": ["ce_max", "ce_mean", "ce_std", "ce_margin"],
    "phi_only": [f"{k}_{a}" for k in PHI_KEYS
                 for a in ("mean", "max", "min", "std")] + ["n_candidates"],
}


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.preprocessing import StandardScaler

    rep = json.load(open("results/rq2_reanalysis/report.json"))
    w = np.array([rep["train"]["coef_unconditional"][k] for k in PHI_KEYS],
                 np.float32)
    Xtr, mtr, cetr, prtr, bytr = load("gate3", "results/gate3")
    Ftr, names, rows_tr = state_rows(Xtr, mtr, cetr, prtr, bytr, w)
    ztr = np.array([r["z"] for r in rows_tr])
    GROUPS["full"] = names
    print(f"[train] {len(Ftr)} states, prevalence {ztr.mean():.3f}", flush=True)

    models = {}
    for g, cols in GROUPS.items():
        idx = [names.index(c) for c in cols if c in names]
        A = Ftr[:, idx]
        sc = StandardScaler().fit(A)
        gbm = HistGradientBoostingClassifier(max_iter=400, max_depth=4,
                                             random_state=42).fit(A, ztr)
        lr = LogisticRegression(max_iter=2000).fit(sc.transform(A), ztr)
        models[g] = (idx, sc, gbm, lr)

    R = {"train_prevalence": float(ztr.mean()), "settings": {}}
    for name, tag, gate_dir in EVAL:
        X, meta, ce, probe, by = load(tag, gate_dir)
        F, _, rows = state_rows(X, meta, ce, probe, by, w)
        z = np.array([r["z"] for r in rows])
        prev = float(z.mean())
        entry = {"n_states": len(rows), "prevalence": prev,
                 "has_probe": bool(rows[0]["has_probe"]), "feature_sets": {}}
        for g, (idx, sc, gbm, lr) in models.items():
            A = F[:, idx]
            p = (gbm.predict_proba(A)[:, 1]
                 + lr.predict_proba(sc.transform(A))[:, 1]) / 2
            k = int(z.sum())
            thr = np.sort(p)[::-1][k - 1] if k > 0 else 1.1
            fire = p >= thr
            precision = float(z[fire].mean()) if fire.any() else float("nan")
            entry["feature_sets"][g] = {
                "auc": float(roc_auc_score(z, p)),
                "average_precision": float(average_precision_score(z, p)),
                "ap_over_prevalence": float(
                    average_precision_score(z, p) / prev),
                "precision_at_prevalence": precision,
                "lift_over_prevalence": float(precision / prev),
            }
        R["settings"][name] = entry
        f = entry["feature_sets"]
        print(f"[{name}] prev {prev:.3f} | AUC ce_margin {f['ce_margin_only']['auc']:.3f}"
              f" ce {f['ce_only']['auc']:.3f} phi {f['phi_only']['auc']:.3f}"
              f" full {f['full']['auc']:.3f}", flush=True)

    os.makedirs("results/rq3_gate_ablation", exist_ok=True)
    out = "results/rq3_gate_ablation/report.json"
    with open(out, "w") as f:
        json.dump(R, f, indent=2)
    print(f"-> {out}")
    print("GATE_ABLATION_DONE")


if __name__ == "__main__":
    main()
