"""Dump state-level gate predictions so ROC curves can be plotted.

Same training and feature sets as rq3_gate_ablation.py; only the artefacts
differ (per-setting scores and labels instead of summary metrics).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.state_delta import PHI_KEYS
from rq3_gate_ablation import GROUPS
from rq3_state_gate import EVAL, load, state_rows


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rep = json.load(open("results/rq2_reanalysis/report.json"))
    w = np.array([rep["train"]["coef_unconditional"][k] for k in PHI_KEYS],
                 np.float32)
    Xtr, mtr, cetr, prtr, bytr = load("gate3", "results/gate3")
    Ftr, names, rows_tr = state_rows(Xtr, mtr, cetr, prtr, bytr, w)
    ztr = np.array([r["z"] for r in rows_tr])
    GROUPS["full"] = names

    models = {}
    for g, cols in GROUPS.items():
        idx = [names.index(c) for c in cols if c in names]
        A = Ftr[:, idx]
        sc = StandardScaler().fit(A)
        gbm = HistGradientBoostingClassifier(max_iter=400, max_depth=4,
                                             random_state=42).fit(A, ztr)
        lr = LogisticRegression(max_iter=2000).fit(sc.transform(A), ztr)
        models[g] = (idx, sc, gbm, lr)

    out = {}
    for name, tag, gate_dir in EVAL:
        X, meta, ce, probe, by = load(tag, gate_dir)
        F, _, rows = state_rows(X, meta, ce, probe, by, w)
        out[f"{tag}__y"] = np.array([r["z"] for r in rows])
        for g, (idx, sc, gbm, lr) in models.items():
            A = F[:, idx]
            p = (gbm.predict_proba(A)[:, 1]
                 + lr.predict_proba(sc.transform(A))[:, 1]) / 2
            out[f"{tag}__{g}"] = p
        print(f"[{name}] {len(rows)} states", flush=True)
    os.makedirs("results/rq3_gate_ablation", exist_ok=True)
    np.savez("results/rq3_gate_ablation/predictions.npz", **out)
    print("-> results/rq3_gate_ablation/predictions.npz")
    print("GATE_PREDICTIONS_DONE")


if __name__ == "__main__":
    main()
