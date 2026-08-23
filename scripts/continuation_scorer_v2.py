"""Continuation scorer v2: CE-sticky override with a dev-tuned margin.

The free-selection scorer (v1) underperforms because most states have no
headroom, so any override risk must be paid for. v2 keeps the CE choice unless
another continuation's score exceeds it by a margin. The margin is tuned ONLY
on the HotpotQA dev setting (gate2) and then frozen for the four confirmatory
settings. Also reports pairwise accuracy of the scorer as a diagnostic.
"""
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from continuation_scorer import (EVAL, FEAT, TAU, boot, build_states,
                                 pairs_from)

EPS = 1e-9


def main():
    from sklearn.linear_model import LogisticRegression
    from piu.retriever import BM25Retriever
    retriever = BM25Retriever("data/bm25_index")
    K = 3

    tr = build_states("results/gate3", retriever, K)
    D = pairs_from(tr)
    sd = D.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = np.vstack([D, -D]) / sd
    y = np.concatenate([np.ones(len(D)), np.zeros(len(D))])
    clf = LogisticRegression(max_iter=2000, fit_intercept=False).fit(Xs, y)
    w = clf.coef_[0]

    cache = {tag: build_states(gd, retriever, K)
             for _, tag, gd in EVAL}

    def evaluate(states, margin):
        gain = defaultdict(list)
        n_override = 0
        for s in states:
            rows = s["rows"]
            scores = [float((r["x"] / sd) @ w) for r in rows]
            ce_f1, ce_score = rows[0]["f1"], scores[0]
            j = int(np.argmax(scores))
            if j != 0 and scores[j] - ce_score > margin:
                pick = rows[j]
                n_override += 1
            else:
                pick = rows[0]
            gain[s["qid"]].append(pick["f1"] - ce_f1)
        return gain, n_override

    # tune margin on dev only
    dev = cache["gate2"]
    grid = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
    dev_curve = {}
    for m in grid:
        g, n_o = evaluate(dev, m)
        dev_curve[m] = {"mean": float(np.mean([np.mean(v)
                                               for v in g.values()])),
                        "n_override": n_o}
    best_m = max(dev_curve, key=lambda m: dev_curve[m]["mean"])
    print(f"[dev tuning] {json.dumps({str(k): round(v['mean'], 4) for k, v in dev_curve.items()})}")
    print(f"[dev tuning] chosen margin = {best_m}", flush=True)

    R = {"k": K, "margin_grid_dev": {str(k): v for k, v in dev_curve.items()},
         "chosen_margin": best_m, "settings": {}}
    for name, tag, gd in EVAL:
        st = cache[tag]
        g, n_o = evaluate(st, best_m)
        r = boot(g)
        # pairwise accuracy diagnostic on decided pairs
        Dte = pairs_from(st)
        acc = float((((Dte / sd) @ w) > 0).mean()) if len(Dte) else None
        # oracle@k for reference
        g_k = defaultdict(list)
        for s in st:
            g_k[s["qid"]].append(max(x["f1"] for x in s["rows"])
                                 - s["rows"][0]["f1"])
        rk = boot(g_k)
        R["settings"][name] = {
            "n_states": len(st), "n_overrides": n_o,
            "override_rate": n_o / len(st),
            "gain_vs_ce": r, "oracle_at_k": rk,
            "pairwise_acc_decided": acc,
            "recovered_fraction": r["mean"] / rk["mean"] if rk["mean"] > 0 else None,
        }
        print(f"[{name}] margin={best_m} overrides {n_o}/{len(st)} | "
              f"gain {r['mean']:+.4f} [{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] "
              f"| pair acc {acc:.3f}", flush=True)

    out = "results/continuation_scorer/report_v2.json"
    with open(out, "w") as f:
        json.dump(R, f, indent=2)
    print(f"-> {out}")
    print("SCORER_V2_DONE")


if __name__ == "__main__":
    main()
