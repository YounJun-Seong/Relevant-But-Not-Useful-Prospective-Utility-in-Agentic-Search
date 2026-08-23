"""Is the anti-alignment signature inherited from CE conditioning? (W1, part 2)

Three analyses:
  A  candidate-level Spearman correlation between the CE score and the
     interpretable features (does CE already encode query alignment / length?);
  B  pairwise utility models refit on ALL decided pairs with the CE score
     difference included as a covariate -- if query_align stays negative after
     controlling for dCE, the signature is not merely inherited;
  C  whether adding phi to CE improves selection: realized downstream F1 of a
     CE+phi selector against CE alone.

Features are standardized by their training standard deviation so coefficients
are comparable across features.
"""
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.data_utils import read_jsonl
from piu.state_delta import PHI_KEYS

EPS = 1e-9
KEYS = ["ce"] + PHI_KEYS
EVAL = [("HotpotQA dev", "gate2", "results/gate2"),
        ("HotpotQA lockbox", "lockbox", "results/lockbox"),
        ("2WikiMultiHopQA", "wiki2", "results/wiki2"),
        ("Qwen2.5-14B", "xagent", "results/xagent"),
        ("MuSiQue", "musique", "results/musique")]


def load(tag, gate_dir):
    z = np.load(f"results/phi_candidates/{tag}.npz", allow_pickle=True)
    X = z["X"]
    meta = [json.loads(m) for m in z["meta"]]
    ce = {r["state_id"]: r["scores"]
          for r in read_jsonl(os.path.join(gate_dir, "candidate_ce_scores.jsonl"))}
    keep, cev = [], []
    for i, m in enumerate(meta):
        s = ce.get(f"{m['question_id']}::s{m['state_idx']}", {}).get(m["doc"])
        if s is None:
            continue
        keep.append(i)
        cev.append(s)
    X = X[keep]
    meta = [meta[i] for i in keep]
    F = np.hstack([np.array(cev, np.float32)[:, None], X])  # ce first
    return F, meta


def state_index(meta):
    idx = defaultdict(list)
    for i, m in enumerate(meta):
        idx[(m["question_id"], m["state_idx"])].append(i)
    return idx


def pairs(F, meta, idx, tau=0.1):
    D = []
    for rows in idx.values():
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                i, j = rows[a], rows[b]
                margin = meta[i]["f1"] - meta[j]["f1"]
                if abs(margin) <= tau:
                    continue
                p, o = (i, j) if margin > 0 else (j, i)
                D.append(F[p] - F[o])
    return np.array(D, np.float32)


def fit(D, cols):
    from sklearn.linear_model import LogisticRegression
    A = D[:, cols]
    Xs = np.vstack([A, -A])
    y = np.concatenate([np.ones(len(A)), np.zeros(len(A))])
    return LogisticRegression(max_iter=1000, fit_intercept=False).fit(Xs, y)


def boot_signs(D, cols, n=300, seed=7):
    rng = np.random.default_rng(seed)
    base = fit(D, cols).coef_[0]
    S = []
    for _ in range(n):
        b = rng.integers(0, len(D), len(D))
        S.append(np.sign(fit(D[b], cols).coef_[0]))
    S = np.array(S)
    return base, [(S[:, k] == np.sign(base[k])).mean() for k in range(len(cols))]


def boot(per_q, n_boot=10000, seed=7):
    qg = [sum(v) / len(v) for v in per_q.values()]
    rng = random.Random(seed)
    b = sorted(sum(qg[rng.randrange(len(qg))] for _ in qg) / len(qg)
               for _ in range(n_boot))
    return {"mean": sum(qg) / len(qg),
            "ci95": [b[int(.025 * n_boot)], b[int(.975 * n_boot)]]}


def main():
    from scipy.stats import spearmanr
    Ftr, mtr = load("gate3", "results/gate3")
    itr = state_index(mtr)
    Dtr = pairs(Ftr, mtr, itr)
    sd = Dtr.std(axis=0)
    sd[sd == 0] = 1.0
    Dtr_z = Dtr / sd
    ALL = list(range(len(KEYS)))
    PHI_ONLY = list(range(1, len(KEYS)))
    w_full = fit(Dtr_z, ALL).coef_[0]
    w_phi = fit(Dtr_z, PHI_ONLY).coef_[0]

    R = {"train_pairs": int(len(Dtr)),
         "coef_ce_adjusted(train)": dict(zip(KEYS, np.round(w_full, 3).tolist())),
         "coef_phi_only(train)": dict(zip(PHI_KEYS, np.round(w_phi, 3).tolist())),
         "settings": {}}

    for name, tag, gate_dir in EVAL:
        F, meta = load(tag, gate_dir)
        idx = state_index(meta)
        # --- A. candidate-level correlations with the CE score
        corr = {}
        for k, key in enumerate(PHI_KEYS):
            rho, p = spearmanr(F[:, 0], F[:, k + 1])
            corr[key] = {"spearman_rho": float(rho), "p": float(p)}
        # --- B. CE-adjusted refit on this setting's decided pairs
        D = pairs(F, meta, idx) / sd
        c_full, s_full = boot_signs(D, ALL)
        c_phi, s_phi = boot_signs(D, PHI_ONLY)
        # --- C. selection with CE alone vs CE+phi
        gain = {"ce_plus_phi": defaultdict(list), "phi_only": defaultdict(list)}
        for (qid, sidx), rows in idx.items():
            ce_pick = max(rows, key=lambda i: F[i, 0])
            for nm, w, cols in (("ce_plus_phi", w_full, ALL),
                                ("phi_only", w_phi, PHI_ONLY)):
                pick = max(rows, key=lambda i: float((F[i, cols] / sd[cols]) @ w))
                gain[nm][qid].append(meta[pick]["f1"] - meta[ce_pick]["f1"])
        R["settings"][name] = {
            "A_ce_feature_correlation": corr,
            "B_ce_adjusted_coefficients": {
                k: {"coef": float(c_full[i]), "bootstrap_sign_stability": float(s_full[i])}
                for i, k in enumerate(KEYS)},
            "B_phi_only_coefficients": {
                k: {"coef": float(c_phi[i]), "bootstrap_sign_stability": float(s_phi[i])}
                for i, k in enumerate(PHI_KEYS)},
            "C_selection_gain_vs_ce": {k: boot(v) for k, v in gain.items()},
        }
        print(f"[{name}] done", flush=True)

    os.makedirs("results/rq2_ce_adjusted", exist_ok=True)
    out = "results/rq2_ce_adjusted/report.json"
    with open(out, "w") as f:
        json.dump(R, f, indent=2)
    print(f"-> {out}")
    print("CE_ADJUSTED_DONE")


if __name__ == "__main__":
    main()
