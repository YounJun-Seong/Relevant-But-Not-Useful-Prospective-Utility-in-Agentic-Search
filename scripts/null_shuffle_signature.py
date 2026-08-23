"""Null-shuffle test for the anti-alignment signature (reviewer W1/RQ2).

Conditioning on 'CE and PIU disagree' can by itself induce a negative
query-alignment coefficient, because the CE score is strongly correlated with
query alignment. The test:

  keep the CE scores and phi(s,d) fixed; permute the measured OUTCOMES among the
  candidates of each state; rebuild the decided pairs and the CE-conflicting
  subset exactly as the pipeline does; refit the interpretable model.

Under this null there is no true relation between phi and utility, but the
conditioning structure is preserved. If the real coefficient lies inside the
null distribution, the signature is an artifact of the conditioning; if it is
more extreme, it is not explained by conditioning alone.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.data_utils import read_jsonl
from piu.state_delta import PHI_KEYS

EVAL = [("HotpotQA dev", "gate2", "results/gate2"),
        ("HotpotQA lockbox", "lockbox", "results/lockbox"),
        ("2WikiMultiHopQA", "wiki2", "results/wiki2"),
        ("Qwen2.5-14B", "xagent", "results/xagent"),
        ("MuSiQue", "musique", "results/musique")]
WATCH = ["query_align", "doc_len", "redundancy", "novelty"]


def load(tag, gate_dir):
    z = np.load(f"results/phi_candidates/{tag}.npz", allow_pickle=True)
    X, meta = z["X"], [json.loads(m) for m in z["meta"]]
    ce = {r["state_id"]: r["scores"]
          for r in read_jsonl(os.path.join(gate_dir, "candidate_ce_scores.jsonl"))}
    by = defaultdict(list)
    for i, m in enumerate(meta):
        by[(m["question_id"], m["state_idx"])].append(i)
    states = []
    for (qid, sidx), rows in by.items():
        sc = ce.get(f"{qid}::s{sidx}")
        if not sc:
            continue
        cand = [i for i in rows if meta[i]["doc"] in sc]
        if len(cand) < 2:
            continue
        states.append({
            "idx": np.array(cand),
            "y": np.array([meta[i]["f1"] for i in cand]),
            "ce": np.array([sc[meta[i]["doc"]] for i in cand]),
        })
    return X, states


def conflicting_deltas(X, states, y_override=None, tau=0.1):
    """Delta-phi rows for decided pairs that are CE-conflicting."""
    D = []
    for k, s in enumerate(states):
        y = s["y"] if y_override is None else y_override[k]
        n = len(y)
        for a in range(n):
            for b in range(a + 1, n):
                m = y[a] - y[b]
                if abs(m) <= tau:
                    continue
                p, o = (a, b) if m > 0 else (b, a)
                if s["ce"][p] > s["ce"][o]:      # CE agrees -> aligned
                    continue
                D.append(X[s["idx"][p]] - X[s["idx"][o]])
    return np.array(D, np.float32)


def fit_coef(D, max_iter=300):
    from sklearn.linear_model import LogisticRegression
    if len(D) < 30:
        return None
    Xs = np.vstack([D, -D])
    y = np.concatenate([np.ones(len(D)), np.zeros(len(D))])
    return LogisticRegression(max_iter=max_iter,
                              fit_intercept=False).fit(Xs, y).coef_[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results/null_shuffle")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    R = {}
    for name, tag, gate_dir in EVAL:
        X, states = load(tag, gate_dir)
        D_real = conflicting_deltas(X, states)
        c_real = fit_coef(D_real)
        rng = np.random.default_rng(args.seed)
        null = []
        sizes = []
        for _ in range(args.n_perm):
            yo = [rng.permutation(s["y"]) for s in states]
            D = conflicting_deltas(X, states, y_override=yo)
            sizes.append(len(D))
            c = fit_coef(D)
            if c is not None:
                null.append(c)
        null = np.array(null)
        entry = {"n_conflicting_pairs_real": int(len(D_real)),
                 "n_conflicting_pairs_null_mean": float(np.mean(sizes)),
                 "n_permutations": int(len(null)), "features": {}}
        for k in WATCH:
            j = PHI_KEYS.index(k)
            nd = null[:, j]
            entry["features"][k] = {
                "real_coef": float(c_real[j]),
                "null_mean": float(nd.mean()),
                "null_ci95": [float(np.percentile(nd, 2.5)),
                              float(np.percentile(nd, 97.5))],
                "pct_null_at_least_as_extreme": float(
                    (nd <= c_real[j]).mean() if c_real[j] < 0
                    else (nd >= c_real[j]).mean()),
            }
        R[name] = entry
        print(f"[{name}] real conflicting pairs {len(D_real)}, "
              f"null {len(null)} perms", flush=True)
    out = os.path.join(args.out, "report.json")
    with open(out, "w") as f:
        json.dump(R, f, indent=2)
    print(f"-> {out}")
    print("NULL_SHUFFLE_DONE")


if __name__ == "__main__":
    main()
