"""RQ2 without conflict conditioning (reviewer W1).

The conditional specialist is trained on pairs whose label is defined by
disagreement with CE, which makes 'invert CE' correct by construction. Here the
specialist is retrained on ALL decided utility pairs and evaluated on tasks a
trivial CE inversion cannot solve:

  (a) pairwise utility accuracy on all decided pairs, against follow-CE and
      invert-CE baselines;
  (b) top-1 candidate selection over the full candidate set D_t;
  (c) realized downstream F1 of the selected candidate.

Selection uses the same model: for an antisymmetric linear model on
phi_i - phi_j, the pairwise tournament winner is argmax_d w . phi(d), so the
scorer is a drop-in retrieval selector. No new rollouts: every candidate's
intervention outcome is already stored.
"""
import argparse
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
EVAL = [("HotpotQA dev", "gate2", "results/gate2"),
        ("HotpotQA lockbox", "lockbox", "results/lockbox"),
        ("2WikiMultiHopQA", "wiki2", "results/wiki2"),
        ("Qwen2.5-14B", "xagent", "results/xagent"),
        ("MuSiQue", "musique", "results/musique")]


def load_phi(tag):
    z = np.load(f"results/phi_candidates/{tag}.npz", allow_pickle=True)
    X = z["X"]
    meta = [json.loads(m) for m in z["meta"]]
    return X, meta


def load_ce(gate_dir):
    return {r["state_id"]: r["scores"]
            for r in read_jsonl(os.path.join(gate_dir,
                                             "candidate_ce_scores.jsonl"))}


def state_index(meta):
    idx = defaultdict(list)
    for i, m in enumerate(meta):
        idx[(m["question_id"], m["state_idx"])].append(i)
    return idx


def build_pairs(X, meta, idx, ce, tau=0.1, conflicting_only=False):
    """All decided within-state pairs, oriented so the first is preferred."""
    D, ce_agree = [], []
    for (qid, sidx), rows in idx.items():
        sc = ce.get(f"{qid}::s{sidx}", {})
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                i, j = rows[a], rows[b]
                margin = meta[i]["f1"] - meta[j]["f1"]
                if abs(margin) <= tau:
                    continue
                p, o = (i, j) if margin > 0 else (j, i)
                ci, co = sc.get(meta[p]["doc"]), sc.get(meta[o]["doc"])
                agree = (ci is not None and co is not None and ci > co)
                if conflicting_only and agree:
                    continue
                D.append(X[p] - X[o])
                ce_agree.append(agree)
    return np.array(D, dtype=np.float32), np.array(ce_agree)


def fit(D):
    from sklearn.linear_model import LogisticRegression
    Xs = np.vstack([D, -D])
    y = np.concatenate([np.ones(len(D)), np.zeros(len(D))])
    return LogisticRegression(max_iter=1000, fit_intercept=False).fit(Xs, y)


def boot(per_q, n_boot=10000, seed=7):
    qg = [sum(v) / len(v) for v in per_q.values()]
    rng = random.Random(seed)
    b = sorted(sum(qg[rng.randrange(len(qg))] for _ in qg) / len(qg)
               for _ in range(n_boot))
    return {"mean": sum(qg) / len(qg),
            "ci95": [b[int(.025 * n_boot)], b[int(.975 * n_boot)]],
            "n_questions": len(qg)}


def selection_eval(X, meta, idx, ce, w, seed=7):
    """Selectors over D_t, scored by the stored intervention outcome."""
    rng = random.Random(seed)
    names = ["base", "random", "lowest_ce", "ce", "specialist",
             "anti_oracle", "oracle"]
    abs_f1 = {n: defaultdict(list) for n in names}
    gain = {n: defaultdict(list) for n in names}
    top1 = {n: [] for n in names}
    for (qid, sidx), rows in idx.items():
        sc = ce.get(f"{qid}::s{sidx}", {})
        if not sc:
            continue
        cand = [i for i in rows if meta[i]["doc"] in sc]
        if not cand:
            continue
        best = max(meta[i]["f1"] for i in cand)
        pick = {
            "random": rng.choice(cand),
            "lowest_ce": min(cand, key=lambda i: sc[meta[i]["doc"]]),
            "ce": max(cand, key=lambda i: sc[meta[i]["doc"]]),
            "specialist": max(cand, key=lambda i: float(X[i] @ w)),
            "anti_oracle": min(cand, key=lambda i: meta[i]["f1"]),
            "oracle": max(cand, key=lambda i: meta[i]["f1"]),
        }
        ce_f1 = meta[pick["ce"]]["f1"]
        base_f1 = meta[cand[0]]["base_f1"]
        for n in names:
            y = base_f1 if n == "base" else meta[pick[n]]["f1"]
            abs_f1[n][qid].append(y)
            gain[n][qid].append(y - ce_f1)
            if n != "base":
                top1[n].append(float(abs(meta[pick[n]]["f1"] - best) < EPS))
    out = {}
    for n in names:
        out[n] = {"abs_f1": boot(abs_f1[n]), "gain_vs_ce": boot(gain[n])}
        if n != "base":
            out[n]["top1_any_max_acc"] = float(np.mean(top1[n]))
    return out


def pair_metrics(D, agree, w):
    pred = (D @ w) > 0
    n = len(D)
    follow = agree.mean()
    return {
        "n_pairs": int(n),
        "conflicting_share": float(1 - agree.mean()),
        "specialist_acc": float(pred.mean()),
        "follow_ce_acc": float(follow),
        "invert_ce_acc": float(1 - follow),
        "specialist_acc_on_aligned": float(pred[agree].mean()) if agree.any() else None,
        "specialist_acc_on_conflicting": float(pred[~agree].mean()) if (~agree).any() else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-tag", default="gate3")
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--out", default="results/rq2_reanalysis")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    Xtr, mtr = load_phi(args.train_tag)
    itr = state_index(mtr)
    cetr = load_ce(f"results/{args.train_tag}")
    Dall, agree_tr = build_pairs(Xtr, mtr, itr, cetr, args.tau)
    Dcon, _ = build_pairs(Xtr, mtr, itr, cetr, args.tau, conflicting_only=True)
    print(f"[train] all decided {len(Dall)} | conflicting-only {len(Dcon)}",
          flush=True)
    clf_all, clf_con = fit(Dall), fit(Dcon)
    w_all = clf_all.coef_[0]
    w_con = clf_con.coef_[0]

    R = {"train": {"tag": args.train_tag, "all_pairs": int(len(Dall)),
                   "conflicting_pairs": int(len(Dcon)),
                   "coef_unconditional": dict(zip(PHI_KEYS,
                                                  np.round(w_all, 3).tolist())),
                   "coef_conflict_conditioned": dict(zip(PHI_KEYS,
                                                         np.round(w_con, 3).tolist()))},
         "settings": {}}

    for name, tag, gate_dir in EVAL:
        X, meta = load_phi(tag)
        idx = state_index(meta)
        ce = load_ce(gate_dir)
        D, agree = build_pairs(X, meta, idx, ce, args.tau)
        R["settings"][name] = {
            "pairwise_all_decided": {
                "unconditional_specialist": pair_metrics(D, agree, w_all),
                "conflict_conditioned_specialist": pair_metrics(D, agree, w_con),
            },
            "selection": selection_eval(X, meta, idx, ce, w_all),
        }
        print(f"[{name}] done", flush=True)

    out = os.path.join(args.out, "report.json")
    with open(out, "w") as f:
        json.dump(R, f, indent=2)
    print(f"-> {out}")
    print("RQ2_REANALYSIS_DONE")


if __name__ == "__main__":
    main()
