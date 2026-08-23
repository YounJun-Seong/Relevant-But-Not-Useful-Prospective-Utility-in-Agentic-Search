"""What does the oracle prefer when relevance fails? (reviewer B급)

On states with CE-relative headroom (Z=1), compare the oracle-winning
candidate d_PIU against the CE-selected candidate d_CE:

  (1) CE-rank of d_PIU within D_t (distribution, mean/median), plus the
      state-count complement of Fig.3c: in what fraction of failure states is
      d_PIU already within the CE top-2 / top-3;
  (2) paired feature differences phi(d_PIU) - phi(d_CE);
  (3) a within-state outcome-permutation NULL for (2): permuting outcomes
      while holding CE scores and features fixed preserves the mechanical
      part of the contrast (the winner is never the CE argmax, and CE rank
      correlates with query alignment), so a feature difference is a finding
      ONLY where the real value falls outside the null band.
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
EVAL = [("HotpotQA dev", "gate2", "results/gate2"),
        ("HotpotQA lockbox", "lockbox", "results/lockbox"),
        ("2WikiMultiHopQA", "wiki2", "results/wiki2"),
        ("Qwen2.5-14B", "xagent", "results/xagent"),
        ("MuSiQue", "musique", "results/musique")]


def boot(per_q, n_boot=10000, seed=7):
    qg = [float(np.mean(v)) for v in per_q.values()]
    rng = random.Random(seed)
    b = sorted(sum(qg[rng.randrange(len(qg))] for _ in qg) / len(qg)
               for _ in range(n_boot))
    return {"mean": sum(qg) / len(qg),
            "ci95": [b[int(.025 * n_boot)], b[int(.975 * n_boot)]]}


def load(tag, gate_dir):
    z = np.load(f"results/phi_candidates/{tag}.npz", allow_pickle=True)
    X, meta = z["X"], [json.loads(m) for m in z["meta"]]
    ce = {r["state_id"]: r["scores"]
          for r in read_jsonl(os.path.join(gate_dir,
                                           "candidate_ce_scores.jsonl"))}
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
        states.append({"qid": qid,
                       "idx": np.array(cand),
                       "y": np.array([meta[i]["f1"] for i in cand]),
                       "ce": np.array([sc[meta[i]["doc"]] for i in cand]),
                       "fam": [meta[i]["ctype"] for i in cand]})
    return X, states


def contrast(X, states, y_override=None):
    """Mean paired phi(winner)-phi(CE) over failure states, per question."""
    per_q = defaultdict(list)
    ranks, fams, in_top = [], [], {2: 0, 3: 0}
    n_fail = 0
    for k, s in enumerate(states):
        y = s["y"] if y_override is None else y_override[k]
        ce_i = int(np.argmax(s["ce"]))
        or_i = int(np.argmax(y))
        if y[or_i] - y[ce_i] <= EPS:
            continue
        n_fail += 1
        per_q[s["qid"]].append(X[s["idx"][or_i]] - X[s["idx"][ce_i]])
        rank = int((s["ce"] > s["ce"][or_i]).sum()) + 1
        ranks.append(rank)
        fams.append(s["fam"][or_i])
        for kk in in_top:
            in_top[kk] += int(rank <= kk)
    mean_diff = np.mean([np.mean(v, axis=0) for v in per_q.values()], axis=0)
    return mean_diff, per_q, ranks, fams, in_top, n_fail


def main():
    R = {}
    for name, tag, gate_dir in EVAL:
        X, states = load(tag, gate_dir)
        d_real, per_q, ranks, fams, in_top, n_fail = contrast(X, states)
        # question-level CI for each feature
        ci = {}
        for j, k in enumerate(PHI_KEYS):
            ci[k] = boot({q: [v[j] for v in vv] for q, vv in per_q.items()})
        # outcome-permutation null
        rng = np.random.default_rng(7)
        null = []
        for _ in range(300):
            yo = [rng.permutation(s["y"]) for s in states]
            null.append(contrast(X, states, y_override=yo)[0])
        null = np.array(null)
        ranks = np.array(ranks)
        from collections import Counter
        R[name] = {
            "n_failure_states": n_fail,
            "ce_rank_of_oracle_winner": {
                "mean": float(ranks.mean()), "median": float(np.median(ranks)),
                "dist": {str(k): int((ranks == k).sum()) for k in (2, 3, 4)}
                        | {"5+": int((ranks >= 5).sum())},
                "share_in_top2": in_top[2] / n_fail,
                "share_in_top3": in_top[3] / n_fail,
            },
            "family_of_oracle_winner": dict(Counter(fams).most_common()),
            "features": {
                k: {"real": float(d_real[j]),
                    "ci95": ci[k]["ci95"],
                    "null_mean": float(null[:, j].mean()),
                    "null_ci95": [float(np.percentile(null[:, j], 2.5)),
                                  float(np.percentile(null[:, j], 97.5))],
                    "outside_null": bool(
                        d_real[j] < np.percentile(null[:, j], 2.5)
                        or d_real[j] > np.percentile(null[:, j], 97.5))}
                for j, k in enumerate(PHI_KEYS)
            },
        }
        print(f"[{name}] {n_fail} failure states | median CE-rank "
              f"{np.median(ranks):.0f} | top3 {in_top[3]/n_fail:.1%}",
              flush=True)
    os.makedirs("results/oracle_winner", exist_ok=True)
    out = "results/oracle_winner/report.json"
    with open(out, "w") as f:
        json.dump(R, f, indent=2)
    print(f"-> {out}")
    print("ORACLE_WINNER_DONE")


if __name__ == "__main__":
    main()
