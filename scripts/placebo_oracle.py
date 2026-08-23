"""Placebo oracle: is Oracle-CE just max-over-8 of noise? (reviewer W1/RQ1)

Four controls, each answering a different version of the objection.

N1 within-state identity permutation
   Outcomes are permuted among the candidates of a state. The maximum is
   invariant, so this asks only "is d_CE better than an arbitrary candidate?".

N2 i.i.d. redraw null
   Each candidate's outcome is redrawn with replacement from the state's own
   outcome multiset, breaking the candidate-outcome link while keeping the
   state's outcome distribution. Gives the spurious gain from taking a maximum
   over eight exchangeable draws.

N3 matched-noise simulation  (needs sampled replicates)
   All candidates in a state are given the SAME expected outcome (the state
   mean) and are perturbed by the empirically observed within-candidate replay
   deviations. This is the literal max-of-noise magnitude at the measured noise
   level.

N4 split-replicate oracle    (needs sampled replicates)
   The candidate is SELECTED on one replicate and EVALUATED on the held-out
   replicates. Winner's curse cannot survive this split.
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

EPS = 1e-9
EVAL = [("HotpotQA dev", "results/gate2"), ("HotpotQA lockbox", "results/lockbox"),
        ("2WikiMultiHopQA", "results/wiki2"), ("Qwen2.5-14B", "results/xagent"),
        ("MuSiQue", "results/musique")]


def qmean(per_q):
    return float(np.mean([np.mean(v) for v in per_q.values()]))


def boot(per_q, n_boot=10000, seed=7):
    qg = [float(np.mean(v)) for v in per_q.values()]
    rng = random.Random(seed)
    b = sorted(sum(qg[rng.randrange(len(qg))] for _ in qg) / len(qg)
               for _ in range(n_boot))
    return {"mean": sum(qg) / len(qg),
            "ci95": [b[int(.025 * n_boot)], b[int(.975 * n_boot)]]}


def load_states(gate_dir):
    ce = {r["state_id"]: r["scores"]
          for r in read_jsonl(os.path.join(gate_dir, "candidate_ce_scores.jsonl"))}
    by = defaultdict(list)
    for u in read_jsonl(os.path.join(gate_dir, "utilities.jsonl")):
        by[(u["question_id"], u["state_idx"])].append(u)
    states = []
    for (qid, sidx), rr in by.items():
        sc = ce.get(f"{qid}::s{sidx}")
        if not sc:
            continue
        docs = [r["doc"] for r in rr if r["doc"] in sc]
        if len(docs) < 2:
            continue
        states.append({
            "qid": qid, "sidx": sidx, "docs": docs,
            "y": np.array([next(r["arm_d"]["f1"] for r in rr if r["doc"] == d)
                           for d in docs]),
            "ce_i": int(np.argmax([sc[d] for d in docs])),
        })
    return states


def observed_and_nulls(states, n_perm=1000, seed=7):
    rng = np.random.default_rng(seed)
    obs = defaultdict(list)
    for s in states:
        obs[s["qid"]].append(float(s["y"].max() - s["y"][s["ce_i"]]))
    out = {"observed_oracle_minus_ce": boot(obs)}
    for tag, mode in (("N1_identity_permutation", "perm"),
                      ("N2_iid_redraw", "iid")):
        means = []
        for _ in range(n_perm):
            per_q = defaultdict(list)
            for s in states:
                y = s["y"]
                yn = rng.permutation(y) if mode == "perm" else \
                    y[rng.integers(0, len(y), len(y))]
                per_q[s["qid"]].append(float(yn.max() - yn[s["ce_i"]]))
            means.append(qmean(per_q))
        means = np.array(means)
        out[tag] = {"null_mean": float(means.mean()),
                    "null_ci95": [float(np.percentile(means, 2.5)),
                                  float(np.percentile(means, 97.5))],
                    "p_observed_ge_null": float(
                        (means >= out["observed_oracle_minus_ce"]["mean"]).mean())}
    return out


def replicate_analyses(gate_dir, sampled_dir, n_perm=1000, seed=7):
    """N3 matched-noise simulation and N4 split-replicate oracle."""
    ce = {r["state_id"]: r["scores"]
          for r in read_jsonl(os.path.join(gate_dir, "candidate_ce_scores.jsonl"))}
    reps = sorted([p for p in os.listdir(sampled_dir)
                   if p.startswith("arm_results_T") and p.endswith(".jsonl")])
    Y = defaultdict(lambda: defaultdict(list))
    for p in reps:
        for r in read_jsonl(os.path.join(sampled_dir, p)):
            if r["arm_type"] == "I2_add":
                Y[(r["question_id"], r["state_idx"])][r["doc"]].append(
                    r["outcome"]["f1"])
    states = []
    for (qid, sidx), dd in Y.items():
        sc = ce.get(f"{qid}::s{sidx}")
        if not sc:
            continue
        docs = [d for d in dd if d in sc and len(dd[d]) == len(reps)]
        if len(docs) < 2:
            continue
        M = np.array([dd[d] for d in docs])          # (n_cand, n_rep)
        states.append({"qid": qid, "M": M,
                       "ce_i": int(np.argmax([sc[d] for d in docs]))})
    R = {"n_states": len(states), "n_replicates": len(reps)}

    # observed (mean over replicates)
    obs = defaultdict(list)
    for s in states:
        m = s["M"].mean(axis=1)
        obs[s["qid"]].append(float(m.max() - m[s["ce_i"]]))
    R["observed_sampled_oracle_minus_ce"] = boot(obs)

    # N4 split-replicate: select on replicate 0, evaluate on the rest
    for sel_r in range(len(reps)):
        ev = [r for r in range(len(reps)) if r != sel_r]
        per_q = defaultdict(list)
        for s in states:
            pick = int(np.argmax(s["M"][:, sel_r]))
            e = s["M"][:, ev].mean(axis=1)
            per_q[s["qid"]].append(float(e[pick] - e[s["ce_i"]]))
        R[f"N4_split_replicate_select_r{sel_r+1}"] = boot(per_q)

    # N3 matched-noise: all candidates share the state mean, perturbed by the
    # observed within-candidate deviations
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_perm):
        per_q = defaultdict(list)
        for s in states:
            M = s["M"]
            dev = M - M.mean(axis=1, keepdims=True)
            flat = dev.ravel()
            sim = M.mean() + rng.choice(flat, size=M.shape, replace=True)
            m = sim.mean(axis=1)
            per_q[s["qid"]].append(float(m.max() - m[s["ce_i"]]))
        means.append(qmean(per_q))
    means = np.array(means)
    R["N3_matched_noise_null"] = {
        "null_mean": float(means.mean()),
        "null_ci95": [float(np.percentile(means, 2.5)),
                      float(np.percentile(means, 97.5))],
        "p_observed_ge_null": float(
            (means >= R["observed_sampled_oracle_minus_ce"]["mean"]).mean()),
    }
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--out", default="results/placebo_oracle")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    R = {}
    for name, gate_dir in EVAL:
        R[name] = observed_and_nulls(load_states(gate_dir), args.n_perm)
        print(f"[{name}] nulls done", flush=True)
    R["HotpotQA lockbox"]["replicate_based"] = replicate_analyses(
        "results/lockbox", "results/decoding_sensitivity", args.n_perm)
    print("[lockbox] replicate analyses done", flush=True)
    out = os.path.join(args.out, "report.json")
    with open(out, "w") as f:
        json.dump(R, f, indent=2)
    print(f"-> {out}")
    print("PLACEBO_ORACLE_DONE")


if __name__ == "__main__":
    main()
