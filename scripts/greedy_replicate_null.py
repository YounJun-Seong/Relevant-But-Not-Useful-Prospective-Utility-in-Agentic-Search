"""Greedy analogue of the matched-noise / split-replicate controls.

The lockbox reliability audit re-ran 20% of states three times under the frozen
GREEDY policy. Those replays give a direct measurement of how much of the
Oracle-CE gain survives when selection and evaluation use different rollouts of
the same policy -- the same question the sampled-decoding split answers, but at
the noise level of the policy the paper actually uses.

Sampled-decoding replicates are restricted to the identical states so the two
noise regimes are compared on the same data.
"""
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.data_utils import read_jsonl, f1_score

LOCK = "results/lockbox"


def boot(per_q, n_boot=10000, seed=7):
    qg = [float(np.mean(v)) for v in per_q.values()]
    rng = random.Random(seed)
    b = sorted(sum(qg[rng.randrange(len(qg))] for _ in qg) / len(qg)
               for _ in range(n_boot))
    return {"mean": sum(qg) / len(qg),
            "ci95": [b[int(.025 * n_boot)], b[int(.975 * n_boot)]],
            "n_questions": len(qg)}


def qmean(per_q):
    return float(np.mean([np.mean(v) for v in per_q.values()]))


def analyze(states, tag, n_perm=1000, seed=7):
    rng = np.random.default_rng(seed)
    out = {"tag": tag, "n_states": len(states)}
    obs = defaultdict(list)
    for s in states:
        m = s["M"].mean(axis=1)
        obs[s["qid"]].append(float(m.max() - m[s["ce_i"]]))
    out["observed_select_and_eval_same"] = boot(obs)
    nrep = states[0]["M"].shape[1]
    splits = []
    for sel_r in range(nrep):
        ev = [r for r in range(nrep) if r != sel_r]
        per_q = defaultdict(list)
        for s in states:
            pick = int(np.argmax(s["M"][:, sel_r]))
            e = s["M"][:, ev].mean(axis=1)
            per_q[s["qid"]].append(float(e[pick] - e[s["ce_i"]]))
        r = boot(per_q)
        out[f"split_select_r{sel_r+1}"] = r
        splits.append(r["mean"])
    out["split_mean_over_folds"] = float(np.mean(splits))
    means = []
    for _ in range(n_perm):
        per_q = defaultdict(list)
        for s in states:
            M = s["M"]
            dev = (M - M.mean(axis=1, keepdims=True)).ravel()
            sim = M.mean() + rng.choice(dev, size=M.shape, replace=True)
            m = sim.mean(axis=1)
            per_q[s["qid"]].append(float(m.max() - m[s["ce_i"]]))
        means.append(qmean(per_q))
    means = np.array(means)
    out["matched_noise_null"] = {
        "null_mean": float(means.mean()),
        "null_ci95": [float(np.percentile(means, 2.5)),
                      float(np.percentile(means, 97.5))],
        "p_null_ge_observed": float(
            (means >= out["observed_select_and_eval_same"]["mean"]).mean())}
    # within-candidate replay dispersion, for context
    sd = [float(s["M"].std(axis=1).mean()) for s in states]
    out["mean_within_candidate_sd"] = float(np.mean(sd))
    return out


def main():
    ce = {r["state_id"]: r["scores"]
          for r in read_jsonl(f"{LOCK}/candidate_ce_scores.jsonl")}
    with open(f"{LOCK}/questions.json") as f:
        qmeta = json.load(f)
    gold = {r["question_id"]: r["answer_gold"]
            for r in read_jsonl(qmeta["pre_path"])}
    plan = {p["arm_id"]: p for p in read_jsonl(f"{LOCK}/arm_plan.jsonl")}
    G = defaultdict(lambda: defaultdict(list))
    for r in read_jsonl(f"{LOCK}/arm_results.jsonl"):
        if r.get("arm_type") not in ("I2_add", "I2_add_rep"):
            continue
        p = plan.get(r["arm_id"])
        if p is None:
            continue
        f1 = f1_score(r["answer"] or "", gold[r["question_id"]])
        G[(r["question_id"], r["state_idx"])][p["doc"]].append(f1)

    greedy = []
    for (qid, sidx), dd in G.items():
        sc = ce.get(f"{qid}::s{sidx}")
        if not sc:
            continue
        docs = [d for d in dd if d in sc and len(dd[d]) >= 3]
        if len(docs) < 2:
            continue
        greedy.append({"qid": qid, "sidx": sidx,
                       "M": np.array([dd[d][:3] for d in docs]),
                       "ce_i": int(np.argmax([sc[d] for d in docs]))})
    keys = {(s["qid"], s["sidx"]) for s in greedy}
    print(f"[greedy] {len(greedy)} audited states", flush=True)

    S = defaultdict(lambda: defaultdict(list))
    for p in sorted(os.listdir("results/decoding_sensitivity")):
        if not (p.startswith("arm_results_T") and p.endswith(".jsonl")):
            continue
        for r in read_jsonl(os.path.join("results/decoding_sensitivity", p)):
            if r["arm_type"] == "I2_add":
                S[(r["question_id"], r["state_idx"])][r["doc"]].append(
                    r["outcome"]["f1"])
    sampled = []
    for k in keys:
        dd = S.get(k, {})
        sc = ce.get(f"{k[0]}::s{k[1]}")
        docs = [d for d in dd if d in sc and len(dd[d]) >= 3]
        if len(docs) < 2:
            continue
        sampled.append({"qid": k[0], "sidx": k[1],
                        "M": np.array([dd[d][:3] for d in docs]),
                        "ce_i": int(np.argmax([sc[d] for d in docs]))})
    print(f"[sampled] {len(sampled)} matched states", flush=True)

    R = {"greedy_replays": analyze(greedy, "greedy (3 replays, T=0)"),
         "sampled_same_states": analyze(sampled, "sampled T=0.7 (3 replicates)")}
    os.makedirs("results/placebo_oracle", exist_ok=True)
    out = "results/placebo_oracle/greedy_vs_sampled_replicates.json"
    with open(out, "w") as f:
        json.dump(R, f, indent=2)
    print(json.dumps(R, indent=2))
    print(f"-> {out}")
    print("GREEDY_REPLICATE_DONE")


if __name__ == "__main__":
    main()
