"""Analysis for the decoding-stochasticity sensitivity check.

Recomputes the H1 endpoints (opportunity rate, Oracle-CE headroom) from sampled
rollouts and compares them against the greedy confirmatory values on the
identical state set. Also reports greedy-vs-sampled pairwise label agreement.

Post-hoc exploratory: no new model, no new preregistered test.
"""
import argparse
import glob
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.data_utils import read_jsonl

EPS = 1e-9
ORDER = ["bm25_top", "dense_top", "ce_top", "mid_rel", "novelty",
         "redundancy", "entity_bridge", "random"]


def bootstrap_ci(per_q, n_boot=10000, seed=7):
    """Question-level paired bootstrap, matching lockbox_eval.py."""
    qg = [sum(v) / len(v) for v in per_q.values()]
    rng = random.Random(seed)
    boots = sorted(sum(qg[rng.randrange(len(qg))] for _ in qg) / len(qg)
                   for _ in range(n_boot))
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]
    return {"n_questions": len(qg), "mean": sum(qg) / len(qg),
            "ci95": [lo, hi], "excludes_zero": lo > 0}


def endpoints(f1_base, f1_doc, ce_rank, states):
    """f1_base: state -> F1; f1_doc: state -> {doc: (F1, ctype)}."""
    opp = 0
    per_q = defaultdict(list)
    for st in states:
        docs = f1_doc[st]
        best = max(v[0] for v in docs.values())
        if best - f1_base[st] > EPS:
            opp += 1
        order = ce_rank.get(f"{st[0]}::s{st[1]}")
        if not order:
            continue
        # same tie rule as the confirmatory pipeline: first in family order
        items = sorted(docs.items(),
                       key=lambda kv: ORDER.index(kv[1][1])
                       if kv[1][1] in ORDER else len(ORDER))
        ce_doc = min(items, key=lambda kv: order.get(kv[0], 10 ** 6))
        pi_doc = max(items, key=lambda kv: kv[1][0])
        per_q[st[0]].append(pi_doc[1][0] - ce_doc[1][0])
    return {"opportunity": {"decisive": opp, "states": len(states),
                            "rate": opp / len(states)},
            "oracle_vs_ce": bootstrap_ci(per_q)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-dir", default="results/lockbox")
    ap.add_argument("--sampled-dir", default="results/decoding_sensitivity")
    ap.add_argument("--tau", type=float, default=0.1)
    args = ap.parse_args()

    ce_rank = {r["state_id"]: {t: i for i, t in enumerate(r["ce_ranking"])}
               for r in read_jsonl(os.path.join(args.gate_dir,
                                                "ce_rankings.jsonl"))}
    # ---- greedy reference ----
    utils = read_jsonl(os.path.join(args.gate_dir, "utilities.jsonl"))
    g_base, g_doc = {}, defaultdict(dict)
    for u in utils:
        st = (u["question_id"], u["state_idx"])
        g_base[st] = u["base"]["f1"]
        g_doc[st][u["doc"]] = (u["arm_d"]["f1"], u["ctype"])

    # ---- sampled replicates ----
    reps = sorted(glob.glob(os.path.join(args.sampled_dir,
                                         "arm_results_T*_r*.jsonl")))
    assert reps, f"no replicate files in {args.sampled_dir}"
    per_rep = []
    acc_base, acc_doc = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for path in reps:
        rb, rd = {}, defaultdict(dict)
        for r in read_jsonl(path):
            st = (r["question_id"], r["state_idx"])
            if r["arm_type"] == "base":
                rb[st] = r["outcome"]["f1"]
                acc_base[st].append(r["outcome"]["f1"])
            else:
                rd[st][r["doc"]] = (r["outcome"]["f1"], r["ctype"])
                acc_doc[st][r["doc"]].append(r["outcome"]["f1"])
        per_rep.append((os.path.basename(path), rb, rd))

    states = sorted(set(acc_base) & set(g_base)
                    & {s for s in acc_doc if acc_doc[s]})
    ctype = {st: {d: g_doc[st][d][1] for d in acc_doc[st] if d in g_doc[st]}
             for st in states}
    m_base = {st: sum(acc_base[st]) / len(acc_base[st]) for st in states}
    m_doc = {st: {d: (sum(v) / len(v), ctype[st][d])
                  for d, v in acc_doc[st].items() if d in ctype[st]}
             for st in states}

    R = {"n_states_compared": len(states), "n_replicates": len(reps),
         "greedy": endpoints(g_base, g_doc, ce_rank, states),
         "sampled_mean_over_replicates": endpoints(m_base, m_doc, ce_rank,
                                                   states)}
    R["sampled_per_replicate"] = {}
    for name, rb, rd in per_rep:
        sub = [st for st in states if st in rb and rd.get(st)]
        rd2 = {st: {d: v for d, v in rd[st].items() if d in ctype[st]}
               for st in sub}
        R["sampled_per_replicate"][name] = endpoints(rb, rd2, ce_rank, sub)

    # ---- pairwise label agreement (greedy-decided pairs) ----
    pairs = [p for p in read_jsonl(os.path.join(args.gate_dir,
                                                f"pairs_tau{args.tau}.jsonl"))
             if not p["tie"]]
    n = zero = nonzero_same = decided_both = same_decided = 0
    for p in pairs:
        st = (p["question_id"], p["state_idx"])
        if st not in m_doc:
            continue
        di, dj = m_doc[st].get(p["doc_i"]), m_doc[st].get(p["doc_j"])
        if di is None or dj is None:
            continue
        n += 1
        margin = di[0] - dj[0]
        want = p["pref"]
        if margin == 0:
            zero += 1
            continue
        got_sign = "i" if margin > 0 else "j"
        nonzero_same += int(got_sign == want)
        if abs(margin) > args.tau:
            decided_both += 1
            same_decided += int(got_sign == want)
    nz = n - zero
    R["label_agreement"] = {
        "n_greedy_decided_pairs_evaluated": n,
        "collapsed_to_exact_tie_under_sampling": zero / n if n else None,
        "sign_agreement_among_nonzero": nonzero_same / nz if nz else None,
        "still_decided_at_tau_under_sampling": decided_both / n if n else None,
        "sign_agreement_among_still_decided":
            same_decided / decided_both if decided_both else None,
        "note": "greedy-decided pairs only; exact ties under sampling are "
                "indeterminate rather than contradictory",
    }

    out = os.path.join(args.sampled_dir, "DECODING_SENSITIVITY_REPORT.json")
    with open(out, "w") as f:
        json.dump(R, f, indent=2)
    print(json.dumps(R, indent=2))
    print(f"-> {out}")
    print("DECODING_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
