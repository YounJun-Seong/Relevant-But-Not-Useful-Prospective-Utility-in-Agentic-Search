"""Does the headroom survive when the intervention actually changes the search?

Reviewer W1: the mean continuation is 0.62 additional search steps, so a
sceptic can argue the measured headroom is a reader-context effect rather than
a trajectory effect. This splits the evaluated states by what the intervention
did to subsequent search behaviour and recomputes Oracle-CE in each subset.

Groups (defined from the two arms that enter the comparison, d_PIU and d_CE):
  no_search        neither arm issues a further search
  any_search       at least one arm issues a further search
  query_changed    at least one searches AND the first post-intervention query
                   differs between the two arms
  both_search      both arms search (strictest read of 'trajectory changed')

Question-level paired bootstrap within each subset, plus the base-relative and
CE-relative positive-state rates.
"""
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


def boot(per_q, n_boot=10000, seed=7):
    qg = [float(np.mean(v)) for v in per_q.values()]
    if len(qg) < 2:
        return None
    rng = random.Random(seed)
    b = sorted(sum(qg[rng.randrange(len(qg))] for _ in qg) / len(qg)
               for _ in range(n_boot))
    return {"mean": sum(qg) / len(qg),
            "ci95": [b[int(.025 * n_boot)], b[int(.975 * n_boot)]],
            "n_questions": len(qg)}


def post_search(events, state_idx):
    """(count of further non-forced searches, first query issued after state)."""
    n, q = 0, None
    for e in events[state_idx + 1:]:
        if e["type"] == "info" and not e.get("forced") and e.get("query"):
            n += 1
            if q is None:
                q = e["query"]
    return n, q


def main():
    R = {}
    for name, gate_dir in EVAL:
        ce_sc = {r["state_id"]: r["scores"]
                 for r in read_jsonl(os.path.join(gate_dir,
                                                  "candidate_ce_scores.jsonl"))}
        arms = {}
        for r in read_jsonl(os.path.join(gate_dir, "arm_results.jsonl")):
            if r.get("arm_type") == "I2_add":
                arms[(r["question_id"], r["state_idx"], r["doc"])] = r
        by = defaultdict(list)
        for u in read_jsonl(os.path.join(gate_dir, "utilities.jsonl")):
            by[(u["question_id"], u["state_idx"])].append(u)

        groups = {k: defaultdict(list) for k in
                  ("all", "no_search", "any_search", "query_changed",
                   "both_search")}
        opp = {k: [] for k in groups}
        steps_hist = defaultdict(int)
        for (qid, sidx), rr in by.items():
            sc = ce_sc.get(f"{qid}::s{sidx}")
            if not sc:
                continue
            cand = [r for r in rr if r["doc"] in sc]
            if len(cand) < 2:
                continue
            d_ce = max(cand, key=lambda r: sc[r["doc"]])
            d_pi = max(cand, key=lambda r: r["arm_d"]["f1"])
            a_ce = arms.get((qid, sidx, d_ce["doc"]))
            a_pi = arms.get((qid, sidx, d_pi["doc"]))
            if a_ce is None or a_pi is None:
                continue
            n_ce, q_ce = post_search(a_ce["events"], sidx)
            n_pi, q_pi = post_search(a_pi["events"], sidx)
            steps_hist[max(n_ce, n_pi)] += 1
            h = d_pi["arm_d"]["f1"] - d_ce["arm_d"]["f1"]
            base_pos = max(r["U_task_raw_f1"] for r in cand) > EPS
            tags = ["all"]
            if n_ce == 0 and n_pi == 0:
                tags.append("no_search")
            if n_ce >= 1 or n_pi >= 1:
                tags.append("any_search")
                if q_ce != q_pi:
                    tags.append("query_changed")
            if n_ce >= 1 and n_pi >= 1:
                tags.append("both_search")
            for t in tags:
                groups[t][qid].append(h)
                opp[t].append((float(h > EPS), float(base_pos)))
        entry = {"post_intervention_search_steps": dict(sorted(steps_hist.items()))}
        for k, g in groups.items():
            n_states = sum(len(v) for v in g.values())
            o = np.array(opp[k]) if opp[k] else np.zeros((0, 2))
            entry[k] = {"n_states": n_states,
                        "headroom": boot(g),
                        "Opp_CE": float(o[:, 0].mean()) if len(o) else None,
                        "Opp_base": float(o[:, 1].mean()) if len(o) else None}
        R[name] = entry
        a, b = entry["no_search"], entry["any_search"]
        print(f"[{name}] no_search {a['n_states']} h={a['headroom']['mean']:+.4f} | "
              f"any_search {b['n_states']} h={b['headroom']['mean']:+.4f}",
              flush=True)
    os.makedirs("results/w1_trajectory", exist_ok=True)
    out = "results/w1_trajectory/report.json"
    with open(out, "w") as f:
        json.dump(R, f, indent=2)
    print(f"-> {out}")
    print("W1_DECOMPOSITION_DONE")


if __name__ == "__main__":
    main()
