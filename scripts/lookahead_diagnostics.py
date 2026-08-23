"""Why cheap gold-free scoring of a lookahead fails.

(1) Consensus diagnostic: among opportunity states, is the useful candidate's
    continuation answer the MODAL answer across the candidate set? If the
    useful continuation is systematically a minority answer, self-consistency
    style aggregation is anti-correlated with utility by construction.
(2) An alternative gold-free rule -- prefer a continuation that terminates with
    an answer and issues no further search -- evaluated at budgets k=1..4.
"""
import json
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.data_utils import read_jsonl, f1_score, normalize_answer

EPS = 1e-9
EVAL = [("HotpotQA dev", "results/gate2"), ("HotpotQA lockbox", "results/lockbox"),
        ("2WikiMultiHopQA", "results/wiki2"), ("Qwen2.5-14B", "results/xagent"),
        ("MuSiQue", "results/musique")]


def boot(per_q, n_boot=10000, seed=7):
    qg = [float(np.mean(v)) for v in per_q.values()]
    rng = random.Random(seed)
    b = sorted(sum(qg[rng.randrange(len(qg))] for _ in qg) / len(qg)
               for _ in range(n_boot))
    return {"mean": sum(qg) / len(qg),
            "ci95": [b[int(.025 * n_boot)], b[int(.975 * n_boot)]]}


def main():
    R = {}
    for name, gate_dir in EVAL:
        with open(os.path.join(gate_dir, "questions.json")) as f:
            qmeta = json.load(f)
        gold = {r["question_id"]: r["answer_gold"]
                for r in read_jsonl(qmeta["pre_path"])}
        sc_all = {r["state_id"]: r["scores"]
                  for r in read_jsonl(os.path.join(gate_dir,
                                                   "candidate_ce_scores.jsonl"))}
        arms = {}
        for r in read_jsonl(os.path.join(gate_dir, "arm_results.jsonl")):
            if r.get("arm_type") == "I2_add":
                arms[(r["question_id"], r["state_idx"], r["doc"])] = r
        by = defaultdict(list)
        for u in read_jsonl(os.path.join(gate_dir, "utilities.jsonl")):
            by[(u["question_id"], u["state_idx"])].append(u)

        n_opp = or_modal = ce_modal = 0
        stop_rule = {k: defaultdict(list) for k in (1, 2, 3, 4)}
        for (qid, sidx), rr in by.items():
            sc = sc_all.get(f"{qid}::s{sidx}")
            if not sc:
                continue
            cand = [r for r in rr if r["doc"] in sc]
            if len(cand) < 2:
                continue
            d_ce = max(cand, key=lambda r: sc[r["doc"]])
            d_or = max(cand, key=lambda r: r["arm_d"]["f1"])
            norm = [normalize_answer(r["arm_d"]["answer"] or "") for r in cand]
            modal = Counter(norm).most_common(1)[0][0]
            if d_or["arm_d"]["f1"] - d_ce["arm_d"]["f1"] > EPS:
                n_opp += 1
                or_modal += int(normalize_answer(
                    d_or["arm_d"]["answer"] or "") == modal)
                ce_modal += int(normalize_answer(
                    d_ce["arm_d"]["answer"] or "") == modal)

            def key(r):
                a = arms.get((qid, sidx, r["doc"]))
                if a is None:
                    return (1, 99)
                n = sum(1 for e in a["events"][sidx + 1:]
                        if e["type"] == "info" and not e.get("forced")
                        and e.get("query"))
                return (0 if (a["finish_reason"] == "answer" and n == 0) else 1, n)
            order = sorted(cand, key=lambda r: -sc[r["doc"]])
            for k in stop_rule:
                pick = min(order[:k], key=key)
                stop_rule[k][qid].append(
                    f1_score(pick["arm_d"]["answer"] or "", gold[qid]))
        R[name] = {
            "n_opportunity_states": n_opp,
            "oracle_answer_is_modal": or_modal / max(n_opp, 1),
            "ce_answer_is_modal": ce_modal / max(n_opp, 1),
            "early_termination_rule": {str(k): boot(v)
                                       for k, v in stop_rule.items()},
        }
        print(f"[{name}] oracle-modal {R[name]['oracle_answer_is_modal']:.3f} "
              f"vs CE-modal {R[name]['ce_answer_is_modal']:.3f}", flush=True)
    os.makedirs("results/lookahead", exist_ok=True)
    out = "results/lookahead/diagnostics.json"
    with open(out, "w") as f:
        json.dump(R, f, indent=2)
    print(f"-> {out}")
    print("LOOKAHEAD_DIAGNOSTICS_DONE")


if __name__ == "__main__":
    main()
