"""Two distinct 'positive state' rates, computed side by side (reviewer (d)).

Opp-base  at least one candidate beats the PRE-INTERVENTION trajectory
          max_d U_raw(d) >= tau            (what the code has always computed)
Opp-CE    at least one candidate beats the CE-SELECTED candidate
          H(s_t) = Y(d_PIU) - Y(d_CE) >= tau   (the RQ3 state-level label)

These are different quantities and must never share a name. Both are reported
at tau in {>0, >=0.25} so the effect-size row is unambiguous.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.data_utils import read_jsonl

EPS = 1e-9
EVAL = [("HotpotQA dev", "results/gate2"), ("HotpotQA lockbox", "results/lockbox"),
        ("2WikiMultiHopQA", "results/wiki2"), ("Qwen2.5-14B", "results/xagent"),
        ("MuSiQue", "results/musique")]


def main():
    R = {}
    for name, gate_dir in EVAL:
        ce = {r["state_id"]: r["scores"]
              for r in read_jsonl(os.path.join(gate_dir,
                                               "candidate_ce_scores.jsonl"))}
        by = defaultdict(list)
        for u in read_jsonl(os.path.join(gate_dir, "utilities.jsonl")):
            by[(u["question_id"], u["state_idx"])].append(u)
        base_gain, ce_gain, strata = [], [], []
        for (qid, sidx), rr in by.items():
            sc = ce.get(f"{qid}::s{sidx}")
            if not sc:
                continue
            cand = [r for r in rr if r["doc"] in sc]
            if len(cand) < 2:
                continue
            base_gain.append(max(r["U_task_raw_f1"] for r in cand))
            d_ce = max(cand, key=lambda r: sc[r["doc"]])
            ce_gain.append(max(r["arm_d"]["f1"] for r in cand)
                           - d_ce["arm_d"]["f1"])
            strata.append(cand[0].get("stratum") or "unknown")
        b, c = np.array(base_gain), np.array(ce_gain)
        entry = {"n_states": int(len(b)),
                 "Opp_base": {"gt0": float((b > EPS).mean()),
                              "ge025": float((b >= 0.25).mean())},
                 "Opp_CE": {"gt0": float((c > EPS).mean()),
                            "ge025": float((c >= 0.25).mean())},
                 "by_stratum": {}}
        st = np.array(strata)
        for s in sorted(set(strata)):
            m = st == s
            entry["by_stratum"][s] = {
                "n_states": int(m.sum()),
                "Opp_base_gt0": float((b[m] > EPS).mean()),
                "Opp_CE_gt0": float((c[m] > EPS).mean())}
        R[name] = entry
        print(f"[{name}] Opp-base {entry['Opp_base']['gt0']:.3f} | "
              f"Opp-CE {entry['Opp_CE']['gt0']:.3f}", flush=True)
    os.makedirs("results/opportunity_definitions", exist_ok=True)
    out = "results/opportunity_definitions/report.json"
    with open(out, "w") as f:
        json.dump(R, f, indent=2)
    print(f"-> {out}")
    print("OPPORTUNITY_DEFS_DONE")


if __name__ == "__main__":
    main()
