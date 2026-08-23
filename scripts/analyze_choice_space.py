"""Downstream comparison on the EXPANDED choice space.

Combines stored D_t outcomes with the new full-pool-argmax arms so that every
selector is evaluated on the candidate it would actually choose from the full
pool C_t. Also recomputes (A) retention and (B) hit@1 over D_t from the
exact-config full-pool scores.

Selectors on the expanded set:
  ce              full-pool CE argmax        (always in D_t by construction)
  bge_qt_full     full-pool BGE (q_t) argmax (stored arm if retained, else new)
  bge_state_full  full-pool BGE-state argmax
  oracle_Dt       best over D_t               (paper's oracle)
  oracle_expanded best over D_t U {new arms}
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.data_utils import read_jsonl, norm_title

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
            "ci95": [b[int(.025 * n_boot)], b[int(.975 * n_boot)]],
            "n_questions": len(qg)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fullpool-dir", default="results/selector_relevance_exact")
    ap.add_argument("--arms-dir", default="results/choice_space")
    args = ap.parse_args()
    R = {}
    for name, tag, gate_dir in EVAL:
        fp_path = os.path.join(args.fullpool_dir, f"fullpool_{tag}.jsonl")
        if not os.path.exists(fp_path):
            print(f"[{name}] no exact full-pool scores yet, skipped")
            continue
        with open(os.path.join(gate_dir, "questions.json")) as f:
            qmeta = json.load(f)
        gold_of = {r["question_id"]: {norm_title(t) for t in r["gold_titles"]}
                   for r in read_jsonl(qmeta["pre_path"])}
        full = {r["state_id"]: r for r in read_jsonl(fp_path)}
        ce_dt = {r["state_id"]: r["scores"]
                 for r in read_jsonl(os.path.join(gate_dir,
                                                  "candidate_ce_scores.jsonl"))}
        bge_dt = {r["state_id"]: r["scores"]
                  for r in read_jsonl(f"results/strong_reranker/{tag}.jsonl")}
        y_dt = defaultdict(dict)
        for u in read_jsonl(os.path.join(gate_dir, "utilities.jsonl")):
            y_dt[f"{u['question_id']}::s{u['state_idx']}"][u["doc"]] = u["arm_d"]["f1"]
        new_path = os.path.join(args.arms_dir, f"fullpool_arms_{tag}.jsonl")
        y_new = defaultdict(dict)
        if os.path.exists(new_path):
            for r in read_jsonl(new_path):
                y_new[f"{r['question_id']}::s{r['state_idx']}"][r["doc"]] = r["outcome"]["f1"]

        ret = {"ce": [], "bge_qt": [], "bge_state": []}
        hit = {k: defaultdict(list) for k in ("ce", "bge_qt", "bge_state")}
        absf = {k: defaultdict(list) for k in
                ("ce", "bge_qt_full", "bge_state_full", "oracle_Dt",
                 "oracle_expanded")}
        n_states = n_gold = n_missing = 0
        for sid, fp in full.items():
            sc, st = ce_dt.get(sid), bge_dt.get(sid)
            Dt = y_dt.get(sid)
            if not sc or not st or not Dt:
                continue
            qid = sid.split("::")[0]
            T = fp["titles"]
            top = {k: T[int(np.argmax(fp[k]))]
                   for k in ("ce", "bge_qt", "bge_state")}
            for k in ret:
                ret[k].append(float(top[k] in Dt))
            # (B) hit@1 over D_t
            gold = gold_of[qid]
            present = {d for d in Dt if norm_title(d) in gold}
            if present:
                n_gold += 1
                picks = {"ce": max(Dt, key=lambda d: sc.get(d, -1e9)),
                         "bge_qt": max(Dt, key=lambda d: st.get(d, {}).get("strong_qt", -1e9)),
                         "bge_state": max(Dt, key=lambda d: st.get(d, {}).get("strong_state", -1e9))}
                for k, d in picks.items():
                    hit[k][qid].append(float(d in present))
            # downstream on the expanded choice space
            y_all = dict(Dt)
            y_all.update(y_new.get(sid, {}))
            y_ce = Dt[max(Dt, key=lambda d: sc.get(d, -1e9))]
            ok = True
            vals = {"ce": y_ce}
            for k, key in (("bge_qt", "bge_qt_full"), ("bge_state", "bge_state_full")):
                if top[k] not in y_all:
                    ok = False
                    break
                vals[key] = y_all[top[k]]
            if not ok:
                n_missing += 1
                continue
            n_states += 1
            vals["oracle_Dt"] = max(Dt.values())
            vals["oracle_expanded"] = max(y_all.values())
            for k, v in vals.items():
                absf[k][qid].append(v)
        if n_states == 0:
            print(f"[{name}] new arms missing for all states; run expand_choice_space.py")
            continue
        entry = {
            "n_states_complete": n_states, "n_states_missing_new_arm": n_missing,
            "n_states_with_gold_in_Dt": n_gold,
            "A_fullpool_argmax_retained_in_Dt": {k: float(np.mean(v)) for k, v in ret.items()},
            "B_hit_at_1_over_Dt": {k: boot(v) for k, v in hit.items()},
            "B_paired_vs_ce": {k: boot({q: [a - b for a, b in zip(hit[k][q], hit["ce"][q])]
                                        for q in hit[k]}) for k in ("bge_qt", "bge_state")},
            "C_abs_f1_expanded": {k: boot(v) for k, v in absf.items()},
            "C_paired_vs_ce": {k: boot({q: [a - b for a, b in zip(absf[k][q], absf["ce"][q])]
                                        for q in absf[k]})
                               for k in ("bge_qt_full", "bge_state_full",
                                         "oracle_Dt", "oracle_expanded")},
        }
        R[name] = entry
        c = entry["C_paired_vs_ce"]
        print(f"[{name}] ret CE {entry['A_fullpool_argmax_retained_in_Dt']['ce']:.3f} "
              f"BGE-st {entry['A_fullpool_argmax_retained_in_Dt']['bge_state']:.3f} | "
              f"BGE-st(full)−CE {c['bge_state_full']['mean']:+.4f} | "
              f"oracle_Dt {c['oracle_Dt']['mean']:+.4f} → expanded {c['oracle_expanded']['mean']:+.4f}",
              flush=True)
    os.makedirs("results/choice_space", exist_ok=True)
    out = "results/choice_space/report_expanded.json"
    with open(out, "w") as f:
        json.dump(R, f, indent=2)
    print(f"-> {out}")
    print("CHOICE_SPACE_EXPANDED_DONE")


if __name__ == "__main__":
    main()
