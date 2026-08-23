"""Is the relevance advantage present on the exact downstream choice space?

Reviewer: BGE's strength is shown on the full pool C_t (~110 docs), but
interventions exist only for the retained set D_t (<=8). The CE full-pool
argmax is guaranteed to be in D_t; the BGE argmaxes are not.

A. Retention: Pr[argmax_{C_t} R(d) in D_t] for CE (sanity: must be 1.0),
   BGE (q_t) and BGE (state-aware).
B. Relevance quality ON D_t: supporting-document hit@1 of each selector over
   the retained set, with paired question-level bootstrap vs CE, on states
   whose D_t contains at least one annotated supporting document.
"""
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
    R = {}
    for name, tag, gate_dir in EVAL:
        with open(os.path.join(gate_dir, "questions.json")) as f:
            qmeta = json.load(f)
        gold_of = {r["question_id"]: {norm_title(t) for t in r["gold_titles"]}
                   for r in read_jsonl(qmeta["pre_path"])}
        ce_dt = {r["state_id"]: r["scores"]
                 for r in read_jsonl(os.path.join(gate_dir,
                                                  "candidate_ce_scores.jsonl"))}
        bge_dt = {r["state_id"]: r["scores"]
                  for r in read_jsonl(f"results/strong_reranker/{tag}.jsonl")}
        full = {r["state_id"]: r
                for r in read_jsonl(f"results/selector_relevance/fullpool_{tag}.jsonl")}
        by = defaultdict(list)
        for u in read_jsonl(os.path.join(gate_dir, "utilities.jsonl")):
            by[(u["question_id"], u["state_idx"])].append(u)

        ret = {"ce": [], "bge_qt": [], "bge_state": []}
        hit = {"random": defaultdict(list), "ce": defaultdict(list),
               "bge_qt": defaultdict(list), "bge_state": defaultdict(list)}
        rng = random.Random(7)
        n_states = n_gold = 0
        for (qid, sidx), rr in by.items():
            sid = f"{qid}::s{sidx}"
            sc, st, fp = ce_dt.get(sid), bge_dt.get(sid), full.get(sid)
            if not sc or not st or not fp:
                continue
            Dt = [r["doc"] for r in rr if r["doc"] in sc and r["doc"] in st]
            if len(Dt) < 2:
                continue
            n_states += 1
            Dset = set(Dt)
            # ---- A. retention of full-pool argmax
            T = fp["titles"]
            for key, col in (("ce", "ce"), ("bge_qt", "bge_qt"),
                             ("bge_state", "bge_state")):
                top = T[int(np.argmax(fp[col]))]
                ret[key].append(float(top in Dset))
            # ---- B. hit@1 over D_t
            gold = gold_of[qid]
            present = {d for d in Dt if norm_title(d) in gold}
            if not present:
                continue
            n_gold += 1
            picks = {
                "random": rng.choice(Dt),
                "ce": max(Dt, key=lambda d: sc[d]),
                "bge_qt": max(Dt, key=lambda d: st[d]["strong_qt"]),
                "bge_state": max(Dt, key=lambda d: st[d]["strong_state"]),
            }
            for k, d in picks.items():
                hit[k][qid].append(float(d in present))
        entry = {
            "n_states": n_states,
            "n_states_with_gold_in_Dt": n_gold,
            "gold_coverage_Dt": n_gold / max(n_states, 1),
            "A_fullpool_argmax_retained_in_Dt": {
                k: float(np.mean(v)) for k, v in ret.items()},
            "B_hit_at_1_over_Dt": {k: boot(v) for k, v in hit.items()},
            "B_paired_vs_ce": {
                k: boot({q: [a - b for a, b in zip(hit[k][q], hit["ce"][q])]
                         for q in hit[k]})
                for k in ("bge_qt", "bge_state")},
        }
        R[name] = entry
        a = entry["A_fullpool_argmax_retained_in_Dt"]
        h = entry["B_hit_at_1_over_Dt"]
        print(f"[{name}] retained: CE {a['ce']:.3f} BGE {a['bge_qt']:.3f} "
              f"BGE-state {a['bge_state']:.3f} | hit@1@Dt: CE {h['ce']['mean']:.3f} "
              f"BGE {h['bge_qt']['mean']:.3f} BGE-state {h['bge_state']['mean']:.3f}",
              flush=True)
    os.makedirs("results/choice_space", exist_ok=True)
    out = "results/choice_space/report.json"
    with open(out, "w") as f:
        json.dump(R, f, indent=2)
    print(f"-> {out}")
    print("CHOICE_SPACE_DONE")


if __name__ == "__main__":
    main()
