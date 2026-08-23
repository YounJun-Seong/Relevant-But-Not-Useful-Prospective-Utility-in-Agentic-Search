"""Per-(state, candidate) phi cache.

The existing phi_cache.npz holds pair differences only. The state-level
reformulation needs phi(s_t, d) for every evaluated candidate so that a
candidate can be scored on its own and a selection can be made over D_t.
"""
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.data_utils import read_jsonl
from piu.state_delta import phi_vector, state_context, PHI_KEYS

DIRS = {"gate2": "results/gate2", "gate3": "results/gate3",
        "lockbox": "results/lockbox", "wiki2": "results/wiki2",
        "xagent": "results/xagent", "musique": "results/musique"}


def main():
    from piu.retriever import BM25Retriever
    retriever = BM25Retriever("data/bm25_index")
    os.makedirs("results/phi_candidates", exist_ok=True)
    for name, gate_dir in DIRS.items():
        out = f"results/phi_candidates/{name}.npz"
        if os.path.exists(out):
            print(f"[{name}] cached", flush=True)
            continue
        with open(os.path.join(gate_dir, "questions.json")) as f:
            qmeta = json.load(f)
        base_pre = {r["question_id"]: r for r in read_jsonl(qmeta["pre_path"])}
        by_state = defaultdict(list)
        for u in read_jsonl(os.path.join(gate_dir, "utilities.jsonl")):
            by_state[(u["question_id"], u["state_idx"])].append(u)
        X, meta, t0 = [], [], time.time()
        for si, ((qid, sidx), rr) in enumerate(by_state.items()):
            rec = base_pre[qid]
            qt, ev_texts, turn, nev = state_context(rec, sidx, retriever)
            q0 = rec["question"]
            for r in rr:
                d = retriever.doc_by_title(r["doc"])
                if d is None:
                    continue
                X.append(phi_vector(q0, qt, ev_texts, d["text"], turn, nev))
                meta.append({"question_id": qid, "state_idx": sidx,
                             "doc": r["doc"], "ctype": r["ctype"],
                             "U": r["U_task_raw_f1"],
                             "f1": r["arm_d"]["f1"],
                             "base_f1": r["base"]["f1"],
                             "stratum": r.get("stratum")})
            if si % 300 == 0:
                print(f"  {name} {si}/{len(by_state)} ({time.time()-t0:.0f}s)",
                      flush=True)
        np.savez(out, X=np.array(X, dtype=np.float32),
                 meta=np.array([json.dumps(m) for m in meta]),
                 keys=np.array(PHI_KEYS))
        print(f"[{name}] {len(X)} candidates in {time.time()-t0:.0f}s -> {out}",
              flush=True)
    print("PHI_CANDIDATES_DONE")


if __name__ == "__main__":
    main()
