"""Final attempt A, step 1 (R14): semantic delta features ψ(s,d) from E5
embeddings, aligned to the decided-pairs ordering of phi_cache.

Per (state, candidate):
  sem_novelty     1 − max cos(d, evidence_i)
  sem_red_mean    mean cos(d, evidence_i)
  state_align     cos(d, mean(evidence))
  last_rel        cos(d, last evidence doc)
  q0_cos, qt_cos  cos(d, question), cos(d, current query)
  unresolved_cov  cos(d, residual of question ⊥ evidence-mean)

Per state (gate-usable, direction-invariant):
  ev_q0_cov       cos(mean evidence, question)
  ev_coherence    mean pairwise cos among evidence
  n_ev            evidence count (scaled)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.data_utils import read_jsonl

PSI_KEYS = ["sem_novelty", "sem_red_mean", "state_align", "last_rel",
            "q0_cos", "qt_cos", "unresolved_cov"]
STATE_KEYS = ["ev_q0_cov", "ev_coherence", "n_ev"]


def main():
    from piu.retriever import BM25Retriever
    from piu.dense import E5DenseRetriever

    retriever = BM25Retriever("data/bm25_index")
    dense = E5DenseRetriever("data/e5_embeddings.fp16.npy", retriever.titles)
    emb = dense.emb  # (N, 768) float32 normalized

    dirs = sys.argv[1:] or ["results/gate3", "results/gate2"]
    for gate_dir in dirs:
        pairs = [p for p in read_jsonl(f"{gate_dir}/pairs_tau0.1.jsonl")
                 if not p["tie"]]
        with open(f"{gate_dir}/questions.json") as f:
            qmeta = json.load(f)
        base_pre = {r["question_id"]: r for r in read_jsonl(qmeta["pre_path"])}

        # encode q0 / qt per state on GPU
        state_ids, texts_q0, texts_qt = [], [], []
        seen_states = {}
        for p in pairs:
            key = (p["question_id"], p["state_idx"])
            if key in seen_states:
                continue
            rec = base_pre[key[0]]
            qt = None
            for e in rec["events"][: key[1] + 1]:
                if e["type"] == "info" and e.get("query"):
                    qt = e["query"]
            seen_states[key] = len(state_ids)
            state_ids.append(key)
            texts_q0.append(rec["question"])
            texts_qt.append(qt or rec["question"])
        q0_emb = dense._encode(texts_q0)
        qt_emb = dense._encode(texts_qt)

        # per-state evidence embedding structures
        state_feat, ev_cache = {}, {}
        for key, si in seen_states.items():
            rec = base_pre[key[0]]
            titles = []
            for e in rec["events"][: key[1] + 1]:
                if e["type"] == "info":
                    titles.extend(e.get("titles", []))
            idxs = [retriever.title2idx[t] for t in titles
                    if t in retriever.title2idx]
            E = emb[idxs] if idxs else np.zeros((0, emb.shape[1]), np.float32)
            ev_mean = E.mean(0) if len(E) else np.zeros(emb.shape[1], np.float32)
            nrm = np.linalg.norm(ev_mean)
            ev_mean = ev_mean / nrm if nrm > 0 else ev_mean
            coh = 0.0
            if len(E) > 1:
                sims = E @ E.T
                coh = (sims.sum() - len(E)) / (len(E) * (len(E) - 1))
            q0 = q0_emb[si]
            resid = q0 - (q0 @ ev_mean) * ev_mean
            rn = np.linalg.norm(resid)
            resid = resid / rn if rn > 0 else resid
            ev_cache[key] = (E, ev_mean, resid, si)
            state_feat[key] = [float(ev_mean @ q0), float(coh),
                               min(len(E) / 12.0, 1.5)]

        def psi(key, title):
            idx = retriever.title2idx.get(title)
            if idx is None:
                return None
            v = emb[idx]
            E, ev_mean, resid, si = ev_cache[key]
            sims = E @ v if len(E) else np.array([0.0])
            return [
                1.0 - float(sims.max()),
                float(sims.mean()),
                float(ev_mean @ v),
                float(E[-1] @ v) if len(E) else 0.0,
                float(q0_emb[si] @ v),
                float(qt_emb[si] @ v),
                float(resid @ v),
            ]

        P, O, S, keep = [], [], [], []
        for p in pairs:
            key = (p["question_id"], p["state_idx"])
            pref_t = p["doc_i"] if p["pref"] == "i" else p["doc_j"]
            oth_t = p["doc_j"] if p["pref"] == "i" else p["doc_i"]
            vp, vo = psi(key, pref_t), psi(key, oth_t)
            # phi_cache skipped pairs whose docs are missing from the corpus;
            # replicate that skip so orderings stay aligned
            from_retr = (retriever.doc_by_title(pref_t) is not None
                         and retriever.doc_by_title(oth_t) is not None)
            if not from_retr:
                continue
            P.append(vp)
            O.append(vo)
            S.append(state_feat[key])
        np.savez(f"{gate_dir}/semantic_cache.npz",
                 psi_pref=np.array(P, np.float32),
                 psi_other=np.array(O, np.float32),
                 state=np.array(S, np.float32))
        print(f"{gate_dir}: {len(P)} pairs, ψ{len(PSI_KEYS)} + state{len(STATE_KEYS)}")
    print("SEMANTIC_DONE")


if __name__ == "__main__":
    main()
