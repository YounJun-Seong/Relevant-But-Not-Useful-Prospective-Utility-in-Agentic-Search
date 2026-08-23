"""A lightweight prospective continuation scorer (reviewer: 'you measure
headroom but cannot use it').

Setting: the agent rolls out the CE top-k candidates (rollouts already exist)
and must then choose one WITHOUT any gold label. This differs from the RQ3
gate, which predicts before intervention whether headroom exists; the scorer
judges, after a budgeted lookahead, which continuation is better.

Inputs per continuation are strictly observable (gold-free):
  candidate CE score (z-scored within the state), behaviour of the
  continuation (extra searches, termination, token/turn counts), lexical
  support of the produced answer in the injected document and in
  post-intervention evidence, agreement with the pre-intervention base answer,
  and whether the answer is the modal answer among the k continuations.
Downstream F1 is used ONLY as the training label (pairwise, tau=0.1), never as
a feature.

Train: HotpotQA training split (gate3). Frozen, then applied to the five
evaluation settings. Selection = argmax score among CE top-k; realized F1 is
looked up from the stored outcome.
"""
import json
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.data_utils import read_jsonl, normalize_answer

EPS = 1e-9
TAU = 0.1
EVAL = [("HotpotQA dev", "gate2", "results/gate2"),
        ("HotpotQA lockbox", "lockbox", "results/lockbox"),
        ("2WikiMultiHopQA", "wiki2", "results/wiki2"),
        ("Qwen2.5-14B", "xagent", "results/xagent"),
        ("MuSiQue", "musique", "results/musique")]
FEAT = ["ce_z", "n_extra_search", "answered", "answered_immediately",
        "gen_tokens", "turns_delta", "ans_len", "ans_empty", "ans_in_doc",
        "ans_in_future_ev", "same_as_base", "is_modal"]


def toks(s):
    return set(normalize_answer(s or "").split())


def boot(per_q, n_boot=10000, seed=7):
    qg = [float(np.mean(v)) for v in per_q.values()]
    rng = random.Random(seed)
    b = sorted(sum(qg[rng.randrange(len(qg))] for _ in qg) / len(qg)
               for _ in range(n_boot))
    return {"mean": sum(qg) / len(qg),
            "ci95": [b[int(.025 * n_boot)], b[int(.975 * n_boot)]],
            "n_questions": len(qg)}


def build_states(gate_dir, retriever, k):
    """Per state: the CE top-k continuations with observable features + f1."""
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
    states = []
    for (qid, sidx), rr in by.items():
        sc = sc_all.get(f"{qid}::s{sidx}")
        if not sc:
            continue
        cand = [r for r in rr if r["doc"] in sc]
        if len(cand) < 2:
            continue
        order = sorted(cand, key=lambda r: -sc[r["doc"]])
        top = order[:k]
        base_ans = normalize_answer(top[0]["base"]["answer"] or "")
        base_turns = top[0]["base"]["turns"]
        cev = np.array([sc[r["doc"]] for r in cand], np.float32)
        mu, sd = float(cev.mean()), float(cev.std() or 1.0)
        norm_answers = [normalize_answer(r["arm_d"]["answer"] or "")
                        for r in top]
        modal, modal_n = Counter(norm_answers).most_common(1)[0]
        rows = []
        for r, na in zip(top, norm_answers):
            a = arms.get((qid, sidx, r["doc"]))
            if a is None:
                rows = []
                break
            n_extra, fut_titles = 0, []
            for e in a["events"][sidx + 1:]:
                if e["type"] == "info" and not e.get("forced") and e.get("query"):
                    n_extra += 1
                    fut_titles.extend(e.get("titles", []))
            at = toks(na)
            doc = retriever.doc_by_title(r["doc"])
            dt = toks(doc["text"][:1500]) if doc else set()
            ft = set()
            for t in fut_titles[:6]:
                d2 = retriever.doc_by_title(t)
                if d2:
                    ft |= toks(d2["text"][:800])
            answered = float(r["arm_d"]["finish_reason"] == "answer")
            f = [
                (sc[r["doc"]] - mu) / sd,
                n_extra / 3.0,
                answered,
                float(answered and n_extra == 0),
                r["arm_d"]["tokens"] / 1000.0,
                (r["arm_d"]["turns"] - base_turns) / 3.0,
                len(at) / 10.0,
                float(len(at) == 0),
                (len(at & dt) / len(at)) if at else 0.0,
                (len(at & ft) / len(at)) if at else 0.0,
                float(na == base_ans),
                float(na == modal and modal_n > 1),
            ]
            rows.append({"doc": r["doc"], "x": np.array(f, np.float32),
                         "f1": r["arm_d"]["f1"],
                         "ans": r["arm_d"]["answer"]})
        if len(rows) == k:
            states.append({"qid": qid, "sidx": sidx, "rows": rows,
                           "f1_all_max": max(r["arm_d"]["f1"] for r in cand)})
    return states


def pairs_from(states):
    D = []
    for s in states:
        rows = s["rows"]
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                m = rows[a]["f1"] - rows[b]["f1"]
                if abs(m) <= TAU:
                    continue
                p, o = (a, b) if m > 0 else (b, a)
                D.append(rows[p]["x"] - rows[o]["x"])
    return np.array(D, np.float32)


def main():
    from sklearn.linear_model import LogisticRegression
    from piu.retriever import BM25Retriever
    retriever = BM25Retriever("data/bm25_index")
    K = 3

    tr = build_states("results/gate3", retriever, K)
    D = pairs_from(tr)
    sd = D.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = np.vstack([D, -D]) / sd
    y = np.concatenate([np.ones(len(D)), np.zeros(len(D))])
    clf = LogisticRegression(max_iter=2000, fit_intercept=False).fit(Xs, y)
    w = clf.coef_[0]
    print(f"[train] {len(tr)} states, {len(D)} decided pairs", flush=True)
    print("coef:", dict(zip(FEAT, np.round(w, 3).tolist())), flush=True)

    R = {"k": K, "train_states": len(tr), "train_pairs": int(len(D)),
         "coefficients": dict(zip(FEAT, [float(v) for v in w])),
         "settings": {}}
    for name, tag, gate_dir in EVAL:
        st = build_states(gate_dir, retriever, K)
        gain = {k2: defaultdict(list)
                for k2 in ("scorer", "oracle_k", "oracle_full")}
        abs_f1 = {s2: defaultdict(list)
                  for s2 in ("ce", "scorer", "oracle_k", "oracle_full")}
        top1 = []
        for s in st:
            rows = s["rows"]
            ce_f1 = rows[0]["f1"]                       # CE top-1
            pick = max(rows, key=lambda r: float((r["x"] / sd) @ w))
            best_k = max(r["f1"] for r in rows)
            gain["scorer"][s["qid"]].append(pick["f1"] - ce_f1)
            gain["oracle_k"][s["qid"]].append(best_k - ce_f1)
            gain["oracle_full"][s["qid"]].append(s["f1_all_max"] - ce_f1)
            abs_f1["ce"][s["qid"]].append(ce_f1)
            abs_f1["scorer"][s["qid"]].append(pick["f1"])
            abs_f1["oracle_k"][s["qid"]].append(best_k)
            abs_f1["oracle_full"][s["qid"]].append(s["f1_all_max"])
            top1.append(float(abs(pick["f1"] - best_k) < EPS))
        g_s = boot(gain["scorer"])
        g_k = boot(gain["oracle_k"])
        g_f = boot(gain["oracle_full"])
        R["settings"][name] = {
            "n_states": len(st),
            "gain_vs_ce": {"scorer": g_s, "oracle_at_k": g_k,
                           "oracle_full": g_f},
            "abs_f1": {k2: boot(v) for k2, v in abs_f1.items()},
            "top1_within_k_acc": float(np.mean(top1)),
            "recovered_fraction_of_oracle_k":
                g_s["mean"] / g_k["mean"] if g_k["mean"] > 0 else None,
            "recovered_fraction_of_full":
                g_s["mean"] / g_f["mean"] if g_f["mean"] > 0 else None,
        }
        print(f"[{name}] scorer gain {g_s['mean']:+.4f} "
              f"[{g_s['ci95'][0]:+.4f},{g_s['ci95'][1]:+.4f}] | "
              f"oracle@{K} {g_k['mean']:+.4f}", flush=True)

    os.makedirs("results/continuation_scorer", exist_ok=True)
    out = "results/continuation_scorer/report.json"
    with open(out, "w") as f:
        json.dump(R, f, indent=2)
    print(f"-> {out}")
    print("CONTINUATION_SCORER_DONE")


if __name__ == "__main__":
    main()
