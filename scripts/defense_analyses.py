"""R21 ②③④ + permutation null — CPU reanalyses of existing logs.

② question-clustered H2: per-question pair accuracy → question bootstrap CI,
   question permutation p, balanced accuracy, label prevalence.
③ candidate-pool robustness: standard-retrieval-only oracle + leave-one-
   family-out H1 gain.
④ opportunity effect-size decomposition: P(max ΔF1 > 0 / ≥0.05 / ≥0.10).
①-aux permutation null: within-state shuffle of utilities → null max-gain.
"""
import json
import os
import sys
from collections import defaultdict

import joblib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from piu.data_utils import read_jsonl

EPS = 1e-9
DIRS = {"lockbox": "results/lockbox", "wiki2": "results/wiki2",
        "qwen14": "results/xagent"}
STANDARD = {"bm25_top", "dense_top", "ce_top"}
FAMILIES = ["dense_top", "ce_top", "mid_rel", "novelty", "redundancy",
            "entity_bridge", "random"]  # bm25_top은 CE 비교 유지 위해 제외 안 함


def load(dirp, retriever):
    from phi_baseline import load_phi_pairs
    utils = read_jsonl(f"{dirp}/utilities.jsonl")
    ce_rank = {r["state_id"]: {t: i for i, t in enumerate(r["ce_ranking"])}
               for r in read_jsonl(f"{dirp}/ce_rankings.jsonl")}
    X, meta = load_phi_pairs(dirp, retriever)
    return utils, ce_rank, X, meta


def clustered_h2(X, meta, ed):
    pred = ed.decision_function(X) > 0
    dis = [i for i, m in enumerate(meta) if not m["ce_agrees"]]
    by_q = defaultdict(list)
    for i in dis:
        by_q[meta[i]["question_id"]].append(bool(pred[i]))
    qaccs = {q: np.mean(v) for q, v in by_q.items()}
    qs = list(qaccs)
    rng = np.random.default_rng(7)
    boots = [np.mean([qaccs[qs[j]] for j in rng.integers(0, len(qs), len(qs))])
             for _ in range(10000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # 질문 단위 permutation: 각 질문 내 pref 방향을 무작위 뒤집기 → acc 분포
    perm = []
    for _ in range(2000):
        flips = rng.random(len(qs)) < 0.5
        # 방향 뒤집으면 해당 질문 acc → 1-acc
        perm.append(np.mean([1 - qaccs[q] if f else qaccs[q]
                             for q, f in zip(qs, flips)]))
    obs = np.mean(list(qaccs.values()))
    p_perm = float(np.mean([p >= obs for p in perm]))
    # balanced accuracy over conflicting pairs: pref=i(=specialist가 i 선호가 정답)
    # 쌍 구성상 정답은 항상 'preferred first' 방향 → balanced = acc (한 클래스).
    # 대안: aligned+conflicting 전체에서 조건별 acc 병기
    return {"n_questions": len(qs), "mean_question_acc": float(obs),
            "bootstrap_ci95": [float(lo), float(hi)],
            "question_permutation_p": p_perm,
            "pair_acc": float(np.mean([pred[i] for i in dis])),
            "n_pairs": len(dis),
            "prevalence_conflicting": len(dis) / len(meta)}


def oracle_gains(utils, ce_rank, keep_types=None, drop_type=None):
    by_state = defaultdict(list)
    for u in utils:
        t = u["ctype"]
        if keep_types is not None and t not in keep_types:
            continue
        if drop_type is not None and t == drop_type:
            continue
        by_state[(u["question_id"], u["state_idx"])].append(u)
    per_q = defaultdict(list)
    for (qid, sidx), rows in by_state.items():
        order = ce_rank.get(f"{qid}::s{sidx}")
        if not order:
            continue
        ce_pick = min(rows, key=lambda r: order.get(r["doc"], 10**6))
        pi_pick = max(rows, key=lambda r: r["U_task_raw_f1"])
        per_q[qid].append(pi_pick["arm_d"]["f1"] - ce_pick["arm_d"]["f1"])
    qg = [np.mean(v) for v in per_q.values()]
    rng = np.random.default_rng(7)
    boots = [np.mean([qg[j] for j in rng.integers(0, len(qg), len(qg))])
             for _ in range(5000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": float(np.mean(qg)), "ci95": [float(lo), float(hi)],
            "n_questions": len(qg)}


def opportunity_decomp(utils):
    by_state = defaultdict(list)
    for u in utils:
        by_state[(u["question_id"], u["state_idx"])].append(u["U_task_raw_f1"])
    best = np.array([max(v) for v in by_state.values()])
    return {"n_states": len(best),
            "P(>0)": float((best > EPS).mean()),
            "P(>=0.05)": float((best >= 0.05).mean()),
            "P(>=0.10)": float((best >= 0.10).mean()),
            "P(>=0.25)": float((best >= 0.25).mean())}


def permutation_null(utils, ce_rank, n_perm=1000):
    """Exchangeability null (auxiliary): oracle−CE 관측 gain을,
    'CE 선택이 pool 내 무작위 후보와 다르지 않다'는 null의
    oracle−random gain 분포와 비교. (winner's curse의 본 방어는 cross-fit)"""
    by_state = defaultdict(list)
    for u in utils:
        by_state[(u["question_id"], u["state_idx"])].append(u)
    rng = np.random.default_rng(7)
    obs = oracle_gains(utils, ce_rank)["mean"]
    nulls = []
    for _ in range(n_perm):
        per_q = defaultdict(list)
        for (qid, sidx), rows in by_state.items():
            if f"{qid}::s{sidx}" not in ce_rank:
                continue
            f1s = [r["arm_d"]["f1"] for r in rows]
            per_q[qid].append(max(f1s) - f1s[rng.integers(len(f1s))])
        nulls.append(np.mean([np.mean(v) for v in per_q.values()]))
    nulls = np.array(nulls)
    return {"observed_oracle_minus_ce": obs,
            "null_oracle_minus_random_mean": float(nulls.mean()),
            "null_ci95": [float(np.percentile(nulls, 2.5)),
                           float(np.percentile(nulls, 97.5))],
            "note": "obs < null mean이면 CE가 무작위보다 나음 (pool 내 CE 신호 존재)"}


def main():
    from piu.retriever import BM25Retriever
    retriever = BM25Retriever("data/bm25_index")
    ed = joblib.load("results/frozen/specialist_conflicting.joblib")
    R = {}
    for name, dirp in DIRS.items():
        utils, ce_rank, X, meta = load(dirp, retriever)
        R[name] = {
            "clustered_H2": clustered_h2(X, meta, ed),
            "oracle_full": oracle_gains(utils, ce_rank),
            "oracle_standard_only": oracle_gains(utils, ce_rank,
                                                 keep_types=STANDARD),
            "leave_one_out": {f: oracle_gains(utils, ce_rank, drop_type=f)["mean"]
                              for f in FAMILIES},
            "opportunity_decomposition": opportunity_decomp(utils),
        }
        print(f"[{name}] done", flush=True)
    R["permutation_null_lockbox"] = permutation_null(
        *load(DIRS["lockbox"], retriever)[:2])
    with open("results/defense_analyses.json", "w") as f:
        json.dump(R, f, indent=2)
    print(json.dumps(R, indent=2))
    print("DEFENSE_DONE")


if __name__ == "__main__":
    main()
