"""Analysis: myopic utility vs full PIU (HotpotQA lockbox)."""
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.data_utils import read_jsonl

EPS = 1e-9


def boot(per_q, n_boot=10000, seed=7):
    qg = [float(np.mean(v)) for v in per_q.values()]
    rng = random.Random(seed)
    b = sorted(sum(qg[rng.randrange(len(qg))] for _ in qg) / len(qg)
               for _ in range(n_boot))
    return {"mean": sum(qg) / len(qg),
            "ci95": [b[int(.025 * n_boot)], b[int(.975 * n_boot)]],
            "n_questions": len(qg)}


def main():
    from scipy.stats import spearmanr, pearsonr
    gate_dir, out_dir = "results/lockbox", "results/myopic"
    ce = {r["state_id"]: r["scores"]
          for r in read_jsonl(os.path.join(gate_dir, "candidate_ce_scores.jsonl"))}
    full = defaultdict(dict)
    full_base = {}
    for u in read_jsonl(os.path.join(gate_dir, "utilities.jsonl")):
        sid = f"{u['question_id']}::s{u['state_idx']}"
        full[sid][u["doc"]] = u["arm_d"]["f1"]
        full_base[sid] = u["base"]["f1"]
    my = defaultdict(dict)
    my_base = {}
    n_refused = n_total = 0
    for r in read_jsonl(os.path.join(out_dir, "myopic_arms.jsonl")):
        sid = f"{r['question_id']}::s{r['state_idx']}"
        if r["arm_type"] == "base":
            my_base[sid] = r["outcome"]["f1"]
        else:
            my[sid][r["doc"]] = r["outcome"]["f1"]
            n_total += 1
            n_refused += int(r["finish_reason"] != "answer")

    pairs_f, pairs_m = [], []
    agree_any, agree_on_opp, n_states, n_opp = 0, 0, 0, 0
    rhos = []
    h_full, h_my, h_cross, h_cross_ce = (defaultdict(list) for _ in range(4))
    disagree_opp = 0
    for sid, fd in full.items():
        sc, md = ce.get(sid), my.get(sid)
        if not sc or not md:
            continue
        docs = [d for d in fd if d in sc and d in md]
        if len(docs) < 2:
            continue
        n_states += 1
        qid = sid.split("::")[0]
        yf = np.array([fd[d] for d in docs])
        ym = np.array([md[d] for d in docs])
        pairs_f.extend(yf.tolist()); pairs_m.extend(ym.tolist())
        if yf.std() > 0 and ym.std() > 0:
            rhos.append(spearmanr(yf, ym).correlation)
        ce_i = int(np.argmax([sc[d] for d in docs]))
        fmax, mmax = yf.max(), ym.max()
        full_winners = {i for i in range(len(docs)) if abs(yf[i] - fmax) < EPS}
        my_arg = int(np.argmax(ym))
        my_winners = {i for i in range(len(docs)) if abs(ym[i] - mmax) < EPS}
        agree = bool(full_winners & my_winners)
        agree_any += agree
        is_opp = (fmax - yf[ce_i]) > EPS
        if is_opp:
            n_opp += 1
            agree_on_opp += agree
            disagree_opp += (not agree)
        # headroom under each condition, and cross-selection
        h_full[qid].append(fmax - yf[ce_i])
        h_my[qid].append(mmax - ym[ce_i])
        h_cross[qid].append(yf[my_arg] - yf[ce_i])          # select myopic, eval full
        h_cross_ce[qid].append(yf[ce_i])                     # reference
    pf, pm = np.array(pairs_f), np.array(pairs_m)
    R = {
        "n_states": n_states, "n_candidate_pairs": int(len(pf)),
        "myopic_refusal_rate(searched despite ban)": n_refused / max(n_total, 1),
        "utility_correlation": {
            "pearson": float(pearsonr(pf, pm)[0]),
            "spearman": float(spearmanr(pf, pm).correlation)},
        "within_state_spearman_mean": float(np.mean(rhos)),
        "within_state_spearman_n": len(rhos),
        "winner_top1_agreement_any_max": agree_any / n_states,
        "opportunity_states_full": n_opp,
        "winner_agreement_on_opportunity_states": agree_on_opp / max(n_opp, 1),
        "winner_disagreement_rate_on_opportunity_states": disagree_opp / max(n_opp, 1),
        "headroom_full_oracle_minus_ce": boot(h_full),
        "headroom_myopic_oracle_minus_ce(myopic outcomes)": boot(h_my),
        "full_headroom_captured_when_selected_by_myopic": boot(h_cross),
        "mean_base_f1_full": float(np.mean(list(full_base.values()))),
        "mean_base_f1_myopic": float(np.mean([my_base[s] for s in full_base if s in my_base])),
    }
    out = os.path.join(out_dir, "report.json")
    with open(out, "w") as f:
        json.dump(R, f, indent=2)
    print(json.dumps(R, indent=2))
    print("MYOPIC_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
