"""RQ3 as state-level relevance-failure identification (reviewer W1/W2).

Condition:  Z(s_t) = 1[ H(s_t) > 0 ],  H(s_t) = Y(d_PIU) - Y(d_CE)
i.e. "does this state contain a candidate that beats the CE-selected one?"
This is not defined by disagreement with CE ordering, so no trivial inversion
solves it, and the unit of analysis is the state rather than the pair.

The gate sees only pre-intervention observables aggregated over D_t.
Routing is then evaluated on realized downstream F1 with the stored outcomes:

  CE only              always take d_CE
  gate -> specialist   take the specialist pick where the gate fires
  oracle state         take the specialist pick where Z=1 truly holds
  oracle               take argmax_d Y(d)             (upper bound)

The gap between 'oracle state' and 'oracle' isolates ranking difficulty from
identification difficulty.
"""
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.data_utils import read_jsonl
from piu.state_delta import PHI_KEYS

EPS = 1e-9
AGG = ["mean", "max", "min", "std"]
EVAL = [("HotpotQA dev", "gate2", "results/gate2"),
        ("HotpotQA lockbox", "lockbox", "results/lockbox"),
        ("2WikiMultiHopQA", "wiki2", "results/wiki2"),
        ("Qwen2.5-14B", "xagent", "results/xagent"),
        ("MuSiQue", "musique", "results/musique")]


def load(tag, gate_dir):
    z = np.load(f"results/phi_candidates/{tag}.npz", allow_pickle=True)
    X, meta = z["X"], [json.loads(m) for m in z["meta"]]
    ce = {r["state_id"]: r["scores"]
          for r in read_jsonl(os.path.join(gate_dir, "candidate_ce_scores.jsonl"))}
    probe = {}
    pf = os.path.join(gate_dir, "gate_features.jsonl")
    if os.path.exists(pf):
        for r in read_jsonl(pf):
            qid, sidx = r["state_id"].split("::s")
            probe[(qid, int(sidx))] = [r["probe_action_answer"], r["probe_conf"],
                                       r["probe_len"] / 96.0]
    by = defaultdict(list)
    for i, m in enumerate(meta):
        by[(m["question_id"], m["state_idx"])].append(i)
    return X, meta, ce, probe, by


def state_rows(X, meta, ce, probe, by, w_spec):
    """Per state: gate features, labels, and the realized F1 of each selector."""
    feats, names, out = [], None, []
    for (qid, sidx), rows in by.items():
        sc = ce.get(f"{qid}::s{sidx}", {})
        cand = [i for i in rows if meta[i]["doc"] in sc]
        if len(cand) < 2:
            continue
        cev = np.array([sc[meta[i]["doc"]] for i in cand], np.float32)
        P = X[cand]
        srt = np.sort(cev)[::-1]
        f, nm = [], []
        f += [srt[0], cev.mean(), cev.std(), srt[0] - srt[1]]
        nm += ["ce_max", "ce_mean", "ce_std", "ce_margin"]
        for k, key in enumerate(PHI_KEYS):
            col = P[:, k]
            f += [col.mean(), col.max(), col.min(), col.std()]
            nm += [f"{key}_{a}" for a in AGG]
        f += [float(len(cand))]
        nm += ["n_candidates"]
        pr = probe.get((qid, sidx))
        f += (pr if pr else [0.0, 0.0, 0.0])
        nm += ["probe_action_answer", "probe_conf", "probe_len"]
        feats.append(f)
        names = nm
        ce_i = max(cand, key=lambda i: sc[meta[i]["doc"]])
        or_i = max(cand, key=lambda i: meta[i]["f1"])
        sp_i = max(cand, key=lambda i: float(X[i] @ w_spec))
        H = meta[or_i]["f1"] - meta[ce_i]["f1"]
        out.append({"question_id": qid, "state_idx": sidx, "H": H,
                    "z": int(H > EPS), "z025": int(H >= 0.25),
                    "y_ce": meta[ce_i]["f1"], "y_oracle": meta[or_i]["f1"],
                    "y_spec": meta[sp_i]["f1"],
                    "has_probe": bool(pr)})
    return np.array(feats, np.float32), names, out


def boot(per_q, n_boot=10000, seed=7):
    qg = [sum(v) / len(v) for v in per_q.values()]
    rng = random.Random(seed)
    b = sorted(sum(qg[rng.randrange(len(qg))] for _ in qg) / len(qg)
               for _ in range(n_boot))
    return {"mean": sum(qg) / len(qg),
            "ci95": [b[int(.025 * n_boot)], b[int(.975 * n_boot)]]}


def routing(rows, fire):
    """fire[i] = gate decision for row i. Returns question-level F1 summaries."""
    sel = {"ce": defaultdict(list), "gate": defaultdict(list),
           "oracle_state": defaultdict(list), "oracle": defaultdict(list),
           "always_specialist": defaultdict(list)}
    for r, f in zip(rows, fire):
        q = r["question_id"]
        sel["ce"][q].append(r["y_ce"])
        sel["gate"][q].append(r["y_spec"] if f else r["y_ce"])
        sel["oracle_state"][q].append(r["y_spec"] if r["z"] else r["y_ce"])
        sel["always_specialist"][q].append(r["y_spec"])
        sel["oracle"][q].append(r["y_oracle"])
    base = boot(sel["ce"])["mean"]
    out = {}
    for k, v in sel.items():
        s = boot(v)
        gq = defaultdict(list)
        for q in v:
            for a, b in zip(v[q], sel["ce"][q]):
                gq[q].append(a - b)
        out[k] = {"abs_f1": s, "gain_vs_ce": boot(gq)}
    out["_ce_abs"] = base
    return out


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    # unconditional specialist weights (RQ2 reanalysis)
    w = np.array(list(json.load(open(
        "results/rq2_reanalysis/report.json"))["train"]["coef_unconditional"].values()
    ), np.float32) if False else None
    rep = json.load(open("results/rq2_reanalysis/report.json"))
    w = np.array([rep["train"]["coef_unconditional"][k] for k in PHI_KEYS],
                 np.float32)

    Xtr, mtr, cetr, prtr, bytr = load("gate3", "results/gate3")
    Ftr, names, rows_tr = state_rows(Xtr, mtr, cetr, prtr, bytr, w)
    ztr = np.array([r["z"] for r in rows_tr])
    print(f"[train] {len(Ftr)} states, Z rate {ztr.mean():.3f}", flush=True)

    sc = StandardScaler().fit(Ftr)
    gbm = HistGradientBoostingClassifier(max_iter=400, max_depth=4,
                                         random_state=42).fit(Ftr, ztr)
    lr = LogisticRegression(max_iter=2000).fit(sc.transform(Ftr), ztr)

    R = {"feature_names": names, "train_states": int(len(Ftr)),
         "train_z_rate": float(ztr.mean()), "settings": {}}
    for name, tag, gate_dir in EVAL:
        X, meta, ce, probe, by = load(tag, gate_dir)
        F, _, rows = state_rows(X, meta, ce, probe, by, w)
        z = np.array([r["z"] for r in rows])
        z025 = np.array([r["z025"] for r in rows])
        p_gbm = gbm.predict_proba(F)[:, 1]
        p_lr = lr.predict_proba(sc.transform(F))[:, 1]
        p_ens = (p_gbm + p_lr) / 2
        aucs = {k: float(roc_auc_score(z, p))
                for k, p in (("gbm", p_gbm), ("logistic", p_lr), ("ens", p_ens))}
        auc025 = {k: float(roc_auc_score(z025, p))
                  for k, p in (("gbm", p_gbm), ("logistic", p_lr), ("ens", p_ens))}
        # route where the gate is most confident, at the true prevalence
        k = int(z.sum())
        thr = np.sort(p_ens)[::-1][k - 1] if k > 0 else 1.1
        fire = p_ens >= thr
        route = routing(rows, fire)
        prec = float(z[fire].mean()) if fire.any() else None
        R["settings"][name] = {
            "n_states": len(rows), "z_rate": float(z.mean()),
            "z025_rate": float(z025.mean()),
            "has_probe_features": bool(rows[0]["has_probe"]),
            "gate_auc_Z": aucs, "gate_auc_Z025": auc025,
            "gate_precision_at_prevalence": prec,
            "routing": route,
        }
        print(f"[{name}] AUC(ens) {aucs['ens']:.3f} | Z {z.mean():.3f}", flush=True)

    os.makedirs("results/rq3_state_gate", exist_ok=True)
    out = "results/rq3_state_gate/report.json"
    with open(out, "w") as f:
        json.dump(R, f, indent=2)
    print(f"-> {out}")
    print("RQ3_STATE_GATE_DONE")


if __name__ == "__main__":
    main()
