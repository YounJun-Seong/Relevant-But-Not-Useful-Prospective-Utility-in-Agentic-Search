"""Learned neural gate for the state-level label (reviewer (c)).

The shallow gates aggregate hand-built features, so a weak AUC could reflect a
weak gate rather than a hard problem. This fine-tunes a transformer directly on
the state-level label

    Z(s_t) = 1[ H(s_t) > 0 ],

from the observable state text plus a canonical rendering of the candidate set.
Trained on the HotpotQA training-split states and evaluated, frozen, on the
five settings -- the same protocol as the shallow gates.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.data_utils import read_jsonl
from piu.predictor import serialize_state
from piu.state_delta import PHI_KEYS
from rq3_state_gate import EVAL, load, state_rows


def build_text(gate_dir, tag, retriever, rows, snippet=140, max_cand=8):
    """(state text, candidate-set text) per state, in canonical order."""
    with open(os.path.join(gate_dir, "questions.json")) as f:
        qmeta = json.load(f)
    base_pre = {r["question_id"]: r for r in read_jsonl(qmeta["pre_path"])}
    ce = {r["state_id"]: r["scores"]
          for r in read_jsonl(os.path.join(gate_dir, "candidate_ce_scores.jsonl"))}
    out = []
    for r in rows:
        sid = f"{r['question_id']}::s{r['state_idx']}"
        sc = ce.get(sid, {})
        a = serialize_state(base_pre[r["question_id"]], r["state_idx"],
                            retriever, snippet_chars=150, max_evidence=6)
        parts = []
        for t in sorted(sc)[:max_cand]:          # canonical (alphabetical)
            d = retriever.doc_by_title(t)
            parts.append(f"{t}: {d['text'][:snippet]}" if d else t)
        out.append((a, " [SEP] ".join(parts)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/rq3_neural_gate")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from sklearn.metrics import roc_auc_score, average_precision_score
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer)
    from piu.retriever import BM25Retriever

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    retriever = BM25Retriever("data/bm25_index")
    rep = json.load(open("results/rq2_reanalysis/report.json"))
    w = np.array([rep["train"]["coef_unconditional"][k] for k in PHI_KEYS],
                 np.float32)

    Xtr, mtr, cetr, prtr, bytr = load("gate3", "results/gate3")
    _, _, rows_tr = state_rows(Xtr, mtr, cetr, prtr, bytr, w)
    txt_tr = build_text("results/gate3", "gate3", retriever, rows_tr)
    ytr = np.array([r["z"] for r in rows_tr], dtype=np.float32)
    print(f"[train] {len(txt_tr)} states, prevalence {ytr.mean():.3f}",
          flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    net = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1, ignore_mismatched_sizes=True).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    pos_w = torch.tensor([(1 - ytr.mean()) / max(ytr.mean(), 1e-6)]).cuda()
    yt = torch.tensor(ytr)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    net.train()
    for ep in range(args.epochs):
        idx = rng.permutation(len(txt_tr))
        tot = 0.0
        for bi in range(0, len(idx), args.batch):
            b = idx[bi:bi + args.batch]
            enc = tok([txt_tr[i][0] for i in b], [txt_tr[i][1] for i in b],
                      padding=True, truncation=True, max_length=args.max_len,
                      return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = net(**enc).logits.squeeze(-1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits.float(), yt[b].cuda(), pos_weight=pos_w)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"  ep{ep} loss {tot/max(1,len(idx)//args.batch):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    net.eval()

    @torch.no_grad()
    def predict(txt):
        out = []
        for bi in range(0, len(txt), 64):
            blk = txt[bi:bi + 64]
            enc = tok([a for a, _ in blk], [b for _, b in blk], padding=True,
                      truncation=True, max_length=args.max_len,
                      return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.extend(torch.sigmoid(net(**enc).logits.squeeze(-1))
                           .float().cpu().tolist())
        return np.array(out)

    R = {"model": args.model, "epochs": args.epochs, "lr": args.lr,
         "train_states": len(txt_tr), "train_prevalence": float(ytr.mean()),
         "settings": {}}
    for name, tag, gate_dir in EVAL:
        X, meta, ce, probe, by = load(tag, gate_dir)
        _, _, rows = state_rows(X, meta, ce, probe, by, w)
        txt = build_text(gate_dir, tag, retriever, rows)
        y = np.array([r["z"] for r in rows])
        p = predict(txt)
        prev = float(y.mean())
        R["settings"][name] = {
            "n_states": len(rows), "prevalence": prev,
            "auc": float(roc_auc_score(y, p)),
            "average_precision": float(average_precision_score(y, p)),
            "ap_over_prevalence": float(average_precision_score(y, p) / prev),
        }
        np.save(os.path.join(args.out, f"pred_{tag}.npy"), p)
        print(f"[{name}] AUC {R['settings'][name]['auc']:.3f} "
              f"(prev {prev:.3f})", flush=True)
    tagm = args.model.split("/")[-1]
    with open(os.path.join(args.out, f"report_{tagm}.json"), "w") as f:
        json.dump(R, f, indent=2)
    print(json.dumps(R["settings"], indent=2))
    print("NEURAL_GATE_DONE")


if __name__ == "__main__":
    main()
