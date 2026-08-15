"""Phase 1 predictor training + R11 judgement.

Train: Gate 3 decided pairs (train split questions), 90/10 question-level
train/val. Test: Gate 2 decided pairs (dev split questions) — zero question
overlap by construction (different HotpotQA splits).

Reports (R11): overall test pairwise accuracy, CE-disagreement subset accuracy
(killer metric), CE baseline accuracy, and CE+PIU combination (λ on val).
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piu.data_utils import read_jsonl
from piu.predictor import (PairwiseScorer, PairDataset, collate,
                           pairwise_loss, serialize_state, serialize_doc)


def load_gate(gate_dir, retriever, tau=0.1, snippet_chars=200, max_evidence=8,
              state_mode="full"):
    pairs = [p for p in read_jsonl(os.path.join(gate_dir, f"pairs_tau{tau}.jsonl"))
             if not p["tie"]]
    with open(os.path.join(gate_dir, "questions.json")) as f:
        qmeta = json.load(f)
    base_pre = {r["question_id"]: r for r in read_jsonl(qmeta["pre_path"])}
    ce_rank = {r["state_id"]: {t: i for i, t in enumerate(r["ce_ranking"])}
               for r in read_jsonl(os.path.join(gate_dir, "ce_rankings.jsonl"))}
    state_cache, rows = {}, []
    for p in pairs:
        qid, sidx = p["question_id"], p["state_idx"]
        key = (qid, sidx)
        if key not in state_cache:
            if state_mode == "question_only":
                state_cache[key] = "question: " + base_pre[qid]["question"]
            else:
                state_cache[key] = serialize_state(
                    base_pre[qid], sidx, retriever,
                    snippet_chars=snippet_chars, max_evidence=max_evidence)
        di = retriever.doc_by_title(p["doc_i"])
        dj = retriever.doc_by_title(p["doc_j"])
        if di is None or dj is None:
            continue
        pref_i = p["pref"] == "i"
        order = ce_rank.get(f"{qid}::s{sidx}", {})
        rank_pref = order.get(p["doc_i"] if pref_i else p["doc_j"], 10**6)
        rank_other = order.get(p["doc_j"] if pref_i else p["doc_i"], 10**6)
        rows.append({
            "question_id": qid, "state": state_cache[key],
            "doc_i": serialize_doc(*( (di["title"], di["text"]) if pref_i
                                      else (dj["title"], dj["text"]) )),
            "doc_j": serialize_doc(*( (dj["title"], dj["text"]) if pref_i
                                      else (di["title"], di["text"]) )),
            "weight": min(abs(p["margin"]), 1.0),
            "abs_margin": abs(p["margin"]),
            "ce_agrees": rank_pref < rank_other,  # CE prefers the PIU-preferred doc
            "ce_rank_pref": rank_pref, "ce_rank_other": rank_other,
        })
    return rows


@torch.no_grad()
def evaluate(model, tokenizer, rows, device, batch_size=32):
    model.eval()
    correct = ce_dis = ce_dis_correct = 0
    margins = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        enc_i, enc_j, _ = collate(batch, tokenizer)
        enc_i = {k: v.to(device) for k, v in enc_i.items()}
        enc_j = {k: v.to(device) for k, v in enc_j.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            si, sj = model.score(enc_i), model.score(enc_j)
        pred = (si > sj).float().cpu()
        margins.extend((si - sj).float().cpu().tolist())
        for b, ok in zip(batch, pred.tolist()):
            correct += int(ok)
            if not b["ce_agrees"]:
                ce_dis += 1
                ce_dis_correct += int(ok)
    n = len(rows)
    return {
        "n": n, "acc": correct / n,
        "ce_disagreement_n": ce_dis,
        "ce_disagreement_acc": (ce_dis_correct / ce_dis) if ce_dis else None,
        "margins": margins,
    }


def combined_eval(rows, margins, lam):
    """CE+PIU: preference by (1-λ)·sign-margin(PIU, z) + λ·(rank diff CE, z)."""
    import statistics
    m_sd = statistics.pstdev(margins) or 1.0
    correct = 0
    for b, m in zip(rows, margins):
        piu_score = m / m_sd
        ce_score = (b["ce_rank_other"] - b["ce_rank_pref"])  # >0 if CE agrees
        ce_score = max(min(ce_score, 20), -20) / 20.0
        s = (1 - lam) * piu_score + lam * ce_score
        correct += int(s > 0)
    return correct / len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="results/gate3")
    ap.add_argument("--test-dir", default="results/gate2")
    ap.add_argument("--index", default="data/bm25_index")
    ap.add_argument("--model", default="microsoft/deberta-v3-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--min-margin", type=float, default=0.0,
                    help="train only on pairs with |margin| >= this")
    ap.add_argument("--snippet-chars", type=int, default=200)
    ap.add_argument("--max-evidence", type=int, default=8)
    ap.add_argument("--state-mode", choices=["full", "question_only"],
                    default="full", help="question_only = f(q,d) ablation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/predictor")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"

    from piu.retriever import BM25Retriever
    from transformers import AutoTokenizer
    retriever = BM25Retriever(args.index)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    kw = dict(snippet_chars=args.snippet_chars, max_evidence=args.max_evidence,
              state_mode=args.state_mode)
    train_rows = load_gate(args.train_dir, retriever, **kw)
    test_rows = load_gate(args.test_dir, retriever, **kw)
    qids = sorted({r["question_id"] for r in train_rows})
    random.shuffle(qids)
    val_q = set(qids[: max(1, len(qids) // 10)])
    val_rows = [r for r in train_rows if r["question_id"] in val_q]
    tr_rows = [r for r in train_rows if r["question_id"] not in val_q
               and r["abs_margin"] >= args.min_margin]
    overlap = {r["question_id"] for r in train_rows} & \
              {r["question_id"] for r in test_rows}
    assert not overlap, f"question overlap train/test: {len(overlap)}"
    print(f"train pairs {len(tr_rows)} | val {len(val_rows)} | test {len(test_rows)} "
          f"| question overlap: 0", flush=True)

    model = PairwiseScorer(args.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    loader = torch.utils.data.DataLoader(
        PairDataset(tr_rows), batch_size=args.batch, shuffle=True,
        collate_fn=lambda b: (b,))
    best_val = -1
    for ep in range(args.epochs):
        model.train()
        total = 0.0
        for step, (batch,) in enumerate(loader):
            enc_i, enc_j, w = collate(batch, tokenizer)
            enc_i = {k: v.to(device) for k, v in enc_i.items()}
            enc_j = {k: v.to(device) for k, v in enc_j.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = pairwise_loss(model.score(enc_i), model.score(enc_j),
                                     w.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            if step % 50 == 0:
                print(f"ep{ep} step{step}/{len(loader)} loss {loss.item():.4f}",
                      flush=True)
        val = evaluate(model, tokenizer, val_rows, device)
        print(f"[ep{ep}] train_loss {total/len(loader):.4f} "
              f"val_acc {val['acc']:.3f} "
              f"val_ce_dis_acc {val['ce_disagreement_acc']}", flush=True)
        if val["acc"] > best_val:
            best_val = val["acc"]
            torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))

    model.load_state_dict(torch.load(os.path.join(args.out, "best.pt")))
    test = evaluate(model, tokenizer, test_rows, device)
    val = evaluate(model, tokenizer, val_rows, device)

    # CE baseline on test pairs + CE+PIU combination (λ from val)
    ce_acc = sum(r["ce_agrees"] for r in test_rows) / len(test_rows)
    lams = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    val_comb = {l: combined_eval(val_rows, val["margins"], l) for l in lams}
    best_lam = max(val_comb, key=val_comb.get)
    test_comb = combined_eval(test_rows, test["margins"], best_lam)

    report = {
        "train_pairs": len(tr_rows), "val_pairs": len(val_rows),
        "test_pairs": len(test_rows),
        "test_acc": test["acc"],
        "test_ce_disagreement_n": test["ce_disagreement_n"],
        "test_ce_disagreement_acc": test["ce_disagreement_acc"],
        "ce_baseline_acc": ce_acc,
        "best_lambda_on_val": best_lam,
        "test_combined_acc": test_comb,
        "val_acc": val["acc"],
        "model": args.model, "epochs": args.epochs, "lr": args.lr,
    }
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print("PREDICTOR_DONE")


if __name__ == "__main__":
    main()
