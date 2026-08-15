"""Phase 1 PIU predictor (R11): f(s, d) scorer trained with pairwise logistic
loss P(d_i ≻ d_j | s) = σ(f(s,d_i) − f(s,d_j)).

Scoring f(s,d) on a single (state, candidate) pair at inference time is what
makes this usable as a retrieval component; the pairwise loss only shapes
training. State serialization uses observable state only (q_0, q_t, query
history, seen evidence snippets) — no chain-of-thought (reasoning-agnostic,
master §3.4).
"""
import torch
import torch.nn as nn


def serialize_state(base_rec, state_idx, retriever=None, snippet_chars=200,
                    max_evidence=8):
    """Observable-state text for one frozen state."""
    q0 = base_rec["question"]
    queries, seen = [], []
    for e in base_rec["events"][: state_idx + 1]:
        if e["type"] == "info":
            if e.get("query"):
                queries.append(e["query"])
            seen.extend(e.get("titles", []))
    qt = queries[-1] if queries else q0
    ev_parts = []
    for t in seen[-max_evidence:]:
        snip = ""
        if retriever is not None:
            doc = retriever.doc_by_title(t)
            if doc:
                snip = ": " + doc["text"][:snippet_chars]
        ev_parts.append(t + snip)
    return (f"question: {q0} | current query: {qt} | "
            f"past queries: {' ; '.join(queries[:-1]) if len(queries) > 1 else '(none)'} | "
            f"evidence so far: {' || '.join(ev_parts) if ev_parts else '(none)'}")


def serialize_doc(title, text, max_chars=600):
    return f"{title}. {text[:max_chars]}"


class PairwiseScorer(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-base"):
        super().__init__()
        from transformers import AutoModelForSequenceClassification

        self.encoder = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1)

    def score(self, batch):
        return self.encoder(**batch).logits.squeeze(-1)


def pairwise_loss(score_i, score_j, weight=None):
    """BCE on σ(f_i − f_j) with target 1 (i preferred). Optional weights."""
    logits = score_i - score_j
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, torch.ones_like(logits), reduction="none")
    if weight is not None:
        loss = loss * weight
    return loss.mean()


class PairDataset(torch.utils.data.Dataset):
    """Each item = (state_text, doc_pref_text, doc_other_text, weight)."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(batch, tokenizer, max_len=512):
    states = [b["state"] for b in batch]
    enc_i = tokenizer(states, [b["doc_i"] for b in batch], padding=True,
                      truncation="only_first", max_length=max_len,
                      return_tensors="pt")
    enc_j = tokenizer(states, [b["doc_j"] for b in batch], padding=True,
                      truncation="only_first", max_length=max_len,
                      return_tensors="pt")
    w = torch.tensor([b.get("weight", 1.0) for b in batch], dtype=torch.float)
    return enc_i, enc_j, w
