"""Dense retrieval leg (E5 memmap brute-force) and cross-encoder scorer.

The E5 embedding matrix lives in RAM (fp16 memmap → float32 chunks); queries
are encoded on GPU. Brute-force matmul over 5.2M×768 takes ~0.1s/query on
this many-core box, which is fine for Gate 1 volumes.
"""
import numpy as np
import torch
import torch.nn.functional as F


class E5DenseRetriever:
    def __init__(self, emb_path, titles, model_name="intfloat/e5-base-v2",
                 device="cuda", chunk=1_000_000):
        from transformers import AutoModel, AutoTokenizer

        # load fully into RAM as float32 once (~16GB, box has 1.7TB): the
        # per-call fp16→fp32 chunk conversion dominated plan wall time
        self.emb = np.asarray(np.load(emb_path, mmap_mode="r"), dtype=np.float32)
        self.titles = titles
        self.chunk = chunk
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.float16).to(device).eval()

    def _encode(self, queries):
        enc = self.tok(["query: " + q for q in queries], padding=True,
                       truncation=True, max_length=128,
                       return_tensors="pt").to(self.device)
        with torch.inference_mode():
            hid = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (hid * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            emb = F.normalize(emb, dim=-1)
        return emb.float().cpu().numpy()

    def search_batch(self, queries, k=20):
        q = self._encode(queries)  # (nq, d)
        nq = q.shape[0]
        n = self.emb.shape[0]
        best_scores = np.full((nq, k), -1e9, dtype=np.float32)
        best_idx = np.zeros((nq, k), dtype=np.int64)
        for start in range(0, n, self.chunk):
            block = self.emb[start:start + self.chunk]
            scores = q @ block.T  # (nq, chunk)
            top = np.argpartition(-scores, min(k, scores.shape[1] - 1), axis=1)[:, :k]
            for qi in range(nq):
                cand_idx = np.concatenate([best_idx[qi], top[qi] + start])
                cand_sc = np.concatenate([best_scores[qi], scores[qi, top[qi]]])
                order = np.argsort(-cand_sc)[:k]
                best_idx[qi], best_scores[qi] = cand_idx[order], cand_sc[order]
        out = []
        for qi in range(nq):
            out.append([
                {"title": self.titles[int(i)], "score": float(s), "idx": int(i)}
                for i, s in zip(best_idx[qi], best_scores[qi])
            ])
        return out

    def search(self, query, k=20):
        return self.search_batch([query], k=k)[0]


class CEScorer:
    """Cross-encoder relevance scorer over (query, passage) pairs."""

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
                 device="cuda", batch=64):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=torch.float16).to(device).eval()
        self.device = device
        self.batch = batch

    def score(self, query, passages):
        scores = []
        with torch.inference_mode():
            for i in range(0, len(passages), self.batch):
                block = passages[i:i + self.batch]
                enc = self.tok([query] * len(block), block, padding=True,
                               truncation=True, max_length=384,
                               return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits.squeeze(-1)
                scores.extend(logits.float().cpu().tolist())
        return scores
