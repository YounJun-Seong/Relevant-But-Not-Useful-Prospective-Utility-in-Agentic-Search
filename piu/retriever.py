"""BM25 retrieval over the HotpotQA wiki abstracts corpus (bm25s backend)."""
import json
import os

import bm25s
import Stemmer


class BM25Retriever:
    def __init__(self, index_dir):
        self.index_dir = index_dir
        self.stemmer = Stemmer.Stemmer("english")
        # mmap=False: the index is ~1.3GB, disk-backed scoring dominates plan
        # phase wall time on this box — keep it in RAM
        self.bm25 = bm25s.BM25.load(index_dir, load_corpus=False, mmap=False)
        with open(os.path.join(index_dir, "docs_meta.json")) as f:
            meta = json.load(f)
        self.titles = meta["titles"]
        self.title2idx = {t: i for i, t in enumerate(self.titles)}
        self.texts_path = meta["texts_path"]
        self._offsets = None

    def _load_offsets(self):
        if self._offsets is None:
            import numpy as np

            self._offsets = np.load(os.path.join(self.index_dir, "text_offsets.npy"))

    def get_text(self, idx):
        """Random-access one abstract from the flat jsonl using byte offsets."""
        self._load_offsets()
        with open(self.texts_path, "rb") as f:
            f.seek(self._offsets[idx])
            line = f.readline()
        return json.loads(line)["text"]

    def doc_by_title(self, title):
        idx = self.title2idx.get(title)
        if idx is None:
            return None
        return {"title": title, "text": self.get_text(idx), "idx": idx}

    def search(self, query, k=10):
        tokens = bm25s.tokenize([query], stemmer=self.stemmer, show_progress=False)
        results, scores = self.bm25.retrieve(tokens, k=k, show_progress=False)
        out = []
        for rank in range(results.shape[1]):
            idx = int(results[0, rank])
            out.append(
                {
                    "title": self.titles[idx],
                    "text": self.get_text(idx),
                    "score": float(scores[0, rank]),
                    "idx": idx,
                }
            )
        return out

    def search_batch(self, queries, k=10):
        tokens = bm25s.tokenize(queries, stemmer=self.stemmer, show_progress=False)
        results, scores = self.bm25.retrieve(tokens, k=k, show_progress=False)
        all_out = []
        for qi in range(results.shape[0]):
            out = []
            for rank in range(results.shape[1]):
                idx = int(results[qi, rank])
                out.append(
                    {
                        "title": self.titles[idx],
                        "text": self.get_text(idx),
                        "score": float(scores[qi, rank]),
                        "idx": idx,
                    }
                )
            all_out.append(out)
        return all_out
