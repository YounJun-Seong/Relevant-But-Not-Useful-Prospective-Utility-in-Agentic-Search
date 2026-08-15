"""Build the bm25s index over corpus.jsonl."""
import json
import os
import sys

import bm25s
import numpy as np
import Stemmer

CORPUS = sys.argv[1] if len(sys.argv) > 1 else "data/corpus.jsonl"
INDEX_DIR = sys.argv[2] if len(sys.argv) > 2 else "data/bm25_index"

with open(CORPUS.replace(".jsonl", "_titles.json")) as f:
    titles = json.load(f)

texts = []
with open(CORPUS) as f:
    for line in f:
        doc = json.loads(line)
        # index on "title. text" so title terms are searchable (entity queries)
        texts.append(doc["title"] + ". " + doc["text"])
print(f"{len(texts)} docs loaded", flush=True)
assert len(texts) == len(titles)

stemmer = Stemmer.Stemmer("english")
tokens = bm25s.tokenize(texts, stemmer=stemmer, show_progress=True)
bm25 = bm25s.BM25()
bm25.index(tokens, show_progress=True)
bm25.save(INDEX_DIR)

offsets = np.load(CORPUS.replace(".jsonl", "_offsets.npy"))
np.save(os.path.join(INDEX_DIR, "text_offsets.npy"), offsets)
with open(os.path.join(INDEX_DIR, "docs_meta.json"), "w") as f:
    json.dump({"titles": titles, "texts_path": os.path.abspath(CORPUS)}, f)
print("INDEX_DONE")
