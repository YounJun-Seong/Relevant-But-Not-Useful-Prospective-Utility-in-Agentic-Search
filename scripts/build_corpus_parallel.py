"""Parallel version: process the extracted wiki dump directory (inner .bz2
files) with multiprocessing, then merge into flat corpus.jsonl + offsets."""
import bz2
import glob
import json
import sys
from multiprocessing import Pool

import numpy as np

RAW_DIR = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/corpus.jsonl"


def process_file(path):
    rows = []
    with open(path, "rb") as f:
        data = bz2.decompress(f.read())
    for line in data.splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        text = "".join(doc["text"]).strip()
        if not text:
            continue
        rows.append((doc["title"], text))
    return rows


if __name__ == "__main__":
    files = sorted(glob.glob(f"{RAW_DIR}/**/*.bz2", recursive=True))
    print(f"{len(files)} shard files", flush=True)
    seen = set()
    titles = []
    offsets = []
    n = 0
    with open(OUT, "wb") as out, Pool(64) as pool:
        for rows in pool.imap(process_file, files, chunksize=8):
            for title, text in rows:
                if title in seen:
                    continue
                seen.add(title)
                row = json.dumps({"title": title, "text": text},
                                 ensure_ascii=False).encode()
                offsets.append(out.tell())
                out.write(row + b"\n")
                titles.append(title)
                n += 1
                if n % 500000 == 0:
                    print(f"{n} docs...", flush=True)
    np.save(OUT.replace(".jsonl", "_offsets.npy"), np.array(offsets, dtype=np.int64))
    with open(OUT.replace(".jsonl", "_titles.json"), "w") as f:
        json.dump(titles, f)
    print(f"DONE {n} docs", flush=True)
