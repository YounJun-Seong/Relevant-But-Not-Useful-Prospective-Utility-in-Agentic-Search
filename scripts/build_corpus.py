"""Extract the HotpotQA enwiki-20171001 abstracts tarball into a flat jsonl:
one line per doc: {"title": ..., "text": ...}. Also writes byte offsets for
random access and a titles list for the BM25 index metadata."""
import bz2
import json
import sys
import tarfile

import numpy as np

TAR = sys.argv[1] if len(sys.argv) > 1 else "data/enwiki-abstracts.tar.bz2"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/corpus.jsonl"

n = 0
titles = []
offsets = []
seen = set()
with open(OUT, "wb") as out, tarfile.open(TAR, "r:bz2") as tar:
    for member in tar:
        if not member.isfile() or not member.name.endswith(".bz2"):
            continue
        f = tar.extractfile(member)
        data = bz2.decompress(f.read())
        for line in data.splitlines():
            if not line.strip():
                continue
            doc = json.loads(line)
            title = doc["title"]
            if title in seen:
                continue
            seen.add(title)
            # 'text' is a list of sentences (abstract); join them.
            text = "".join(doc["text"]).strip()
            if not text:
                continue
            row = json.dumps({"title": title, "text": text}, ensure_ascii=False).encode()
            offsets.append(out.tell())
            out.write(row + b"\n")
            titles.append(title)
            n += 1
            if n % 200000 == 0:
                print(f"{n} docs...", flush=True)

np.save(OUT.replace(".jsonl", "_offsets.npy"), np.array(offsets, dtype=np.int64))
with open(OUT.replace(".jsonl", "_titles.json"), "w") as f:
    json.dump(titles, f)
print(f"DONE {n} docs")
