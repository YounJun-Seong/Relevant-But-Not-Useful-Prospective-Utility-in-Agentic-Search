"""Embed the corpus with intfloat/e5-base-v2 into a fp16 memmap for the dense
retrieval leg. E5 protocol: 'passage: ' prefix, mean pooling, L2 normalize."""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def mean_pool(hidden, mask):
    mask = mask.unsqueeze(-1).float()
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus.jsonl")
    ap.add_argument("--out", default="data/e5_embeddings.fp16.npy")
    ap.add_argument("--model", default="intfloat/e5-base-v2")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--max-len", type=int, default=256)
    args = ap.parse_args()

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16).to(device).eval()

    texts = []
    with open(args.corpus) as f:
        for line in f:
            d = json.loads(line)
            texts.append("passage: " + d["title"] + ". " + d["text"])
    n = len(texts)
    print(f"{n} passages", flush=True)

    dim = model.config.hidden_size
    out = np.lib.format.open_memmap(args.out, mode="w+", dtype=np.float16, shape=(n, dim))
    with torch.inference_mode():
        for i in range(0, n, args.batch):
            batch = texts[i:i + args.batch]
            enc = tok(batch, padding=True, truncation=True, max_length=args.max_len,
                      return_tensors="pt").to(device)
            hid = model(**enc).last_hidden_state
            emb = F.normalize(mean_pool(hid, enc["attention_mask"]), dim=-1)
            out[i:i + len(batch)] = emb.half().cpu().numpy()
            if (i // args.batch) % 200 == 0:
                print(f"{i}/{n}", flush=True)
    out.flush()
    print("EMBED_DONE", flush=True)


if __name__ == "__main__":
    main()
