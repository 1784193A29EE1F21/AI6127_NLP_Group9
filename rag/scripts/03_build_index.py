"""Step 3 (Q2): build the BM25 index and the dense embedding index.

Usage: uv run python scripts/03_build_index.py
"""
from __future__ import annotations

import argparse
import json
import time

from rag import config
from rag.bm25 import BM25Index
from rag.corpus import load_chunks
from rag.dense import DenseEncoder, DenseIndex
from rag.text import tokenize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["bm25", "dense", "both"], default="both")
    ap.add_argument("--model", default=config.EMBED_MODEL)
    ap.add_argument("--name", default="dense", help="file prefix of the dense index")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default=None, help="cuda / cpu / mps (auto if empty)")
    args = ap.parse_args()

    chunks = load_chunks(config.PATHS.chunks_file)
    texts = [c.index_text() for c in chunks]
    out = config.PATHS.index_dir
    log = {}
    print(f"{len(chunks):,} chunks")

    if args.only in ("bm25", "both"):
        t = time.time()
        toks = [tokenize(x) for x in texts]
        bm25 = BM25Index().fit(toks)
        bm25.save(out)
        log["bm25_seconds"] = round(time.time() - t, 1)
        log["bm25_vocab"] = len(bm25.vocab)
        print(f"BM25: vocab={len(bm25.vocab):,}  {log['bm25_seconds']} s")

    if args.only in ("dense", "both"):
        t = time.time()
        enc = DenseEncoder(args.model, device=args.device)
        print(f"Encoding with {args.model} (dim={enc.dim}, max_seq_len={enc.model.max_seq_length}) ...")
        emb = enc.encode_passages(texts, batch_size=args.batch_size)
        DenseIndex(emb, args.model).save(out, args.name)
        log[f"{args.name}_seconds"] = round(time.time() - t, 1)
        log[f"{args.name}_passages_per_second"] = round(len(texts) / (time.time() - t), 1)
        print(f"Dense: {emb.shape}  {log[f'{args.name}_seconds']} s")

    with open(out / f"build_log_{args.only}_{args.name}.json", "w") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()
