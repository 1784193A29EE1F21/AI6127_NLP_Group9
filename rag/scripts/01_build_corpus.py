"""Step 1 (Q1): clean and chunk the Wikipedia files, compute corpus statistics.

Usage: uv run python scripts/01_build_corpus.py [--pattern "*/wiki_*"]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

from rag.config import PATHS
from rag.corpus import (chunk_document, chunk_to_dict, clean_document, iter_raw_docs_from_files,
                        iter_raw_docs_from_hf, iter_wiki_files, save_jsonl)
from rag.text import words


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["local", "hf"], default="local")
    ap.add_argument("--raw_dir", type=Path, default=None,
                    help="folder with WikiExtractor output (default: data/raw, else ../wiki_output)")
    ap.add_argument("--pattern", default="AA/wiki_[0-9][0-9]",
                    help="files inside wiki_output. Default = AA/wiki_00..99 (all annotated files). "
                         "Use \"*/wiki_*\" for every source file")
    ap.add_argument("--max_files", type=int, default=0, help="0 = all files")
    ap.add_argument("--hf_max_doc_id", type=int, default=5700,
                    help="HF source: keep articles with page id <= this (wiki_00..wiki_49 end at ~5670)")
    ap.add_argument("--target_words", type=int, default=180)
    ap.add_argument("--max_words", type=int, default=300)
    ap.add_argument("--overlap_sentences", type=int, default=1)
    ap.add_argument("--min_doc_words", type=int, default=50, help="drop stubs / disambiguation pages")
    ap.add_argument("--out_dir", type=Path, default=PATHS.corpus_dir)
    args = ap.parse_args()

    t0 = time.time()
    if args.source == "local":
        wiki_output = PATHS.root.parent / "wiki_output"
        if args.raw_dir is not None:
            src = args.raw_dir
            files = sorted(p for p in src.glob(args.pattern) if p.is_file()) or iter_wiki_files(src)
        elif PATHS.raw_dir.exists() and iter_wiki_files(PATHS.raw_dir):
            src, files = PATHS.raw_dir, iter_wiki_files(PATHS.raw_dir)
        elif wiki_output.exists():
            src = wiki_output
            files = sorted(p for p in wiki_output.glob(args.pattern) if p.is_file())
        else:  # wiki_* files lying directly in the project folder
            src = PATHS.root
            files = sorted(p for p in PATHS.root.glob("wiki_*") if p.is_file())
        args.raw_dir = src
        print(f"Source folder: {src}")
        if args.max_files:
            files = files[: args.max_files]
        if not files:
            sys.exit(f"No wiki_* files found in {args.raw_dir}. Copy the WikiExtractor output there "
                     f"or use --source hf.")
        rel = [str(p.relative_to(src)) for p in files]
        print(f"Reading {len(files)} file(s): {rel[0]} ... {rel[-1]}")
        raw_docs = iter_raw_docs_from_files(files)
    else:
        raw_docs = iter_raw_docs_from_hf(max_doc_id=args.hf_max_doc_id)

    docs, chunks = [], []
    raw_words = 0
    seen_ids, seen_text = set(), set()
    dropped = Counter()
    for doc_id, url, title, body in raw_docs:
        raw_words += len(body.split())
        if doc_id in seen_ids:
            dropped["duplicate_id"] += 1
            continue
        seen_ids.add(doc_id)
        doc = clean_document(doc_id, url, title, body)
        if doc.n_words < args.min_doc_words:
            dropped["too_short"] += 1
            continue
        if title.lower().endswith("(disambiguation)"):
            dropped["disambiguation"] += 1
            continue
        cs = chunk_document(doc, args.target_words, args.max_words, overlap_sentences=args.overlap_sentences)
        kept = []
        for c in cs:  # exact-duplicate passage removal
            key = c.text.lower()
            if key in seen_text:
                dropped["duplicate_chunk"] += 1
                continue
            seen_text.add(key)
            kept.append(c)
        if not kept:
            continue
        docs.append({"doc_id": doc.doc_id, "title": doc.title, "url": doc.url, "n_words": doc.n_words,
                     "n_chunks": len(kept), "sections": [s for s, _ in doc.sections]})
        chunks.extend(kept)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(docs, args.out_dir / "docs.jsonl")
    save_jsonl((chunk_to_dict(c) for c in chunks), args.out_dir / "chunks.jsonl")

    # corpus statistics (Q1.3)
    vocab = Counter()
    n_words = 0
    for c in chunks:
        ws = words(c.text)
        vocab.update(w.lower() for w in ws)
        n_words += len(ws)
    clean_text_words = sum(d["n_words"] for d in docs)
    lens = [c.n_words for c in chunks]
    stats = {
        "n_documents": len(docs),
        "n_chunks": len(chunks),
        "n_words_raw": raw_words,
        "n_words_clean_documents": clean_text_words,
        "n_words_in_chunks_with_overlap": n_words,
        "n_types_lowercased": len(vocab),
        "chunk_words_mean": round(statistics.mean(lens), 1) if lens else 0,
        "chunk_words_median": statistics.median(lens) if lens else 0,
        "chunk_words_min": min(lens) if lens else 0,
        "chunk_words_max": max(lens) if lens else 0,
        "dropped": dict(dropped),
        "chunking": {"target_words": args.target_words, "max_words": args.max_words,
                     "overlap_sentences": args.overlap_sentences},
        "build_seconds": round(time.time() - t0, 1),
    }
    with open(args.out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    if stats["n_chunks"] < 10_000 or stats["n_words_clean_documents"] < 100_000:
        print("WARNING: the assignment needs >= 10,000 passages and >= 100,000 words. Add more wiki files.")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = PATHS.results_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 3.5))
        plt.hist(lens, bins=50, color="#4C72B0")
        plt.xlabel("words per chunk")
        plt.ylabel("number of chunks")
        plt.title(f"Chunk length distribution (n={len(chunks):,})")
        plt.tight_layout()
        plt.savefig(fig_dir / "chunk_length_hist.png", dpi=150)
        print("saved", fig_dir / "chunk_length_hist.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
