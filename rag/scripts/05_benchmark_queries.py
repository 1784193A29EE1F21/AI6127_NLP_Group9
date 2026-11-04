"""Step 5 (Q2): five example queries with results and per-stage latency.

Usage: uv run python scripts/05_benchmark_queries.py [--generate]
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

from rag.config import PATHS
from rag.pipeline import GenerationConfig, RAGSystem, RetrievalConfig

QUERIES = [
    # (query, title filter, why it is interesting)
    ("When and where was An American in Paris first performed, and who conducted the premiere?", "",
     "multi-part factual question (the notebook v1 pipeline missed the conductor)"),
    ("Why did Ayn Rand reject libertarianism?", "", "'why' question needing an explanation"),
    ("酸雨是如何形成的？它对环境有什么影响？", "", "Chinese query over an English corpus (multilingual)"),
    ("What is the difference between Ajax the Great and Ajax the Lesser?", "",
     "comparison across two articles"),
    ("What tools does it use to control inflation?", "central bank",
     "metadata filter: search only inside articles whose title contains 'central bank'"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--repeats", type=int, default=5, help="timed runs per query (after one warm-up)")
    args = ap.parse_args()
    if args.repeats <= 0:
        ap.error("--repeats must be positive")

    t = time.perf_counter()
    rag = RAGSystem()
    load_s = time.perf_counter() - t
    if args.generate:
        try:
            _ = rag.llm
        except Exception as e:
            print(f"!! LLM not available ({e}) - running retrieval only. Fill GOOGLE_API_KEY in .env.")
            args.generate = False
    t = time.perf_counter()
    rag.warmup()
    warm_s = time.perf_counter() - t

    out = {"index_load_seconds": round(load_s, 2), "model_warmup_seconds": round(warm_s, 2),
           "n_passages": len(rag.chunks), "queries": []}
    for q, title_filter, why in QUERIES:
        cfg = RetrievalConfig(mode="hybrid", prf=False, rerank=True, top_k=5, title_filter=title_filter,
                              max_per_doc=0)
        rag.retrieve(q, cfg)  # warm-up for this query shape
        stage_times = []
        for _ in range(args.repeats):
            hits, info = rag.retrieve(q, cfg)
            stage_times.append(info["timings"])
        timing = {k: round(1000 * statistics.mean(t[k] for t in stage_times), 1) for k in stage_times[0]}
        entry = {"query": q, "why": why, "title_filter": title_filter, "retrieval_ms": timing,
                 "results": [{"rank": h["rank"], "title": h["title"], "section": h["section"], "url": h["url"],
                              "bm25_rank": h.get("bm25_rank"), "dense_rank": h.get("dense_rank"),
                              "rerank_score": h.get("rerank"), "text": h["text"]} for h in hits]}
        if info.get("note"):
            entry["note"] = info["note"]
        if args.generate:
            res = rag.answer(q, cfg, GenerationConfig(style="detailed", self_check=True))
            entry["answer"] = res["answer"]
            entry["draft_answer"] = res["draft_answer"]
            entry["unsupported_claims"] = res.get("self_check", {}).get("unsupported", [])
            entry["end_to_end_ms"] = {k: round(1000 * v, 1) for k, v in res["timings"].items()}
        out["queries"].append(entry)
        print(f"\n### {q}\n  retrieval: {timing}")
        for h in hits:
            print(f"  [{h['rank']}] {h['title']} > {h['section']}  (bm25 #{h.get('bm25_rank')}, "
                  f"dense #{h.get('dense_rank')}, rerank {h.get('rerank')})")
        if args.generate:
            print("  ANSWER:", entry["answer"][:500])

    PATHS.results_dir.mkdir(parents=True, exist_ok=True)
    with open(PATHS.results_dir / "five_queries.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(PATHS.results_dir / "five_queries.md", "w", encoding="utf-8") as f:
        f.write(f"Index: {out['n_passages']:,} passages; load {load_s:.1f}s; model warm-up {warm_s:.1f}s\n\n")
        f.write("| # | Query | BM25 ms | Dense ms | Fusion ms | Rerank ms | Total retrieval ms |"
                + (" End-to-end ms |" if args.generate else "") + "\n")
        f.write("|---|---|---|---|---|---|---|" + ("---|" if args.generate else "") + "\n")
        for i, e in enumerate(out["queries"], 1):
            t = e["retrieval_ms"]
            f.write(f"| {i} | {e['query']} | {t.get('bm25')} | {t.get('dense')} | {t.get('fusion')} | "
                    f"{t.get('rerank')} | {t.get('retrieval_total')} |"
                    + (f" {e['end_to_end_ms'].get('total')} |" if args.generate else "") + "\n")
        for i, e in enumerate(out["queries"], 1):
            f.write(f"\n## Q{i}. {e['query']}\n*{e['why']}*\n\n")
            if e.get("answer"):
                f.write(f"**Answer:** {e['answer']}\n\n")
            for r in e["results"]:
                f.write(f"{r['rank']}. **{r['title']} > {r['section']}** ({r['url']}) - "
                        f"{r['text'][:300]}...\n")
    print("\nsaved results/five_queries.json and results/five_queries.md")


if __name__ == "__main__":
    main()
