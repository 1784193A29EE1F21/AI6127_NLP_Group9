"""Step 4 (Q3): retrieval evaluation (P@K, R@K, MRR, nDCG) for every configuration.

Usage: uv run python scripts/04_eval_retrieval.py [--llm]
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from rag.config import PATHS
from rag.corpus import load_jsonl
from rag.dense import DenseIndex
from rag.metrics import hit_at_k, mrr_at_k, retrieval_metrics
from rag.pipeline import RAGSystem, RetrievalConfig

K = 10


def configs(has_v1: bool, use_llm: bool):
    C = []
    C.append(("BM25", "dense", RetrievalConfig(mode="bm25", rerank=False, top_k=K)))
    C.append(("BM25 + PRF", "dense", RetrievalConfig(mode="bm25", prf=True, rerank=False, top_k=K)))
    if has_v1:
        C.append(("Dense v1 (paraphrase-MiniLM, notebook)", "dense_v1",
                  RetrievalConfig(mode="dense", rerank=False, top_k=K)))
    C.append(("Dense (multilingual-e5-small)", "dense", RetrievalConfig(mode="dense", rerank=False, top_k=K)))
    C.append(("Hybrid RRF", "dense", RetrievalConfig(mode="hybrid", rerank=False, top_k=K)))
    C.append(("Hybrid RRF + PRF", "dense", RetrievalConfig(mode="hybrid", prf=True, rerank=False, top_k=K)))
    if has_v1:
        C.append(("Dense v1 + rerank (notebook pipeline)", "dense_v1",
                  RetrievalConfig(mode="dense", rerank=True, top_k=K)))
    C.append(("Hybrid RRF + rerank", "dense", RetrievalConfig(mode="hybrid", rerank=True, top_k=K)))
    C.append(("Hybrid RRF + PRF + rerank", "dense",
              RetrievalConfig(mode="hybrid", prf=True, rerank=True, top_k=K)))
    if use_llm:
        C.append(("Hybrid + LLM rewrite + rerank", "dense",
                  RetrievalConfig(mode="hybrid", rewrite=True, rerank=True, top_k=K)))
        C.append(("Hybrid + HyDE + rerank", "dense",
                  RetrievalConfig(mode="hybrid", hyde=True, rerank=True, top_k=K)))
    return C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=PATHS.eval_file)
    ap.add_argument("--n", type=int, default=0, help="evaluate only the first n questions (0 = all)")
    ap.add_argument("--llm", action="store_true", help="also evaluate LLM query rewriting and HyDE")
    ap.add_argument("--llm_n", type=int, default=200, help="questions used for LLM configurations")
    ap.add_argument("--only", nargs="*", default=None, help="run only configs whose name contains one of these")
    ap.add_argument("--out", type=Path, default=PATHS.results_dir / "retrieval")
    args = ap.parse_args()

    qs = [q for q in load_jsonl(args.eval) if q["gold_chunk_ids"]]
    if args.n:
        qs = qs[: args.n]
    print(f"{len(qs)} questions with passage-level relevance labels")

    base = RAGSystem()
    systems = {"dense": base}
    has_v1 = (PATHS.index_dir / "dense_v1_emb.npy").exists()
    if has_v1:
        v1 = copy.copy(base)
        v1.dense = DenseIndex.load(PATHS.index_dir, "dense_v1")
        v1.encoder = None
        systems["dense_v1"] = v1
    base.warmup()

    doc_of_chunk = {c.chunk_id: c.doc_id for c in base.chunks}
    (args.out / "runs").mkdir(parents=True, exist_ok=True)
    with open(args.out / "qrels.trec", "w") as f:
        for q in qs:
            for cid in q["gold_chunk_ids"]:
                f.write(f"{q['id']} 0 {cid} 1\n")

    rows, per_query = [], {}
    for name, dense_name, cfg in configs(has_v1, args.llm):
        if args.only and not any(s.lower() in name.lower() for s in args.only):
            continue
        sysm = systems[dense_name]
        subset = qs[: args.llm_n] if (cfg.rewrite or cfg.hyde) else qs
        agg = defaultdict(list)
        lat = []
        pq = {}
        run_path = args.out / "runs" / (name.replace(" ", "_").replace("+", "plus").replace("(", "")
                                        .replace(")", "").replace(",", "") + ".trec")
        with open(run_path, "w") as run:
            for q in subset:
                t = time.perf_counter()
                hits, _ = sysm.retrieve(q["question"], cfg)
                lat.append(time.perf_counter() - t)
                ranked = [h["chunk_id"] for h in hits]
                rel = set(q["gold_chunk_ids"])
                m = retrieval_metrics(ranked, rel)
                docs_ranked = [doc_of_chunk[c] for c in ranked]
                m["DocHit@5"] = hit_at_k(docs_ranked, {q["gold_doc_id"]}, 5)
                m["DocMRR@10"] = mrr_at_k(docs_ranked, {q["gold_doc_id"]}, 10)
                for k, v in m.items():
                    agg[k].append(v)
                first = next((h["rank"] for h in hits if h["chunk_id"] in rel), None)
                pq[q["id"]] = {"first_relevant_rank": first, "top3": [f"{h['title']} > {h['section']}" for h in hits[:3]]}
                for h in hits:
                    s = h.get("rerank", h.get("rrf", 0))
                    run.write(f"{q['id']} Q0 {h['chunk_id']} {h['rank']} {s} {name.replace(' ', '_')}\n")
        row = {"config": name, "n": len(subset)}
        row.update({k: round(float(np.mean(v)), 4) for k, v in agg.items()})
        row["latency_ms_mean"] = round(1000 * float(np.mean(lat)), 1)
        row["latency_ms_p95"] = round(1000 * float(np.percentile(lat, 95)), 1)
        rows.append(row)
        per_query[name] = pq
        print(f"{name:45s} MRR@10={row['MRR@10']:.3f} nDCG@10={row['nDCG@10']:.3f} "
              f"R@5={row['R@5']:.3f} Hit@5={row['Hit@5']:.3f} {row['latency_ms_mean']} ms")

    # save tables
    cols = ["config", "n", "P@1", "P@3", "P@5", "R@1", "R@5", "R@10", "Hit@1", "Hit@5", "Hit@10", "MRR@10",
            "nDCG@10", "DocHit@5", "DocMRR@10", "latency_ms_mean", "latency_ms_p95"]
    with open(args.out / "summary.csv", "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(f'"{r.get(c)}"' if c == "config" else str(r.get(c, "")) for c in cols) + "\n")
    with open(args.out / "summary.md", "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n")
    with open(args.out / "per_query.json", "w", encoding="utf-8") as f:
        json.dump(per_query, f, ensure_ascii=False, indent=1)

    # breakdown by question type for the best configuration
    best = max(rows, key=lambda r: r["MRR@10"])["config"]
    by_type = defaultdict(list)
    for q in qs:
        r = per_query[best].get(q["id"])
        if r is not None:
            by_type[q["question_type"]].append(1.0 / r["first_relevant_rank"] if r["first_relevant_rank"] else 0.0)
    with open(args.out / "by_question_type.csv", "w") as f:
        f.write("question_type,n,MRR@10\n")
        for t, v in sorted(by_type.items()):
            f.write(f"{t},{len(v)},{np.mean(v):.4f}\n")

    # case studies: fixed by the final pipeline (for the report, Q3)
    cases = []
    final = per_query.get("Hybrid RRF + PRF + rerank") or per_query[best]
    for q in qs:
        f_rank = final.get(q["id"], {}).get("first_relevant_rank")
        b_rank = per_query.get("BM25", {}).get(q["id"], {}).get("first_relevant_rank")
        d_rank = per_query.get("Dense (multilingual-e5-small)", {}).get(q["id"], {}).get("first_relevant_rank")
        if f_rank == 1 and (b_rank is None or b_rank > 3 or d_rank is None or d_rank > 3):
            cases.append({"id": q["id"], "question": q["question"], "gold": q["source_title"],
                          "bm25_rank": b_rank, "dense_rank": d_rank, "final_rank": f_rank,
                          "final_top3": final[q["id"]]["top3"]})
    with open(args.out / "case_studies.json", "w", encoding="utf-8") as f:
        json.dump(cases[:50], f, ensure_ascii=False, indent=1)
    print(f"best config: {best};  {len(cases)} case-study queries saved")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        (args.out / "figures").mkdir(exist_ok=True)
        names = [r["config"] for r in rows]
        x = np.arange(len(rows))
        plt.figure(figsize=(max(7, len(rows) * 1.1), 4))
        for i, met in enumerate(["MRR@10", "nDCG@10", "Hit@5"]):
            plt.bar(x + (i - 1) * 0.27, [r[met] for r in rows], width=0.27, label=met)
        plt.xticks(x, names, rotation=30, ha="right", fontsize=8)
        plt.ylim(0, 1.15)
        plt.legend(ncol=3, loc="upper center", frameon=False)
        plt.title("Retrieval quality by configuration")
        plt.tight_layout()
        plt.savefig(args.out / "figures" / "retrieval_comparison.png", dpi=150)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
