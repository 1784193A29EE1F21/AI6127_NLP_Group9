"""Step 2 (Q1/Q4): merge annotation files into data/eval/eval_qa.jsonl and link gold passages.

Usage: uv run python scripts/02_build_eval.py --ann_dirs "../Data Annotation - 00-49"
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from rag.config import PATHS
from rag.corpus import load_jsonl, save_jsonl
from rag.metrics import normalize_answer
from rag.annotations import default_annotation_dirs, attach_annotation_variants

REQUIRED = ["id", "question", "answer", "source_title", "source_url", "evidence", "question_type"]


def evidence_recall(evidence: str, text: str) -> float:
    ev = normalize_answer(evidence).split()
    if not ev:
        return 0.0
    tx = set(normalize_answer(text).split())
    return sum(1 for w in ev if w in tx) / len(ev)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann_dirs", nargs="+", type=Path,
                    default=default_annotation_dirs())
    ap.add_argument("--chunks", type=Path, default=PATHS.chunks_file)
    ap.add_argument("--valid_ids", type=Path, default=None,
                    help="optional file with ids that passed the inter-annotator check (one per line)")
    ap.add_argument("--min_recall", type=float, default=0.6)
    ap.add_argument("--out", type=Path, default=PATHS.eval_file)
    args = ap.parse_args()

    missing = [str(d) for d in args.ann_dirs if not d.is_dir()]
    if missing:
        ap.error("Annotation folders not found: " + ", ".join(missing))
    if not args.chunks.is_file():
        ap.error(f"Corpus not found: {args.chunks}; run 01_build_corpus.py first")

    records, problems = [], Counter()
    for d in args.ann_dirs:
        for f in sorted(Path(d).glob("*_qa.json")):
            data = json.loads(f.read_text(encoding="utf-8"))
            for r in data:
                if any(not str(r.get(k, "")).strip() for k in REQUIRED):
                    problems["missing_field"] += 1
                    continue
                r["annotation_file"] = f.name
                records.append(r)
    print(f"loaded {len(records)} annotated QA pairs from {len(args.ann_dirs)} folder(s)")
    if not records:
        ap.error("No valid annotated QA records found; existing evaluation files were not changed")

    if args.valid_ids:
        valid = {l.strip() for l in args.valid_ids.read_text(encoding="utf-8").splitlines() if l.strip()}
        before = len(records)
        records = [r for r in records if r["id"] in valid]
        print(f"kept {len(records)}/{before} records that passed the IAA validation")
        if not records:
            ap.error("IAA filter removed all records; existing evaluation files were not changed")

    # de-duplication (same id, or same normalised question)
    seen_q, seen_id, uniq = set(), set(), []
    for r in records:
        qn = normalize_answer(r["question"])
        if r["id"] in seen_id or qn in seen_q:
            problems["duplicate"] += 1
            continue
        seen_id.add(r["id"])
        seen_q.add(qn)
        uniq.append(r)

    # link to corpus passages
    chunks_by_doc = defaultdict(list)
    if args.chunks.exists():
        for c in load_jsonl(args.chunks):
            chunks_by_doc[c["doc_id"]].append(c)
    else:
        print(f"WARNING: {args.chunks} not found - run 01_build_corpus.py first; gold chunks left empty")

    out = []
    for r in uniq:
        m = re.search(r"curid=(\d+)", r["source_url"])
        doc_id = m.group(1) if m else ""
        cands = chunks_by_doc.get(doc_id, [])
        scored = sorted(((evidence_recall(r["evidence"], c["text"]), c["chunk_id"]) for c in cands),
                        reverse=True)
        gold = [cid for s, cid in scored if s >= args.min_recall]
        if not gold and scored and scored[0][0] >= 0.4:
            gold = [scored[0][1]]
        if not cands:
            problems["doc_not_in_corpus"] += 1
        elif not gold:
            problems["evidence_not_found"] += 1
        out.append({
            "id": r["id"],
            "question": r["question"].strip(),
            "answer": r["answer"].strip(),
            "evidence": r["evidence"].strip(),
            "question_type": r["question_type"],
            "source_title": r["source_title"],
            "source_url": r["source_url"],
            "gold_doc_id": doc_id,
            "gold_chunk_ids": gold,
            "best_evidence_recall": round(scored[0][0], 3) if scored else 0.0,
            "annotation_file": r["annotation_file"],
        })

    # Keep one question per ID, retaining the independent annotation variants.
    # The original human annotation stays the canonical answer; no majority vote
    # or automatic adjudication is implied by this integration.
    variant_summary = attach_annotation_variants(out, args.ann_dirs)
    save_jsonl(out, args.out)
    if variant_summary.get('attached'):
        review_queue = []
        for row in out:
            reasons = []
            if not row['question_type_all_agree']:
                reasons.append('question_type_disagreement')
            if row['ai_needs_review']:
                reasons.append('ai_answer_uncertain')
            if reasons:
                review_queue.append({**row, 'review_reasons': reasons})
        save_jsonl(review_queue, args.out.parent / 'review_queue.jsonl')
        variant_summary['review_queue_count'] = len(review_queue)
    report = {
        "n_records": len(out),
        "n_with_gold_chunks": sum(1 for r in out if r["gold_chunk_ids"]),
        "question_type_distribution": dict(Counter(r["question_type"] for r in out)),
        "n_source_articles": len({r["gold_doc_id"] for r in out}),
        "avg_answer_words": round(sum(len(r["answer"].split()) for r in out) / max(1, len(out)), 1),
        "problems": dict(problems),
        "annotation_variants": variant_summary,
    }
    with open(args.out.parent / "eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    if len(out) < 1000:
        print(f"NOTE: the assignment requires >= 1,000 labelled records (currently {len(out)}).")


if __name__ == "__main__":
    main()
