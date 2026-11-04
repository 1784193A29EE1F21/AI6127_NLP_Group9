"""Step 6 (Q4/Q5): answer quality, RAG metrics and ablation study.

Usage: uv run python scripts/06_eval_generation.py --n 30 [--only A B E]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dataclasses import asdict

from tqdm import tqdm

from rag import config
from rag.config import PATHS
from rag.llm import LLM, QuotaExceeded
from rag.corpus import load_jsonl
from rag.generator import closed_book_answer
from rag.judge import RAGJudge
from rag.metrics import bertscore_f1, exact_match, rouge_l, token_f1, token_recall
from rag.pipeline import GenerationConfig, RAGSystem, RetrievalConfig
from rag.ablations import generation_configs, GENERATION_CONTRASTS, paired_delta


def ablation_configs(include_all: bool):
    return generation_configs(include_all)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=PATHS.eval_file)
    ap.add_argument("--n", type=int, default=200, help="number of questions (API cost!)")
    ap.add_argument("--all", action="store_true", help="all 19 configurations, including full-system leave-one-out")
    ap.add_argument("--only", nargs="*", default=None, help="run configs whose name starts with these letters")
    ap.add_argument("--seed", type=int, default=42, help="random seed for choosing the questions")
    ap.add_argument("--bertscore", action="store_true")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel questions; keep 1 on free API tiers (tokens-per-minute limits)")
    ap.add_argument("--out", type=Path, default=PATHS.results_dir / "generation")
    args = ap.parse_args()
    if args.n <= 0 or args.workers <= 0:
        ap.error("--n and --workers must be positive")
    if args.workers != 1:
        ap.error("Use --workers 1 for attributable LLM call/token measurements")
    selected_configs = [(name,r,g) for name,r,g in ablation_configs(args.all)
                        if not args.only or any(name.startswith(s) for s in args.only)]
    if not selected_configs:
        ap.error('No ablation configurations matched --only')

    # Fail before loading GPU models when the model endpoint is not configured.
    generator_llm = LLM()

    qs = load_jsonl(args.eval)
    if not qs:
        ap.error("Evaluation dataset is empty")
    # random subset, balanced across question types
    random.Random(args.seed).shuffle(qs)
    by_t: dict[str, list] = {}
    for q in qs:
        by_t.setdefault(q["question_type"], []).append(q)
    subset = []
    while len(subset) < min(args.n, len(qs)):
        for t in sorted(by_t):
            if by_t[t] and len(subset) < args.n:
                subset.append(by_t[t].pop(0))
    print(f"{len(subset)} questions")

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / 'sample_ids.json', 'w', encoding='utf-8') as f:
        json.dump({'seed':args.seed,'ids':[q['id'] for q in subset]},f,indent=2)
    with open(args.out/'run_manifest.json','w',encoding='utf-8') as f:
        json.dump({'dataset_sha256':hashlib.sha256(args.eval.read_bytes()).hexdigest(),
                   'corpus_sha256':hashlib.sha256(PATHS.chunks_file.read_bytes()).hexdigest(),
                   'provider':generator_llm.provider,'model':generator_llm.model,
                   'n_requested':len(subset),'seed':args.seed,'workers':1,'temperature':0.0,
                   'configurations':[{'name':name,'retrieval':asdict(r) if r else None,
                                       'generation':asdict(g) if g else None} for name,r,g in selected_configs],
                   'warning':'Cached and uncached call counts are separate; paired comparisons use matched successful IDs.'},
                  f,ensure_ascii=False,indent=2)
    rag = RAGSystem(llm=generator_llm)
    rag.warmup()
    judge_llm = rag.llm
    if config.JUDGE_MODEL:
        judge_llm = LLM(provider=config.JUDGE_PROVIDER or None, model=config.JUDGE_MODEL,
                        base_url=config.JUDGE_BASE_URL or None, api_key=config.JUDGE_API_KEY or None)
    print(f"generator: {rag.llm.provider}/{rag.llm.model}   judge: {judge_llm.provider}/{judge_llm.model}")
    judge = RAGJudge(judge_llm, encoder=rag.get_encoder())
    # A shared generator/judge client must only be counted once.
    clients = [rag.llm] if judge_llm is rag.llm else [rag.llm, judge_llm]
    quota_hit = False
    args.out.mkdir(parents=True, exist_ok=True)

    summary = []
    completed = {}
    failures = []
    docs_by_id = {q['id']:q['gold_doc_id'] for q in subset}
    for name, rcfg, gcfg in selected_configs:
        tok_before = (sum(c.stats.prompt_tokens for c in clients),
                      sum(c.stats.output_tokens for c in clients))
        quota_flag = [False]

        def run(q):
            nonlocal_quota = quota_flag
            if nonlocal_quota[0]:
                return None
            try:
                return _run(q)
            except QuotaExceeded as e:
                nonlocal_quota[0] = True
                failures.append({'config':name,'id':q['id'],'error_type':'quota','message':str(e)})
                print("\n!!", e)
                return None
            except Exception as e:  # one failed question must not kill the whole run
                failures.append({'config':name,'id':q['id'],'error_type':type(e).__name__,'message':str(e)})
                print(f"\n!! question {q['id']} failed: {e}")
                return None

        def _run(q):
            t = time.perf_counter()
            calls_before = sum(c.stats.calls for c in clients)
            cache_before = sum(c.stats.cache_hits for c in clients)
            if rcfg is None:
                ans, hits = closed_book_answer(rag.llm, q["question"]), []
            else:
                res = rag.answer(q["question"], rcfg, gcfg)
                ans, hits = res["answer"], res["hits"]
            generation_calls = sum(c.stats.calls for c in clients) - calls_before
            lat = time.perf_counter() - t
            j = judge.evaluate(q["question"], q["answer"], q["evidence"], ans, hits)
            if rcfg is None:
                j["faithfulness"] = j["context_precision"] = j["context_recall"] = None
            gold_ids = set(q.get("gold_chunk_ids", []))
            return {"id": q["id"], "question_type": q["question_type"], "question": q["question"],
                    "gold": q["answer"], "answer": ans, "latency_s": lat, "llm_calls": generation_calls,
                    "all_calls_including_judge":sum(c.stats.calls for c in clients)-calls_before,
                    "cache_hits_including_judge":sum(c.stats.cache_hits for c in clients)-cache_before,
                    "EM": exact_match(ans, q["answer"]), "F1": token_f1(ans, q["answer"]),
                    "ROUGE_L": rouge_l(ans, q["answer"]), "gold_token_recall": token_recall(ans, q["answer"]),
                    "gold_passage_retrieved": bool(gold_ids & {h["chunk_id"] for h in hits}),
                    "retrieved": [f"{h['title']} > {h['section']}" for h in hits], **j}

        with ThreadPoolExecutor(args.workers) as ex:
            rows = list(tqdm(ex.map(run, subset), total=len(subset), desc=name))
        n_failed = sum(r is None for r in rows)
        with open(args.out/'failures.jsonl','w',encoding='utf-8') as f:
            for failure in failures:
                f.write(json.dumps(failure,ensure_ascii=False)+'\n')
        rows = [r for r in rows if r is not None]
        if quota_flag[0]:
            quota_hit = True
        if not rows:
            print(f"no results for {name} - skipped")
            if quota_hit:
                break
            continue
        if args.bertscore:
            bs = bertscore_f1([r["answer"] for r in rows], [r["gold"] for r in rows])
            if bs:
                for r, b in zip(rows, bs):
                    r["BERTScore_F1"] = b
        fname = name.split()[0] + ".jsonl"
        completed[name.split()[0]] = {r['id']:r for r in rows}
        with open(args.out / fname, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        def mean(key):
            vals = [r[key] for r in rows if r.get(key) is not None]
            return round(statistics.mean(vals), 4) if vals else None

        s = {"config": name, "n": len(rows), "n_failed": n_failed}
        for k in ["EM", "F1", "ROUGE_L", "gold_token_recall", "BERTScore_F1", "llm_correctness",
                  "faithfulness", "answer_relevance", "context_precision", "context_recall",
                  "gold_passage_retrieved", "latency_s", "llm_calls", "all_calls_including_judge", "cache_hits_including_judge"]:
            s[k] = mean(k)
        s["noncommittal_rate"] = mean("noncommittal")
        s["latency_p95_s"] = round(sorted(r["latency_s"] for r in rows)[int(0.95 * (len(rows) - 1))], 3)
        # tokens are only counted for non-cached calls; includes judge calls
        s["tokens_in"] = sum(c.stats.prompt_tokens for c in clients) - tok_before[0]
        s["tokens_out"] = sum(c.stats.output_tokens for c in clients) - tok_before[1]
        summary.append(s)
        print(json.dumps(s, indent=1))
        if quota_hit:
            print("Daily quota reached - stopping. Re-run the same command later: finished calls are cached.")
            break

    if not summary:
        with open(args.out/'run_status.json','w',encoding='utf-8') as f:
            json.dump({'complete':False,'n_requested':len(subset),'n_configurations_finished':0},f,indent=2)
        raise SystemExit('No configuration produced results; experiment incomplete')
    paired = []
    for feature,control,treatment in GENERATION_CONTRASTS:
        if control not in completed or treatment not in completed:
            continue
        a,b = completed[control],completed[treatment]
        for metric in ['EM','F1','ROUGE_L','llm_correctness','faithfulness','answer_relevance',
                       'context_precision','context_recall','gold_passage_retrieved']:
            ids = sorted(i for i in a.keys() & b.keys() if a[i].get(metric) is not None and b[i].get(metric) is not None)
            if ids:
                paired.append({'feature':feature,'control':control,'treatment':treatment,'metric':metric,
                               'n_requested':len(subset),'n_missing_pair':len(subset)-len(ids),'matched_ids':ids,
                               **paired_delta([a[i][metric] for i in ids],[b[i][metric] for i in ids],
                                              [docs_by_id[i] for i in ids],seed=args.seed)})
    with open(args.out/'paired_comparisons.json','w',encoding='utf-8') as f:
        json.dump(paired,f,ensure_ascii=False,indent=2)
    common_ids = sorted(set.intersection(*(set(rows) for rows in completed.values())))
    with open(args.out/'common_sample_summary.json','w',encoding='utf-8') as f:
        json.dump({'n_requested':len(subset),'n_common_success':len(common_ids),'ids':common_ids,
                   'configurations_completed':list(completed),
                   'metrics':{code:{metric:statistics.mean(rows[i][metric] for i in common_ids)
                                   if common_ids else None for metric in ['EM','F1','ROUGE_L']}
                              for code,rows in completed.items()}},f,ensure_ascii=False,indent=2)
    status = {'complete':len(completed)==len(selected_configs) and all(len(rs)==len(subset) for rs in completed.values()),
              'n_requested':len(subset),'n_configurations_requested':len(selected_configs),
              'n_configurations_with_results':len(completed),
              'per_config':{name.split()[0]:{'n_success':len(completed.get(name.split()[0],{})),
                                           'n_missing':len(subset)-len(completed.get(name.split()[0],{}))}
                            for name,_,_ in selected_configs}}
    with open(args.out/'run_status.json','w',encoding='utf-8') as f:
        json.dump(status,f,ensure_ascii=False,indent=2)
    cols = list(summary[0].keys())
    with open(args.out / "summary.csv", "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for s in summary:
            f.write(",".join(f'"{s[c]}"' if c == "config" else str(s[c]) for c in cols) + "\n")
    with open(args.out / "summary.md", "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for s in summary:
            f.write("| " + " | ".join(str(s[c]) for c in cols) + " |\n")
    print("saved", args.out / "summary.md")


if __name__ == "__main__":
    main()
