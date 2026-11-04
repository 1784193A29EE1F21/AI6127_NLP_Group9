"""Q4 with the two existing human annotations only: prepare, evaluate, report."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from rag import config
from rag.config import PATHS
from rag.corpus import load_jsonl, save_jsonl
from rag.generator import closed_book_answer, format_passages
from rag.human_pair import human_only, reference_scores, validate_pair, validate_reference_claims
from rag.judge import JUDGE_PROMPT, context_precision, context_recall
from rag.metrics import retrieval_metrics
from rag.pipeline import RAGSystem

spec=importlib.util.spec_from_file_location('api_base',PATHS.root/'scripts/13_api_experiments.py')
api=importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)
OUT=PATHS.results_dir/'q4_two_humans_2026-11-02'
DATA=PATHS.eval_dir/'human_pair/eval_qa.jsonl'
PHASES=['generation','closed_book','judge','agreement']
TARGETS={p:1359 for p in PHASES}


def hit_path(question):
    import hashlib
    key=hashlib.sha256(json.dumps([question,asdict(api.RCFG)],sort_keys=True).encode()).hexdigest()
    return OUT/'retrieval_cache'/(key+'.json')


def hits_for(question):
    return json.loads(hit_path(question).read_text(encoding='utf-8'))


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    DATA.parent.mkdir(parents=True,exist_ok=True)
    questions=human_only(load_jsonl(PATHS.eval_file))
    assert len(questions)==1359
    if DATA.exists():
        assert load_jsonl(DATA)==questions,'Human-only dataset changed'
    else:
        save_jsonl(questions,DATA)
    identity={'dataset_sha256':api.sha(DATA),'human_groups':['human_original','human_reannotated'],
              'n':1359,'selection':'All aligned human questions; no AI filtering',
              'corpus_sha256':api.sha(PATHS.chunks_file),'provider':config.LLM_PROVIDER,
              'base_url':config.OPENAI_BASE_URL,'model':config.LLM_MODEL,
              'retrieval':asdict(api.RCFG),'generation':asdict(api.GCFG),
              'reference_protocol':'Human original primary; max over two human references secondary',
              'judge_protocol':'Two human references; same-model automated judge',
              'workers':4,'rpm':config.LLM_RPM,'temperature':0.}
    manifest=OUT/'input_manifest.json'
    if manifest.exists(): assert json.loads(manifest.read_text(encoding='utf-8'))==identity
    else: api.atomic_json(manifest,identity)
    cache=OUT/'retrieval_cache'
    cache.mkdir(exist_ok=True)
    ranks_path=PATHS.results_dir/'ablation_full_2026-11-02/Hybrid_RRF_plus_rerank.trec.jsonl'
    ranks={r['id']:r for r in load_jsonl(ranks_path)}
    chunks={r['chunk_id']:r for r in load_jsonl(PATHS.chunks_file)}
    retrieval=[]
    for q in questions:
        ids=ranks[q['id']]['ranked'][:5]
        old=api.SOURCE/'retrieval_cache'/hit_path(q['question']).name
        if old.exists():
            pair=json.loads(old.read_text(encoding='utf-8'))
            assert [h['chunk_id'] for h in pair[0]]==ids,'Historical ranking mismatch'
        else:
            hits=[]
            for rank,cid in enumerate(ids,1):
                c=chunks[cid]
                hits.append({k:c.get(k,'') for k in ['chunk_id','doc_id','title','section','url','text']})
                hits[-1]['rank']=rank
            pair=[hits,{'timings':{'retrieval_total':0.},'config':asdict(api.RCFG),
                        'source':'Historical full-1359 ranking; no fresh retrieval timing claimed'}]
        if hit_path(q['question']).exists():
            assert json.loads(hit_path(q['question']).read_text(encoding='utf-8'))==pair
        else: api.atomic_json(hit_path(q['question']),pair)
        metrics=retrieval_metrics(ids,set(q['gold_chunk_ids']),ks=[1,3,5])
        metrics={k:v for k,v in metrics.items() if '@10' not in k}
        retrieval.append({'id':q['id'],'metrics':metrics,
             'context_recall_lexical':context_recall(q['evidence'],[h['text'] for h in pair[0]])})
    save_jsonl(retrieval,OUT/'retrieval_per_question.jsonl')
    api.atomic_json(OUT/'retrieval_summary.json',{'n':1359,
        'metrics':{k:float(np.mean([r['metrics'][k] for r in retrieval])) for k in retrieval[0]['metrics']},
        'rankings_sha256':api.sha(ranks_path),'gold_evidence':'Human-original evidence with heuristic chunk matching'})
    # Reuse only matching API answers, never three-way agreement or single-reference judges.
    old_out=PATHS.results_dir/'api_deepseek_flash_2026-11-02'
    old_manifest=json.loads((old_out/'input_manifest.json').read_text(encoding='utf-8'))
    compatible=all(old_manifest[k]==identity[k] for k in ['provider','base_url','model','retrieval','generation'])
    copied=0
    if compatible:
        by_id={q['id']:q for q in questions}
        for phase in ['generation','closed_book']:
            (OUT/phase).mkdir(exist_ok=True)
            for path in (old_out/phase).glob('*.json'):
                row=json.loads(path.read_text(encoding='utf-8'))
                if row['id'] not in by_id or (OUT/phase/path.name).exists(): continue
                q=by_id[row['id']]
                if phase=='generation':
                    assert row['query']==q['question']
                    assert row['hits']==hits_for(q['question'])[0]
                row.update(reference_scores(row['answer'],q))
                row['reused_from']=str(old_out.relative_to(PATHS.root)/phase/path.name)
                api.atomic_json(OUT/phase/path.name,row)
                copied+=1
    api.atomic_json(OUT/'data_manifest.json',{'n':1359,'n_human_groups':2,
        'question_type_counts':dict(Counter(q['question_type'] for q in questions)),
        'question_type_agreement_n':sum(q['question_type_human_agree'] for q in questions),
        'question_type_agreement':sum(q['question_type_human_agree'] for q in questions)/1359,
        'copied_compatible_api_outputs_this_prepare':copied,'contains_ai_annotations':False,
        'dataset_sha256':api.sha(DATA)})
    print(f'Prepared two-human dataset: {len(questions)} questions; reused {copied} compatible API outputs',flush=True)
    return questions


def structured(llm,prompt,validator):
    for attempt in range(2):
        value=llm.generate_json(prompt)
        try: return validator(value)
        except (ValueError,TypeError):
            if attempt: raise
            # A malformed cached response must not make retries repeat the same invalid JSON.
            llm._cache_path(prompt,0.,True).unlink(missing_ok=True)
    raise AssertionError('Unreachable')


def worker(phase,q,out,gate):
    llm=api.MeasuredLLM(out,gate)
    start=time.perf_counter()
    try:
        if phase=='generation':
            system=object.__new__(RAGSystem)
            system._llm=llm
            system.retrieve=lambda query,cfg:hits_for(query)
            result=system.answer(q['question'],api.RCFG,api.GCFG)
            result.update(reference_scores(result['answer'],q))
        elif phase=='closed_book':
            answer=closed_book_answer(llm,q['question'])
            result={'answer':answer,**reference_scores(answer,q)}
        elif phase=='agreement':
            variants=q['annotation_variants']
            prompt=('Compare the two supplied HUMAN answers to the question. '
                'AB must be agree (same essential answer, requested parts covered; compatible additional details allowed), '
                'partial (essential requested part missing), disagree (contradictory core or refusal versus an answer), '
                'or uncertain (cannot confidently determine). No target agreement rate. '
                'Do not rewrite or adjudicate the human answers. Compare their meanings, not token overlap. '
                'Return JSON with AB and a short reason.\n'+json.dumps({'question':q['question'],
                'A':variants['human_original']['answer'],'B':variants['human_reannotated']['answer']},ensure_ascii=False))
            result=structured(llm,prompt,validate_pair)
        else:
            row=json.loads((out/'generation'/(q['id']+'.json')).read_text(encoding='utf-8'))
            references=json.dumps(q['reference_answers'],ensure_ascii=False)
            prompt=JUDGE_PROMPT.format(question=q['question'],gold=references,
                      passages=format_passages(row['hits']),answer=row['answer'])
            prompt+=('\nThe gold field contains two human reference answers (deduplicated). Accept either '
                'reference only where consistent with the retrieved evidence; do not treat their concatenation '
                'as a single required answer. Additionally return reference_claims: a nonempty list of '
                '{"claim":"atomic essential fact needed to answer the question from the human references",'
                '"supported":true or false}, marking whether each fact is covered by retrieved passages. '
                'Merge duplicate facts; disregard irrelevant extra reference details. This assesses semantic '
                'evidence coverage separately from support for claims in the SYSTEM ANSWER.')
            def validate(raw):
                api.validate_judgment(raw,len(row['hits']))
                validate_reference_claims(raw)
                return raw
            raw=structured(llm,prompt,validate)
            claims=raw['claims']; reference_claims=raw['reference_claims']
            result={'raw_judgment':raw,
                'faithfulness':sum(c['supported'] for c in claims)/len(claims) if claims else 1.,
                'context_precision':context_precision(raw['passage_useful']),
                'context_relevance':sum(raw['passage_useful'])/len(row['hits']) if row['hits'] else 0.,
                'context_recall_semantic':sum(c['supported'] for c in reference_claims)/len(reference_claims),
                'context_recall_lexical':context_recall(q['evidence'],[h['text'] for h in row['hits']]),
                'llm_correctness':raw['correctness'],'noncommittal':raw['noncommittal']}
        return {'id':q['id'],'question_type':q['question_type'],**result,
                'wall_seconds':time.perf_counter()-start,'usage':llm.stats.as_dict(),'api_requests':llm.requests}
    finally: llm._http.close()


def read_phase(phase):
    return [json.loads(p.read_text(encoding='utf-8')) for p in sorted((OUT/phase).glob('*.json'))]


def summary(questions):
    data={phase:read_phase(phase) for phase in PHASES}
    result={'n_target':1359,'human_groups':2,'selection':'all aligned human questions',
            'model':config.LLM_MODEL,'counts':{p:len(rows) for p,rows in data.items()},
            'generation_quality':None,'closed_book_quality':None,'RAG_metrics':None,
            'human_answer_agreement':None,'performance':{},'monetary_cost':None,
            'cost_formula':'input_tokens/1e6*input_price_per_million + output_tokens/1e6*output_price_per_million',
            'limitations':['Same-model automated judge, not independent human adjudication.',
               'No AI annotations are included; the API compares the two human answers only.',
               'Existing 1359 questions also informed retrieval selection; not an untouched held-out test.',
               'Two human references retained; no fabricated adjudicated gold answers.',
               'Main bulk generation uses cached retrieval; fresh end-to-end measurements are reported separately.',
               'Semantic RAG metrics are custom judge metrics, not an execution of official RAGAS.',
               'Provider pricing is unknown; monetary cost cannot be computed honestly.']}
    for phase,key in [('generation','generation_quality'),('closed_book','closed_book_quality')]:
        rows=data[phase]
        if len(rows)==1359:
            result[key]={'primary_reference':{k:float(np.mean([r[k] for r in rows])) for k in ['EM','F1','ROUGE_L']},
                         'two_human_references':{k:float(np.mean([r['multi_reference'][k] for r in rows])) for k in ['EM','F1','ROUGE_L']}}
    judges=data['judge']
    if len(judges)==1359 and all('answer_relevance' in r for r in judges):
        result['RAG_metrics']={k:float(np.mean([r[k] for r in judges])) for k in
             ['faithfulness','answer_relevance','context_precision','context_relevance',
              'context_recall_semantic','context_recall_lexical','llm_correctness','noncommittal']}
    agreements=data['agreement']
    if len(agreements)==1359:
        counts=Counter(r['AB'] for r in agreements)
        result['human_answer_agreement']={'n':1359,'counts':dict(counts),
            'strict_agreement':counts['agree']/1359,'reaches_80_percent':counts['agree']/1359>=.8,
            'measurement':'API-assisted semantic equivalence of two human answers; not a third annotation'}
    save_jsonl([r for r in agreements if r['AB']!='agree'],OUT/'human_disagreements.jsonl')
    for phase,rows in data.items():
        if rows:
            result['performance'][phase]={'n':len(rows),
                'seconds_p50':float(np.median([r['wall_seconds'] for r in rows])),
                'seconds_p95':float(np.percentile([r['wall_seconds'] for r in rows],95)),
                'api_calls':sum(r['usage']['calls'] for r in rows),
                'prompt_tokens':sum(r['usage']['prompt_tokens'] for r in rows),
                'output_tokens':sum(r['usage']['output_tokens'] for r in rows),
                'reused_outputs_n':sum('reused_from' in r for r in rows)}
    api.atomic_json(OUT/'summary.json',result)
    # Export inspectable complete/partial per-question outputs without any annotation C.
    for phase,rows in data.items(): save_jsonl(rows,OUT/(phase+'_all.jsonl'))
    return result


def benchmark(questions):
    path=OUT/'fresh_end_to_end_benchmark.json'
    if path.exists(): return
    gate=api.Gate(config.LLM_RPM)
    llm=api.MeasuredLLM(OUT,gate)
    llm.generate=lambda *a,**kw: api.MeasuredLLM.generate(llm,*a,**{**kw,'use_cache':False})
    start=time.perf_counter(); system=RAGSystem(llm=llm); system.warmup()
    startup=time.perf_counter()-start
    # Two fixed questions/type for fresh end-to-end latency and scaling comparison.
    selected=[]
    for kind in sorted({q['question_type'] for q in questions}):
        selected.extend([q for q in questions if q['question_type']==kind][:2])
    retrieval=[]
    for q in selected:
        started=time.perf_counter(); system.retrieve(q['question'],api.RCFG)
        retrieval.append(time.perf_counter()-started)
    conditions=[]
    for workers in [1,2,4]:
        # Fresh retrieval is done for each request in its measured wall time; serialize GPU access.
        import threading
        retrieval_lock=threading.Lock()
        def task(q):
            model=api.MeasuredLLM(OUT,gate)
            model.generate=lambda *a,**kw: api.MeasuredLLM.generate(model,*a,**{**kw,'use_cache':False})
            local=object.__new__(RAGSystem); local._llm=model
            def retrieve(query,cfg):
                with retrieval_lock: return system.retrieve(query,cfg)
            local.retrieve=retrieve
            started=time.perf_counter()
            try:
                row=local.answer(q['question'],api.RCFG,api.GCFG)
                return {'id':q['id'],'seconds':time.perf_counter()-started,
                        'usage':model.stats.as_dict(),'answer':row['answer']}
            finally: model._http.close()
        started=time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool: rows=list(pool.map(task,selected))
        elapsed=time.perf_counter()-started
        conditions.append({'workers':workers,'n':len(rows),'elapsed_seconds':elapsed,
            'records_per_second':len(rows)/elapsed,
            'latency_p50':float(np.median([r['seconds'] for r in rows])),
            'latency_p95':float(np.percentile([r['seconds'] for r in rows],95)),'rows':rows})
        api.atomic_json(OUT/'benchmark_partial.json',{'conditions':conditions})
    import psutil
    result={'startup_seconds':startup,'fresh_retrieval_seconds_mean':float(np.mean(retrieval)),
            'process_rss_bytes':psutil.Process().memory_info().rss,'conditions':conditions,
            'rpm':config.LLM_RPM,'sample_ids':[q['id'] for q in selected],
            'description':'10 fixed questions, fresh model calls and retrieval, same 30 RPM cap, 1/2/4 workers; descriptive, no significance claim'}
    api.atomic_json(path,result)
    llm._http.close()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=['prepare','run','report'])
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--phase',choices=['all']+PHASES,default='all')
    args=ap.parse_args()
    if args.action=='run' and (OUT/'STOP_GENERATION').exists():
        raise RuntimeError('Generation was explicitly stopped by the user. Use scripts/15_q4_existing_only.py for offline analysis.')
    if args.action=='prepare': prepare(); return
    if args.action=='report':
        summary(load_jsonl(DATA)); build_report(); return
    questions=prepare()
    lock=OUT/'running.lock'
    with lock.open('x',encoding='utf-8') as f: f.write(str(os.getpid()))
    api.worker=worker
    gate=api.Gate(config.LLM_RPM)
    try:
        phases=PHASES if args.phase=='all' else [args.phase]
        selected=questions[:args.limit] if args.limit else questions
        for phase in phases:
            if phase=='judge':
                assert all((OUT/'generation'/(q['id']+'.json')).exists() for q in selected)
            api.run_phase(phase,selected,OUT,gate,4,targets=TARGETS)
            if phase=='judge': api.enrich_relevance(OUT,questions)
            summary(questions); build_report()
        if not args.limit and args.phase=='all':
            status=json.loads((OUT/'status.json').read_text(encoding='utf-8'))
            status['phase']='fresh_end_to_end_benchmark'
            api.atomic_json(OUT/'status.json',status)
            benchmark(questions)
        result=summary(questions); build_report()
        status=json.loads((OUT/'status.json').read_text(encoding='utf-8'))
        execution_complete=all(n==1359 for n in result['counts'].values()) and result['RAG_metrics'] is not None and (OUT/'fresh_end_to_end_benchmark.json').exists()
        agreement_met=bool(result['human_answer_agreement'] and result['human_answer_agreement']['reaches_80_percent'])
        status.update(phase='batch_finished' if args.limit else ('finished' if agreement_met else 'needs_human_agreement_review'),
                      execution_complete=execution_complete,complete=execution_complete and agreement_met)
        api.atomic_json(OUT/'status.json',status)
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    except Exception as error:
        status=json.loads((OUT/'status.json').read_text(encoding='utf-8')) if (OUT/'status.json').exists() else {}
        status.update(phase='stopped_error',complete=False,error_type=type(error).__name__)
        api.atomic_json(OUT/'status.json',status)
        summary(questions); build_report()
        raise
    finally: lock.unlink(missing_ok=True)


def build_report():
    path=PATHS.root.parent/'reports/build_q4_two_humans.py'
    if not path.exists(): return
    report_spec=importlib.util.spec_from_file_location('q4_report',path)
    module=importlib.util.module_from_spec(report_spec); report_spec.loader.exec_module(module)
    module.main()


if __name__=='__main__': main()
