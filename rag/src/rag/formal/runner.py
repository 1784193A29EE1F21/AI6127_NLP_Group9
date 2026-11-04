from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timezone
import json
import os
import shutil
import threading
import time
import numpy as np
from rag import config
from rag.config import PATHS
from rag.corpus import load_jsonl
from rag.generator import closed_book_answer,generate_answer,self_check
from rag.pipeline import RAGSystem,RetrievalConfig,GenerationConfig
from .client import FormalLLM,Journal,Options,BudgetLimit,PermanentAPIError
from .contract import ROOT,DATA,Q4,digest,sha,write_json,matrix
from .evaluation import qa_scores,citation_validity,judge_prompt,validate_judge,judge_metrics
from .migrate import import_compatible


class LockedModel:
    def __init__(self,model,lock): self.model=model; self.lock=lock
    def __getattr__(self,name):
        attr=getattr(self.model,name)
        if not callable(attr): return attr
        def call(*args,**kwargs):
            with self.lock: return attr(*args,**kwargs)
        return call


class Experiment:
    def __init__(self,out=ROOT):
        self.out=out; self.options=Options.from_env(); self.questions=load_jsonl(DATA)
        self.by_id={q['id']:q for q in self.questions}; self.dataset_sha=sha(DATA)
        self.configs=matrix(); self.system=None; self.rankings={}; self.gpu_lock=threading.Lock()
        self.journal=Journal(out,self.options)
        files=[PATHS.root/'src/rag'/p for p in ['pipeline.py','generator.py','query.py','metrics.py','llm.py',
             'bm25.py','dense.py','fusion_rerank.py','text.py','config.py','human_pair.py']]
        files+=sorted((PATHS.root/'src/rag/formal').glob('*.py'))
        files.append(PATHS.root/'scripts/16_formal_full.py')
        self.code={str(p.relative_to(PATHS.root)):sha(p) for p in files}
        self.version=digest(self.code)
        manifest={'dataset_sha256':self.dataset_sha,'code_sha256':self.code,'pipeline_version':self.version,
            'model':config.LLM_MODEL,'provider':config.LLM_PROVIDER,'endpoint':config.OPENAI_BASE_URL,
            'parameters':{'temperature':0.,'top_p':None,'max_tokens':self.options.max_output_tokens},
            'embedding_model':json.loads((PATHS.index_dir/'dense_meta.json').read_text(encoding='utf-8'))['model_name'],
            'rerank_model':config.RERANK_MODEL,
            'requirements_sha256':sha(PATHS.root.parent/'task1.md'),
            'options':asdict(self.options),'matrix':self.configs}
        from .preflight import environment_versions
        manifest['library_versions']=environment_versions()
        path=out/'config/frozen_experiment.json'
        if path.exists():
            if json.loads(path.read_text(encoding='utf-8'))!=manifest:
                raise ValueError('Frozen code/model/configuration changed; do not mix experimental versions')
        else:
            write_json(path,manifest)
            snapshot=out/'config/code_snapshot'
            for p in files:
                dest=snapshot/p.relative_to(PATHS.root); dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,dest)
        self.status_path=out/'generation/execution_status.json'
        self.counters=None
        self.imported_existing=False

    def load_system(self):
        if self.system is not None: return
        self.system=RAGSystem(); self.system.warmup()
        self.system.bm25.rows(); self.system.bm25.inv_vocab()
        self.system.encoder=LockedModel(self.system.get_encoder(),self.gpu_lock)
        self.system._reranker=LockedModel(self.system.get_reranker(),self.gpu_lock)
        self.chunks={c.chunk_id:self.system._hit(i) for i,c in enumerate(self.system.chunks)}
        mappings={'A':'Hybrid_RRF','B':'Hybrid_RRF_plus_rerank','G':'Hybrid_RRF_plus_rerank',
             'H':'Hybrid_RRF_plus_PRF_plus_rerank','J':'Hybrid_RRF_plus_rerank_plus_diversity'}
        legacy=PATHS.results_dir/'ablation_full_2026-11-02'
        for code,name in mappings.items():
            path=legacy/(name+'.trec.jsonl')
            self.rankings[code]={r['id']:r['ranked'][:5] for r in load_jsonl(path)}
        # Verify top-k invariance and actual execution against three fixed questions/config.
        checks=[]
        for code in ['A','B','H','J']:
            cfg=RetrievalConfig(**self.configs[code]['retrieval'])
            for q in self.questions[:3]:
                hits,_=self.system.retrieve(q['question'],cfg)
                ok=[h['chunk_id'] for h in hits]==self.rankings[code][q['id']]
                checks.append({'config':code,'id':q['id'],'matches':ok})
                if not ok: raise ValueError('Historical retrieval is not reproducible; refuse unverified reuse')
        write_json(self.out/'data_validation/retrieval_reuse_verification.json',{
            'checks':checks,'n':len(checks),'scope':'Sample execution plus complete input/index/code hash checks',
            'full_retrieval_not_rerun':True})

    def client(self,qid,code):
        return FormalLLM(self.out,self.journal,self.dataset_sha,qid,code,self.version)

    def record_path(self,code,qid): return self.out/'generation/records'/code/(qid+'.json')
    def evaluation_path(self,code,qid,variant='final'):
        return self.out/'evaluation/records'/code/variant/(qid+'.json')

    def refresh_counts(self):
        counts={}
        for c in self.configs:
            records=[json.loads(p.read_text(encoding='utf-8')) for p in (self.out/'generation/records'/c).glob('*.json')]
            success=sum(r.get('execution_status')=='success' for r in records)
            failed=sum(r.get('execution_status')=='failed' for r in records)
            counts[c]={'success':success,'failed':failed,'not_run':1359-len(records),'total':1359,
                'coverage':success/1359,'judged':sum(json.loads(p.read_text(encoding='utf-8')).get('execution_status')=='success'
                   for p in (self.out/'evaluation/records'/c/'final').glob('*.json'))}
        self.counters=counts

    def status(self,phase,code=None):
        if self.counters is None: self.refresh_counts()
        counts=deepcopy(self.counters)
        for value in counts.values(): value['coverage']=value['success']/1359
        write_json(self.status_path,{'phase':phase,'active_configuration':code,'updated_at':datetime.now(timezone.utc).isoformat(),
            'model':config.LLM_MODEL,'dataset_sha256':self.dataset_sha,'pipeline_version':self.version,
            'counts':counts,'expected_configuration_questions':25821,'complete':False,
            'independent_human_review':'pending','verified_cost_usd_so_far':self.journal.spent if self.options.price_source else None})

    def generate_one(self,code,q,profile=None):
        llm=self.client(q['id'],code); started=time.perf_counter(); traces=[]
        if profile:
            llm.scope_prefix='benchmark_'+profile+'/'
            llm.cache_allowed=False
        try:
            if code=='0':
                answer=closed_book_answer(llm,q['question'])
                result={'query':q['question'],'answer':answer,'draft_answer':answer,'hits':[],
                    'timings':{'total':time.perf_counter()-started},'iterative_rounds':[]}
            else:
                cfg=RetrievalConfig(**self.configs[code]['retrieval'])
                gen=GenerationConfig(**self.configs[code]['generation'])
                system=object.__new__(RAGSystem); system.__dict__=self.system.__dict__.copy(); system._llm=llm
                def retrieve(query,r=None):
                    r=r or cfg; start=time.perf_counter()
                    if not profile and not traces and code in self.rankings and query==q['question'] and r==cfg:
                        hits=[deepcopy(self.chunks[cid]) for cid in self.rankings[code][q['id']]]
                        for rank,h in enumerate(hits,1): h['rank']=rank
                        elapsed=time.perf_counter()-start
                        info={'timings':{'retrieval_cache_lookup':elapsed,'retrieval_total':elapsed},
                            'config':asdict(r),'cache_hit':True,'origin':'verified historical full-1359 ranking'}
                    else:
                        hits,info=RAGSystem.retrieve(system,query,r)
                        info['cache_hit']=False
                    traces.append({'query':query,'retrieval':asdict(r),'hits':deepcopy(hits),
                                   'info':deepcopy(info),'wall_seconds':time.perf_counter()-start})
                    return hits,info
                system.retrieve=retrieve
                result=system.answer(q['question'],cfg,gen)
                online_refs=len(llm.used)
                if gen.iterative and not profile:
                    before=result['draft_answer']
                    initial_ids={h['chunk_id'] for h in traces[0]['hits']}
                    added_ids=[h['chunk_id'] for h in result['hits'] if h['chunk_id'] not in initial_ids]
                    if added_ids:
                        llm.scope_prefix='before_iteration/'
                        before=generate_answer(llm,q['question'],traces[0]['hits'],gen.style)
                        result['draft_answer_before_iteration']=before
                        if gen.self_check:
                            check=self_check(llm,before,traces[0]['hits'],revise=True)
                            result['self_check_before_iteration']=check
                            before=check['revised_answer']
                        llm.scope_prefix=''
                    elif gen.self_check:
                        before=result['answer']
                    result['answer_before_iteration']=before
                    result['initial_retrieved_chunks']=[h['chunk_id'] for h in traces[0]['hits']]
                    result['additional_retrieved_chunks']=[h['chunk_id'] for h in result['hits']
                        if h['chunk_id'] not in result['initial_retrieved_chunks']]
                result['online_call_count']=online_refs
            online_seconds=result['timings']['total']
            if llm.stats.cache_hits or any(t['info'].get('cache_hit') for t in traces): online_seconds=None
            return {'id':q['id'],'question_id':q['id'],'configuration_id':code,'question_type':q['question_type'],
                'original_question':q['question'],'canonical_reference':q['answer'],
                'alternative_human_reference':q['annotation_variants']['human_reannotated']['answer'],
                'execution_status':'success','dataset_sha256':self.dataset_sha,'pipeline_version':self.version,
                'model':config.LLM_MODEL,'parameters':{'temperature':0.,'top_p':None,'max_tokens':self.options.max_output_tokens},
                **result,**qa_scores(result['answer'],q),**citation_validity(result['answer'],result['hits'],code=='0'),
                'retrieval_trace':traces,'call_refs':llm.used,'usage':llm.stats.as_dict(),
                'online_end_to_end_seconds':online_seconds,'bulk_wall_seconds':time.perf_counter()-started}
        finally: llm.close()

    def evaluate_one(self,code,q,variant='final'):
        row=json.loads(self.record_path(code,q['id']).read_text(encoding='utf-8'))
        if row['execution_status']!='success': raise ValueError('Cannot judge a failed generation')
        if variant=='iteration_resolution': return self.resolve_iteration(code,q,row)
        if variant=='final': answer,hits=row['answer'],row.get('hits',[])
        elif variant=='draft': answer,hits=row['draft_answer'],row['hits']
        else: answer,hits=row['answer_before_iteration'],row['retrieval_trace'][0]['hits']
        final_path=self.evaluation_path(code,q['id'])
        if variant!='final' and final_path.exists() and answer==row['answer'] and hits==row['hits']:
            result=json.loads(final_path.read_text(encoding='utf-8'))
            return {**result,'variant':variant,'reused_final_judgment':'Same exact answer and context; no new judge request'}
        llm=self.client(q['id'],code); llm.stage_override='judge_'+variant
        llm.validators['judge_'+variant]=lambda raw:validate_judge(raw,len(hits),answer)
        started=time.perf_counter()
        try:
            raw=llm.generate_json(judge_prompt(q,answer,hits))
            result=judge_metrics(raw,q,answer,hits)
            if raw['noncommittal']: proxy=0.
            elif raw['generated_questions']:
                vectors=self.system.get_encoder().encode_queries([q['question']]+raw['generated_questions'])
                proxy=float(np.mean(vectors[1:]@vectors[0]))
            else: proxy=None
            result['answer_relevance_reverse_question_proxy']=proxy
            return {'id':q['id'],'question_id':q['id'],'configuration_id':code,'variant':variant,
                'question_type':q['question_type'],'execution_status':'success','raw_judgment':raw,
                **result,**qa_scores(answer,q),'call_refs':llm.used,'usage':llm.stats.as_dict(),
                'wall_seconds':time.perf_counter()-started,'dataset_sha256':self.dataset_sha,'pipeline_version':self.version}
        finally: llm.close()

    def resolve_iteration(self,code,q,row):
        initial=row['retrieval_trace'][0]['hits']; initial_ids={h['chunk_id'] for h in initial}
        added=[h for h in row['hits'] if h['chunk_id'] not in initial_ids]
        common={'id':q['id'],'configuration_id':code,'variant':'iteration_resolution',
                'execution_status':'success','dataset_sha256':self.dataset_sha,'pipeline_version':self.version}
        if not added:
            return {**common,'resolved':False,'reason':'No distinct additional passages were added.',
                'source':'deterministic_trace_check','supporting_added_passage_numbers':[],'call_refs':[]}
        from rag.generator import format_passages
        missing=(row.get('iterative_rounds') or [{}])[0].get('missing','')
        prompt=('You are evaluating whether follow-up retrieval fills an actual stated evidence gap. '
            'Do not assume the follow-up succeeds. Use only the passages. Return JSON: '
            '{"resolved":true/false/null,"reason":"explanation","supporting_added_passage_numbers":[1,...]}. '
            'null means uncertain. Passage numbers refer to ADDED PASSAGES only.\n\n'
            f"QUESTION: {q['question']}\nSTATED GAP: {missing}\n\nINITIAL PASSAGES:\n{format_passages(initial)}\n\n"
            f"ADDED PASSAGES:\n{format_passages(added)}")
        def validate(raw):
            if not isinstance(raw,dict) or 'resolved' not in raw or (raw['resolved'] is not None and type(raw['resolved']) is not bool):
                raise ValueError('Invalid iterative evidence resolution')
            if not isinstance(raw.get('reason'),str) or not raw['reason'].strip(): raise ValueError('Missing resolution explanation')
            ids=raw.get('supporting_added_passage_numbers')
            if not isinstance(ids,list) or any(type(n) is not int or n<1 or n>len(added) for n in ids):
                raise ValueError('Invalid added-passage attribution')
            if raw['resolved'] is True and not ids: raise ValueError('Resolved gap needs actual additional evidence attribution')
        llm=self.client(q['id'],code); llm.stage_override='iteration_resolution'
        llm.validators['iteration_resolution']=validate
        try:
            raw=llm.generate_json(prompt)
            return {**raw,**common,'source':'anonymous_model_assessment','call_refs':llm.used,'usage':llm.stats.as_dict()}
        finally: llm.close()

    def run_jobs(self,jobs,phase,worker,path_for):
        pending=[]
        for code,q,variant in jobs:
            path=path_for(code,q['id'],variant)
            if not path.exists() or json.loads(path.read_text(encoding='utf-8')).get('execution_status')!='success':
                pending.append((code,q,variant))
        for attempt in range(2):
            if not pending: break
            retry=[]; iterator=iter(pending)
            with ThreadPoolExecutor(max_workers=self.options.workers) as pool:
                running={}
                def submit():
                    task=next(iterator,None)
                    if task:
                        if shutil.disk_usage(self.out).free<1024**3: raise BudgetLimit('Less than 1 GiB free; stop before filling disk')
                        code,q,variant=task
                        running[pool.submit(worker,code,q,variant)]=task
                for _ in range(self.options.workers): submit()
                completed=0
                while running:
                    done,_=wait(running,return_when=FIRST_COMPLETED)
                    for future in done:
                        code,q,variant=running.pop(future); path=path_for(code,q['id'],variant)
                        previous=json.loads(path.read_text(encoding='utf-8')).get('execution_status') if path.exists() else None
                        try:
                            record=future.result(); write_json(path,record)
                            if self.counters is not None:
                                if phase=='generation':
                                    if previous=='failed': self.counters[code]['failed']-=1
                                    elif previous is None: self.counters[code]['not_run']-=1
                                    self.counters[code]['success']+=1
                                elif variant=='final' and previous!='success': self.counters[code]['judged']+=1
                            print(f'{phase} {code} {variant} {q["id"]}: saved',flush=True)
                        except (BudgetLimit,PermanentAPIError): raise
                        except Exception as error:
                            reason=str(error).replace(os.environ.get('OPENAI_API_KEY','__NONE__'),'[REDACTED]')
                            write_json(path,{'id':q['id'],'question_id':q['id'],'configuration_id':code,
                                'variant':variant,'execution_status':'failed','attempt_pass':attempt+1,
                                'error_type':type(error).__name__,'error_details':reason,'dataset_sha256':self.dataset_sha,
                                'pipeline_version':self.version,'not_scored':True})
                            retry.append((code,q,variant))
                            if phase=='generation' and self.counters is not None and previous is None:
                                self.counters[code]['not_run']-=1; self.counters[code]['failed']+=1
                            print(f'{phase} {code} {q["id"]}: failed ({type(error).__name__})',flush=True)
                        completed+=1
                        if completed%10==0: self.status(phase,code)
                        submit()
            pending=retry
        self.status(phase)

    def run(self,codes=None,limit=0):
        self.load_system()
        questions=self.questions[:limit] if limit else self.questions
        codes=codes or list(self.configs)
        if not limit and not self.imported_existing:
            report=import_compatible(self.out,self.questions,self.options,self.version)
            print(f'Verified imports: {report["imported"]}',flush=True)
            self.refresh_counts()
            self.imported_existing=True
        for code in codes:
            self.status('generation',code)
            jobs=[(code,q,'final') for q in questions]
            self.run_jobs(jobs,'generation',lambda c,q,v:self.generate_one(c,q),lambda c,i,v:self.record_path(c,i))
        # No reference-bearing judge requests occur until all selected generation groups finish.
        for code in codes:
            jobs=[]; extra=[]
            for q in questions:
                row=json.loads(self.record_path(code,q['id']).read_text(encoding='utf-8'))
                if row['execution_status']!='success': continue
                jobs.append((code,q,'final'))
                if code!='0' and self.configs[code]['generation']['self_check']: extra.append((code,q,'draft'))
                if code!='0' and self.configs[code]['generation']['iterative']: extra.append((code,q,'before_iteration'))
                if code!='0' and self.configs[code]['generation']['iterative'] and len(row.get('retrieval_trace',[]))>1:
                    extra.append((code,q,'iteration_resolution'))
            self.run_jobs(jobs,'evaluation',self.evaluate_one,self.evaluation_path)
            self.run_jobs(extra,'evaluation_intermediates',self.evaluate_one,self.evaluation_path)
        self.status('smoke_finished' if limit else 'generation_and_judging_finished')
