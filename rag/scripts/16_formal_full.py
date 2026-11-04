"""Formal full-data Q4/Q5 entrypoint; dry-run never contacts a model endpoint."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import psutil
from rag.config import PATHS
from rag.formal.contract import ROOT,Q4,write_json


def active_q4():
    lock=Q4/'running.lock'
    if not lock.exists(): return False
    pid=int(lock.read_text(encoding='utf-8').strip())
    try:
        process=psutil.Process(pid)
        if process.create_time()>lock.stat().st_mtime+2: return False
        return process.is_running() and process.name().lower().startswith('python')
    except psutil.NoSuchProcess: return False


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=['preflight','dry-run','smoke','run','analyze','validate'])
    ap.add_argument('--out',type=Path,default=ROOT)
    ap.add_argument('--smoke-n',type=int,default=2)
    args=ap.parse_args()
    out=args.out if args.out.is_absolute() else PATHS.root/args.out
    experiment_root=out
    if args.action in ['preflight','dry-run']:
        from rag.formal.preflight import run
        result=run(out)
        print(json.dumps({'action':args.action,'n':result['dataset']['n'],'configurations':19,
            'new_API_calls':0,'independent_human_review':'pending','estimate':result['cost_estimate']},ensure_ascii=False,indent=2))
        return
    if args.action=='analyze':
        from rag.formal.analysis import analyze
        analyze(out); return
    if args.action=='validate':
        from rag.formal.analysis import audit
        print(json.dumps(audit(out),ensure_ascii=False,indent=2)); return
    if active_q4(): raise RuntimeError('Current Q4 process is still live. Wait for it; do not duplicate model traffic.')
    from rag.formal.preflight import run as preflight
    from rag.formal.runner import Experiment
    if args.action=='smoke':
        from rag.formal.contract import digest,sha
        from rag.formal.client import Options
        from dataclasses import asdict
        sources=sorted((PATHS.root/'src/rag/formal').glob('*.py'))
        tag=digest({'code':{p.name:sha(p) for p in sources},'runtime_options':asdict(Options.from_env())})[:10]
        if Options.from_env().max_cost_usd is None:
            out=out.with_name(out.name+'_smoke_'+tag)
    preflight(out)
    lock=out/'running.lock'
    if lock.exists():
        prior=int(lock.read_text(encoding='utf-8').strip())
        if psutil.pid_exists(prior): raise RuntimeError('Another formal experiment process may still be live')
        lock.rename(lock.with_name('stale_'+str(prior)+'.lock'))
    with lock.open('x',encoding='utf-8') as f: f.write(str(os.getpid()))
    try:
        experiment=Experiment(out)
        if args.action=='smoke':
            import time
            smoke_started=time.perf_counter()
            # Two real questions exercising both ordinary and missing-evidence paths.
            ids=['wiki_00_01','wiki_89_15']
            if args.smoke_n<2: raise ValueError('Smoke requires at least two real questions')
            for kind in ['why','cause_effect','factual','definition','comparison']:
                for q in experiment.questions:
                    if len(ids)>=args.smoke_n: break
                    if q['question_type']==kind and q['id'] not in ids:
                        ids.append(q['id']); break
            for q in experiment.questions:
                if len(ids)>=args.smoke_n:break
                if q['id'] not in ids:ids.append(q['id'])
            experiment.questions=[experiment.by_id[qid] for qid in ids]
            experiment.run(codes=['0','A','G','F','K'],limit=len(ids))
            for code in ['0','A','G','F','K']:
                for qid in ids:
                    for path in [experiment.record_path(code,qid),experiment.evaluation_path(code,qid)]:
                        if not path.exists() or json.loads(path.read_text(encoding='utf-8')).get('execution_status')!='success':
                            raise RuntimeError('Smoke execution contains a failed or missing generation/judgment')
            elapsed=time.perf_counter()-smoke_started
            raw_files=list((out/'generation/raw_calls').glob('*.json'))
            raw=[json.loads(p.read_text(encoding='utf-8')) for p in raw_files]
            result={'status':'passed','n_questions':len(ids),'configurations':['0','A','G','F','K'],
                'elapsed_seconds':elapsed,'HTTP_attempts':len(raw),'requests_per_second':len(raw)/elapsed,
                'http_429':sum(r.get('http_status')==429 for r in raw),'failed_attempts':sum(r.get('status')=='failed' for r in raw),
                'rpm':experiment.options.rpm,'workers':experiment.options.workers,
                'formal_full_results':False,'warning':'Preflight only; not a full-data result'}
            write_json(out/'SMOKE_RESULT.json',result)
            write_json(experiment_root/'latest_smoke_result.json',{**result,'output_directory':str(out)})
        else:
            from rag.formal.performance import benchmark
            from rag.formal.analysis import analyze,audit
            experiment.run(codes=['0','A','G'])
            benchmark(experiment,codes=['0','A','G'])
            analyze(out)
            q4_audit=audit(out,encoder=experiment.system.get_encoder(),scope='Q4')
            if not q4_audit['automated_complete']:
                raise RuntimeError('Q4 audit did not pass; do not release a misleading final writing bundle')
            print('Q4 audit completed; continuing remaining Q5 configurations.',flush=True)
            experiment.run()
            benchmark(experiment)
            analyze(out)
            result=audit(out,encoder=experiment.system.get_encoder())
            write_json(out/'reports/completion_audit.json',result)
            state=json.loads(experiment.status_path.read_text(encoding='utf-8'))
            state.update(phase='complete' if result['all_requirements_complete'] else
                        ('automated_complete_pending_human_review' if result['automated_complete'] else 'needs_recovery'),
                         complete=result['all_requirements_complete'],automated_complete=result['automated_complete'])
            write_json(experiment.status_path,state)
            if not result['automated_complete']:
                raise RuntimeError('Formal audit found incomplete/invalid automatic results; inspect completion_audit.json and resume')
    except Exception as error:
        write_json(out/'last_stop.json',{'exception_type':type(error).__name__,
            'reason':str(error).replace(os.environ.get('OPENAI_API_KEY','__NONE__'),'[REDACTED]'),
            'complete':False})
        raise
    finally: lock.unlink(missing_ok=True)


if __name__=='__main__': main()
