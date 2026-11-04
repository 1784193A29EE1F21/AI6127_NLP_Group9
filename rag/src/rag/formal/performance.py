"""Fresh-call, fresh-retrieval profiles kept separate from formal quality predictions."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
import random
import time
import numpy as np
from .contract import write_json


def benchmark(experiment,codes=None):
    rng=random.Random(42); selected=[]
    for kind in sorted({q['question_type'] for q in experiment.questions}):
        selected.extend(rng.sample([q for q in experiment.questions if q['question_type']==kind],2))
    conditions=[(c,4) for c in (codes or ['0','A','B','D','E','F','G','K'])]+[('G',1),('G',2)]
    result=[]
    for code,workers in conditions:
        folder=experiment.out/'performance/profiles'/code/('workers_'+str(workers))
        folder.mkdir(parents=True,exist_ok=True)
        complete=folder/'condition.json'
        if complete.exists():
            result.append(json.loads(complete.read_text(encoding='utf-8'))); continue
        before_calls=len(list((experiment.out/'generation/raw_calls').glob('*.json')))
        start=time.perf_counter()
        def work(q):
            path=folder/(q['id']+'.json')
            # A partial interrupted condition is restarted in full to retain a measured interval.
            row=experiment.generate_one(code,q,profile='workers_'+str(workers))
            row['quality_dataset_record']=False
            write_json(path,row)
            return row
        experiment.status('fresh_performance_benchmark',code)
        with ThreadPoolExecutor(max_workers=workers) as pool: rows=list(pool.map(work,selected))
        elapsed=time.perf_counter()-start
        latencies=[r['online_end_to_end_seconds'] for r in rows]
        item={'configuration':code,'workers':workers,'n':len(rows),'ids':[q['id'] for q in selected],
            'elapsed_seconds':elapsed,'records_per_second':len(rows)/elapsed,
            'latency_mean':float(np.mean(latencies)),
            **{f'latency_p{p}':float(np.percentile(latencies,p)) for p in [50,90,95,99]},
            'new_call_records':len(list((experiment.out/'generation/raw_calls').glob('*.json')))-before_calls,
            'prompt_tokens':sum(r['usage']['prompt_tokens'] for r in rows),
            'output_tokens':sum(r['usage']['output_tokens'] for r in rows),
            'dataset_sha256':experiment.dataset_sha,'pipeline_version':experiment.version,
            'created_at':datetime.now(timezone.utc).isoformat(),
            'description':'Fresh API calls and retrieval, no historical ranking cache; sample profile, not 1359-question quality score'}
        write_json(complete,item); result.append(item)
        write_json(experiment.out/'performance/fresh_profiles.json',{'conditions':result,'sample_seed':42,
            'rate_limit_rpm':experiment.options.rpm,'order_randomized':False,
            'limitations':'Fixed condition order and a small balanced sample; descriptive scaling under the same RPM cap.'})
    write_json(experiment.out/'performance/fresh_profiles.json',{'conditions':result,'sample_seed':42,
        'rate_limit_rpm':experiment.options.rpm,'order_randomized':False,
        'limitations':'Fixed sample and nonrandom condition order; records retain their actual timestamps.'})
    return result
