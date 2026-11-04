"""Resume the approved six-group study, deliver Q4 first, then finish B/S/T."""
from dataclasses import asdict
from datetime import datetime,timezone
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import psutil
from rag import config
from rag.config import PATHS
from rag.formal.contract import sha,write_json
from rag.six.scope import CODES,matrix,check_matrix,original_matrix,install

OUT=PATHS.results_dir/'q4_q5_six_groups'
SOURCE=PATHS.results_dir/'q4_q5_formal_full'


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def scope_sources():
    paths=sorted((PATHS.root/'src/rag/six').glob('*.py'))
    paths.extend([Path(__file__),PATHS.root/'scripts/19_resume_fast.py'])
    return {str(p.relative_to(PATHS.root)):sha(p) for p in paths}


def initialize():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest_path=OUT/'config/scope_manifest.json'
    if manifest_path.exists():
        scope=json.loads(manifest_path.read_text(encoding='utf-8'))
        if scope['scope_code_sha256']!=scope_sources():raise RuntimeError('Frozen six-group adapter changed')
        return
    check_matrix(matrix())
    source_manifest=json.loads((SOURCE/'config/frozen_experiment.json').read_text(encoding='utf-8'))
    for relative,wanted in source_manifest['code_sha256'].items():
        if sha(PATHS.root/relative)!=wanted:raise RuntimeError('Original core changed: '+relative)
    for folder in ['data_validation','generation/records','generation/raw_calls','generation/imported_calls',
                   'generation/call_cache','evaluation/records','performance/profiles']:
        source=SOURCE/folder;target=OUT/folder
        if source.exists():shutil.copytree(source,target,dirs_exist_ok=True)
    (OUT/'generation').mkdir(exist_ok=True)
    shutil.copy2(SOURCE/'generation/api_calls.jsonl',OUT/'generation/api_calls.jsonl')
    (OUT/'config').mkdir(exist_ok=True)
    for name in ['model_config.json','generation_config.json','environment_versions.json','runtime_amendment.json']:
        shutil.copy2(SOURCE/'config'/name,OUT/'config'/name)
    (OUT/'provenance').mkdir(exist_ok=True)
    shutil.copy2(SOURCE/'config/frozen_experiment.json',OUT/'provenance/original_19_group_manifest.json')
    shutil.copytree(SOURCE/'config/runtime_history',OUT/'provenance/runtime_history',dirs_exist_ok=True)
    # Only the preserved 0/A/G configurations are eligible for reuse here.
    for c in set(original_matrix())-set(CODES):
        if (OUT/'generation/records'/c).exists():raise RuntimeError('Cancelled configuration records unexpectedly copied')
    write_json(OUT/'config/ablation_matrix.json',matrix())
    write_json(OUT/'config/generation_config.json',{'final_Q4_system':'G','Q4_comparison':['0','A','G'],
        'formal_scope':CODES,'n_questions_per_configuration':1359,'AI_reference_used':False})
    write_json(OUT/'config/scope_manifest.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'configuration_ids':CODES,'question_count':1359,'expected_configuration_questions':8154,
        'scope_code_sha256':scope_sources(),'source_directory':str(SOURCE),
        'dataset_sha256':source_manifest['dataset_sha256'],
        'preserved_core_pipeline_version':source_manifest['pipeline_version'],
        'reason':'User explicitly replaced nineteen-group plan with 0/A/B/G/S/T.',
        'reuse':'Unchanged 0/A/G definitions, records, raw calls and scoped caches; S/T have new configuration IDs.'})
    for relative in scope_sources():
        target=OUT/'config/scope_code_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(PATHS.root/relative,target)
    write_json(PATHS.results_dir/'active_experiment.json',{'directory':str(OUT),'configuration_ids':CODES,
        'launcher':'scripts/22_six_group_study.py','old_scope_superseded':True})


def run():
    initialize()
    source=json.loads((SOURCE/'config/frozen_experiment.json').read_text(encoding='utf-8'))
    config.OPENAI_BASE_URL=source['endpoint']
    mapping={'workers':'MAX_CONCURRENT_REQUESTS','rpm':'LLM_RPM','retries':'MAX_RETRIES','timeout':'REQUEST_TIMEOUT',
        'max_output_tokens':'MAX_OUTPUT_TOKENS','max_cost_usd':'MAX_COST_USD','input_price':'INPUT_PRICE_PER_MTOK',
        'output_price':'OUTPUT_PRICE_PER_MTOK','price_source':'API_PRICE_SOURCE'}
    for key,name in mapping.items():
        value=source['options'][key]
        if value is None or value=='':os.environ.pop(name,None)
        else:os.environ[name]=str(value)
    runtime=load_module('six_runtime_adapter',PATHS.root/'scripts/19_resume_fast.py')
    install()
    from rag.formal.runner import Experiment as OriginalExperiment
    class SixExperiment(runtime.FastExperiment):
        def load_system(self):
            current=self.configs
            try:
                self.configs=original_matrix()
                return super().load_system()
            finally:self.configs=current
        def status(self,phase,code=None):
            super().status(phase,code)
            state=json.loads(self.status_path.read_text(encoding='utf-8'))
            state.update(expected_configuration_questions=8154,configuration_ids=CODES,scope='six_groups')
            write_json(self.status_path,state)
    lock=OUT/'running.lock'
    if lock.exists():
        pid=int(lock.read_text().strip())
        if psutil.pid_exists(pid):raise RuntimeError('Six-group experiment already running')
        lock.rename(lock.with_name('stale_'+str(pid)+'.lock'))
    lock.write_text(str(os.getpid()),encoding='utf-8')
    try:
        experiment=SixExperiment(OUT)
        from rag.formal.performance import benchmark
        from rag.formal.analysis import analyze
        from rag.six.verification import verify
        experiment.run(codes=['0','A','G'])
        benchmark(experiment,codes=['0','A','G'])
        analyze(OUT)
        q4=verify(OUT,encoder=experiment.system.get_encoder(),scope='Q4')
        if not q4['automated_complete']:raise RuntimeError('Q4 audit failed; inspect reports/Q4_completion_audit.json')
        print('Q4 audit completed. Running only B, S and T.',flush=True)
        experiment.run(codes=['B','S','T'])
        benchmark(experiment,codes=CODES)
        analyze(OUT)
        result=verify(OUT,encoder=experiment.system.get_encoder())
        state=json.loads(experiment.status_path.read_text(encoding='utf-8'))
        state.update(phase='automated_complete_pending_human_review' if result['automated_complete'] else 'needs_recovery',
            automated_complete=result['automated_complete'],complete=result['all_requirements_complete'])
        write_json(experiment.status_path,state)
        if not result['automated_complete']:raise RuntimeError('Six-group audit failed; inspect completion audit')
    except Exception as exc:
        write_json(OUT/'last_stop.json',{'exception_type':type(exc).__name__,
            'reason':str(exc).replace(os.environ.get('OPENAI_API_KEY','__NONE__'),'[REDACTED]'),'complete':False})
        raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run'],default='run',nargs='?')
    if parser.parse_args().action=='prepare':initialize()
    else:run()
