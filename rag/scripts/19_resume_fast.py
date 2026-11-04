"""Resume the frozen experiment with an explicitly recorded scheduling-only override."""
from dataclasses import asdict,replace
from datetime import datetime,timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import threading
from rag.config import PATHS
from rag import config
from rag.formal.contract import ROOT,sha,write_json
import rag.formal.runner as runner


def faster_options(options,overrides=None):
    # Network scheduling changes never alter model/sampling or spending controls.
    overrides=overrides or {'rpm':60,'workers':6}
    if set(overrides)-{'rpm','workers','timeout'}:
        raise ValueError('Only scheduling and network timeout may be amended')
    updated=replace(options,**overrides)
    if updated.workers<1 or updated.rpm<1 or updated.timeout<=0:
        raise ValueError('Invalid runtime override')
    return updated


def service_backoff(journal):
    """Coordinate retries across workers when the upstream reports overload."""
    original=journal.record
    lock=threading.Lock()
    failures=0
    def record(entry):
        nonlocal failures
        delay=None
        with lock:
            if entry.get('status')=='failed' and entry.get('http_status') in [502,503,504]:
                failures+=1;delay=min(120,15*2**min(failures-1,3))
            elif entry.get('status')=='success':failures=0
        if delay:
            journal.defer(delay)
            print(f'Upstream {entry["http_status"]}: shared cooldown {delay}s; checkpoint retained.',flush=True)
        return original(entry)
    journal.record=record


class FastExperiment(runner.Experiment):
    def __init__(self,out=ROOT):
        super().__init__(out)
        amendment_path=out/'config/runtime_amendment.json'
        amendment=json.loads(amendment_path.read_text(encoding='utf-8'))
        if amendment['pipeline_version']!=self.version or amendment['dataset_sha256']!=self.dataset_sha:
            raise ValueError('Runtime amendment belongs to a different frozen experiment')
        if amendment['launcher_sha256']!=sha(Path(__file__)):
            raise ValueError('Runtime launcher differs from the recorded amendment')
        overrides={key:amendment['effective_options'][key] for key in amendment['only_changed_fields']}
        self.options=faster_options(self.options,overrides)
        self.journal.options=self.options
        service_backoff(self.journal)
        self.endpoint_override=amendment.get('endpoint_override')
        model_path=out/'config/model_config.json'
        model=json.loads(model_path.read_text(encoding='utf-8'))
        write_json(model_path,{**model,'options':asdict(self.options),
            'endpoint':self.endpoint_override or config.OPENAI_BASE_URL,
            'original_frozen_endpoint':config.OPENAI_BASE_URL,
            'runtime_amendment':'config/runtime_amendment.json',
            'original_frozen_options_preserved':True})
        write_json(out/'config/runtime_active.json',{'updated_at':datetime.now(timezone.utc).isoformat(),
            'pid':os.getpid(),'effective_options':asdict(self.options),
            'dataset_sha256':self.dataset_sha,'pipeline_version':self.version,
            'effective_endpoint':self.endpoint_override or config.OPENAI_BASE_URL,
            'scope':'Recorded runtime/endpoint amendment; original frozen manifest preserved. Raw calls retain their actual endpoint.'})

    def client(self,qid,code):
        llm=super().client(qid,code)
        if self.endpoint_override:llm.base=self.endpoint_override.rstrip('/')
        return llm


def main():
    # Constructor validates the original immutable manifest before applying scheduling changes.
    frozen=json.loads((ROOT/'config/frozen_experiment.json').read_text(encoding='utf-8'))
    # Validate original provenance and imports against the original endpoint; new requests
    # use the separately recorded override, which also changes their cache identities.
    config.OPENAI_BASE_URL=frozen['endpoint']
    original=frozen['options']
    mapping={'workers':'MAX_CONCURRENT_REQUESTS','rpm':'LLM_RPM','retries':'MAX_RETRIES',
        'timeout':'REQUEST_TIMEOUT','max_output_tokens':'MAX_OUTPUT_TOKENS','max_cost_usd':'MAX_COST_USD',
        'input_price':'INPUT_PRICE_PER_MTOK','output_price':'OUTPUT_PRICE_PER_MTOK','price_source':'API_PRICE_SOURCE'}
    for field,name in mapping.items():
        if original[field] is None or original[field]=='':os.environ.pop(name,None)
        else:os.environ[name]=str(original[field])
    runner.Experiment=FastExperiment
    script=PATHS.root/'scripts/16_formal_full.py'
    spec=importlib.util.spec_from_file_location('frozen_formal_entrypoint',script)
    entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
    sys.argv=[str(script),'run']
    entry.main()


if __name__=='__main__':main()
