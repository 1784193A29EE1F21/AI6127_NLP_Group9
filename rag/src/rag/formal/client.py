"""Stateless API client with full request provenance and conservative budget guards."""
from __future__ import annotations
from dataclasses import dataclass,asdict
from datetime import datetime,timezone
import json
import math
import os
import re
import threading
import time
import uuid
import httpx
from rag import config
from rag.llm import LLMStats,parse_json
from .contract import cache_identity,digest,write_json


class BudgetLimit(RuntimeError): pass
class PermanentAPIError(RuntimeError): pass


@dataclass(frozen=True)
class Options:
    workers:int=4
    retries:int=3
    timeout:float=120.
    rpm:int=30
    max_output_tokens:int|None=None
    max_cost_usd:float|None=None
    input_price:float|None=None
    output_price:float|None=None
    price_source:str=''

    @classmethod
    def from_env(cls):
        def optional(name,fn): return fn(os.environ[name]) if os.environ.get(name,'').strip() else None
        item=cls(workers=int(os.environ.get('MAX_CONCURRENT_REQUESTS','4')),
            retries=int(os.environ.get('MAX_RETRIES','3')),timeout=float(os.environ.get('REQUEST_TIMEOUT','120')),
            rpm=int(os.environ.get('LLM_RPM',str(config.LLM_RPM or 30))),
            max_output_tokens=optional('MAX_OUTPUT_TOKENS',int),max_cost_usd=optional('MAX_COST_USD',float),
            input_price=optional('INPUT_PRICE_PER_MTOK',float),output_price=optional('OUTPUT_PRICE_PER_MTOK',float),
            price_source=os.environ.get('API_PRICE_SOURCE',''))
        if item.workers<1 or item.retries<1 or not math.isfinite(item.timeout) or item.timeout<=0 or item.rpm<0:
            raise ValueError('Invalid API concurrency/retry/timeout/rpm settings')
        if item.max_output_tokens is not None and item.max_output_tokens<1: raise ValueError('Invalid MAX_OUTPUT_TOKENS')
        for number in [item.max_cost_usd,item.input_price,item.output_price]:
            if number is not None and (not math.isfinite(number) or number<0): raise ValueError('Invalid API cost setting')
        if item.max_cost_usd is not None and (item.max_output_tokens is None or item.input_price is None or
                item.output_price is None or not item.price_source.strip()):
            raise BudgetLimit('MAX_COST_USD requires verified prices/source and MAX_OUTPUT_TOKENS; refusing unbounded cost')
        return item


def stage_for(prompt):
    if prompt.startswith('You rewrite user questions'): return 'rewrite'
    if prompt.startswith('Write a short, factual Wikipedia'): return 'hyde'
    if prompt.startswith('You are a strict fact-checker'): return 'self_check'
    if prompt.startswith('Rewrite the ANSWER'): return 'revision'
    if '"follow_up_query"' in prompt and prompt.startswith('You answer questions'): return 'evidence_assessment'
    if prompt.startswith('You are evaluating'): return 'judge'
    if prompt.startswith('Compare the two'): return 'annotation_assistance'
    if prompt.startswith('Answer the question.'): return 'closed_book'
    return 'answer'


def context_text(prompt):
    match=re.search(r'PASSAGES:\n(.*?)(?:\n\nQUESTION:|\n\nANSWER:|\n\nSYSTEM ANSWER:)',prompt,re.S)
    return match.group(1) if match else (prompt.split('PASSAGES:\n',1)[1] if 'PASSAGES:\n' in prompt else '')


def check_json_stage(data,stage):
    if not isinstance(data,dict): raise ValueError('Structured response must be a JSON object')
    if stage=='rewrite':
        if not isinstance(data.get('english_query'),str) or not data['english_query'].strip(): raise ValueError('Missing rewritten query')
        if not isinstance(data.get('keywords'),list) or not data['keywords'] or any(not isinstance(k,str) for k in data['keywords']):
            raise ValueError('Invalid rewrite keywords')
    if stage=='evidence_assessment':
        if type(data.get('sufficient')) is not bool or any(not isinstance(data.get(k),str) for k in ['missing','follow_up_query']):
            raise ValueError('Invalid evidence sufficiency response')
    if stage=='self_check':
        claims=data.get('claims')
        if not isinstance(claims,list) or not claims or any(not isinstance(c,dict) or not c.get('claim') or
               type(c.get('supported')) is not bool for c in claims): raise ValueError('Invalid self-check claims')
    return data


class Journal:
    def __init__(self,out,options):
        self.out=out; self.options=options; self.lock=threading.Lock(); self.next=0.; self.cooldown=0.; self.reservations={}
        self.log=out/'generation/api_calls.jsonl'; self.log.parent.mkdir(parents=True,exist_ok=True)
        self.spent=0.
        if self.log.exists():
            latest={}
            with self.log.open(encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        record=json.loads(line); latest[record['call_id']]=(record['status'],record.get('cost_usd') or 0.,record.get('pending_budget_reconciliation',False))
            self.spent=sum(r[1] for r in latest.values())
            if options.max_cost_usd is not None and any(r[0]=='started' or r[2] for r in latest.values()):
                raise BudgetLimit('Unreconciled in-flight requests from an interrupted run; cannot safely honor cost cap')

    def throttle(self):
        started=time.monotonic()
        with self.lock:
            now=time.monotonic(); slot=max(now,self.next,self.cooldown)
            self.next=slot+(60/self.options.rpm if self.options.rpm else 0)
        while True:
            delay=max(0,slot-time.monotonic())
            if delay: time.sleep(delay)
            with self.lock:
                if slot>=self.cooldown: break
                slot=max(time.monotonic(),self.next,self.cooldown)
                self.next=slot+(60/self.options.rpm if self.options.rpm else 0)
        return time.monotonic()-started

    def defer(self,seconds):
        with self.lock: self.cooldown=max(self.cooldown,time.monotonic()+seconds)

    def reserve(self,call_id,prompt):
        o=self.options
        with self.lock:
            if o.max_cost_usd is None: return
            # UTF-8 byte count plus framing is a conservative token upper bound.
            amount=((len(prompt.encode('utf-8'))+1024)*o.input_price+o.max_output_tokens*o.output_price)/1e6
            if self.spent+sum(self.reservations.values())+amount>o.max_cost_usd:
                raise BudgetLimit('Cost cap reached (including in-flight reservations); progress is saved')
            self.reservations[call_id]=amount

    def begin(self,entry):
        with self.lock:
            write_json(self.out/'generation/raw_calls'/(entry['call_id']+'.json'),entry)
            with self.log.open('a',encoding='utf-8') as f:
                f.write(json.dumps(entry,ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())

    def record(self,entry):
        with self.lock:
            if self.options.max_cost_usd is not None and entry.get('cost_usd') is None:
                entry['pending_budget_reconciliation']=True
                entry['reserved_unknown_cost_usd']=self.reservations.get(entry['call_id'])
            else:self.reservations.pop(entry['call_id'],None)
            self.spent+=entry.get('cost_usd') or 0.
            write_json(self.out/'generation/raw_calls'/(entry['call_id']+'.json'),entry)
            with self.log.open('a',encoding='utf-8') as f:
                f.write(json.dumps(entry,ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())


class FormalLLM:
    def __init__(self,out,journal,dataset_sha,qid,config_id,pipeline_version):
        self.out=out; self.journal=journal; self.options=journal.options
        self.dataset_sha=dataset_sha; self.qid=qid; self.config_id=config_id; self.version=pipeline_version
        self.provider=config.LLM_PROVIDER; self.model=config.LLM_MODEL
        if self.provider!='openai': raise ValueError('Formal runner currently requires the configured OpenAI-compatible endpoint')
        self.base=config.OPENAI_BASE_URL.rstrip('/'); self.key=os.environ.get('OPENAI_API_KEY','')
        if not self.key: raise PermanentAPIError('OPENAI_API_KEY is missing; do not put keys in reports')
        self.http=httpx.Client(timeout=self.options.timeout,follow_redirects=False)
        self.stats=LLMStats(); self.stats.unknown_usage_calls=0
        self.used=[]; self.stage_override=None; self.scope_prefix=''; self.validators={}; self.cache_allowed=True

    def parameters(self,temperature,json_mode):
        return {'temperature':temperature,'top_p':None,'max_tokens':self.options.max_output_tokens,
                'provider':self.provider,'endpoint':self.base,
                'response_format':'json_object' if json_mode else None,
                'uncontrolled_parameters':'Unspecified parameters use provider defaults; no seed/version pin available'}

    def generate(self,prompt,temperature=0.,json_mode=False,use_cache=True,max_retries=None):
        use_cache=use_cache and self.cache_allowed
        stage=self.stage_override or self.scope_prefix+stage_for(prompt)
        identity=cache_identity(self.dataset_sha,self.qid,self.config_id,self.model,
             self.parameters(temperature,json_mode),prompt,context_text(prompt),self.version,stage)
        key=digest(identity); path=self.out/'generation/call_cache'/key[:2]/(key+'.json')
        if use_cache and path.exists():
            cached=json.loads(path.read_text(encoding='utf-8'))
            if cached['identity']!=identity or cached['status']!='success': raise ValueError('Invalid formal cache identity')
            import hashlib
            if cached.get('text_sha256')!=hashlib.sha256(cached['text'].encode()).hexdigest():
                raise ValueError('Corrupt formal response cache')
            self.stats.cache_hits+=1; self.used.append({'cache_key':key,'call_id':cached['call_id'],'stage':stage,'cache_hit':True})
            return cached['text']
        body={'model':self.model,'temperature':temperature,'messages':[{'role':'user','content':prompt}]}
        if json_mode: body['response_format']={'type':'json_object'}
        if self.options.max_output_tokens is not None: body['max_tokens']=self.options.max_output_tokens
        last='unknown'
        for attempt in range(max_retries or self.options.retries):
            call_id=uuid.uuid4().hex
            self.journal.reserve(call_id,prompt)
            queue=self.journal.throttle(); start=time.perf_counter()
            entry={'call_id':call_id,'cache_key':key,'identity':identity,'attempt':attempt+1,
                'created_at':datetime.now(timezone.utc).isoformat(),'endpoint':self.base+'/chat/completions',
                'request':body,'queue_seconds':queue,'status':'started','usage':None,'cost_usd':None}
            self.journal.begin(entry)
            fatal=False
            try:
                response=self.http.post(self.base+'/chat/completions',json=body,headers={'Authorization':'Bearer '+self.key})
                entry['http_status']=response.status_code
                entry['rate_limit_headers']={k:v for k,v in response.headers.items() if 'ratelimit' in k.lower() or k.lower()=='retry-after'}
                if response.status_code>=400:
                    fatal=response.status_code in (400,401,402,403,404)
                    entry['error_category']='permanent_request_or_auth' if fatal else 'upstream_or_network'
                    if response.status_code==429:
                        message=response.text.lower()
                        insufficient=any(s in message for s in ['insufficient','balance','余额','额度不足','quota exhausted','per day','daily quota','rpd','tpd'])
                        fatal=insufficient
                        entry['error_category']='balance_or_quota' if insufficient else 'rate_limit'
                        retry=response.headers.get('retry-after','')
                        try: delay=float(retry)
                        except ValueError:
                            try:
                                from email.utils import parsedate_to_datetime
                                delay=max(0,parsedate_to_datetime(retry).timestamp()-time.time())
                            except (ValueError,TypeError): delay=5*2**attempt
                        if not insufficient:self.journal.defer(max(0,delay))
                    raise RuntimeError(f'HTTP {response.status_code}: {response.text[:600]}')
                raw=response.json(); entry['response']=raw
                usage=raw.get('usage') or {}; entry['usage']=usage
                pt,ot=usage.get('prompt_tokens'),usage.get('completion_tokens')
                self.stats.calls+=1
                self.stats.prompt_tokens+=pt or 0; self.stats.output_tokens+=ot or 0
                self.stats.unknown_usage_calls+=int(pt is None or ot is None)
                o=self.options
                if pt is not None and ot is not None and o.input_price is not None and o.output_price is not None and o.price_source:
                    entry['cost_usd']=(pt*o.input_price+ot*o.output_price)/1e6
                choice=raw['choices'][0]; text=choice['message']['content']
                if not isinstance(text,str) or not text.strip(): raise ValueError('Empty or non-text model response')
                if choice.get('finish_reason')=='length': raise ValueError('Truncated model output (finish_reason=length)')
                if json_mode:
                    parsed=check_json_stage(parse_json(text),stage.rsplit('/',1)[-1])
                    if stage in self.validators: self.validators[stage](parsed)
                entry.update(status='success',seconds=time.perf_counter()-start)
            except (BudgetLimit,PermanentAPIError): raise
            except Exception as error:
                last=str(error).replace(self.key,'[REDACTED]')
                entry.update(status='failed',seconds=time.perf_counter()-start,error=last)
                self.journal.record(entry)
                if fatal: raise PermanentAPIError(last) from None
                if attempt+1<(max_retries or self.options.retries): time.sleep(min(30,2**(attempt+1)))
                continue
            # Persistence failures must never cause another billable model request.
            self.stats.seconds+=entry['seconds']; self.journal.record(entry)
            write_json(path,{'identity':identity,'status':'success','call_id':call_id,'text':text,
                            'text_sha256':__import__('hashlib').sha256(text.encode()).hexdigest(),
                            'raw_response':raw,'request':body,'source':'native_formal_API'})
            self.used.append({'cache_key':key,'call_id':call_id,'stage':stage,'cache_hit':False})
            return text
        raise RuntimeError(f'Model request exhausted retries: {last}')

    def generate_json(self,prompt,temperature=0.,default=None):
        return parse_json(self.generate(prompt,temperature,json_mode=True),default)

    def close(self): self.http.close()
