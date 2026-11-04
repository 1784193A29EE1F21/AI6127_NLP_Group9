import json
from dataclasses import replace
import httpx
import pytest
from rag.formal.contract import cache_identity,digest,matrix,check_matrix
from rag.formal.client import Options,Journal,FormalLLM,BudgetLimit
from rag.formal.evaluation import token_pr,citation_validity,validate_judge,mean_boolean


def test_full_matrix_has_exactly_19_and_one_module_controls():
    configs=matrix(); assert len(configs)==19 and check_matrix(configs)
    configs['E']['retrieval']['prf']=True
    with pytest.raises(ValueError): check_matrix(configs)


def test_cache_contract_changes_for_every_required_scope():
    args={'dataset_sha':'dataset','qid':'q','config_id':'A','model':'m',
          'parameters':{'temperature':0},'prompt':'p','context':['passage'],'pipeline_version':'v','stage':'answer'}
    base=digest(cache_identity(**args))
    for field,value in [('dataset_sha','other'),('qid','q2'),('config_id','B'),('model','m2'),
         ('parameters',{'temperature':.5}),('prompt','changed'),('context',['other']),('pipeline_version','v2'),('stage','revision')]:
        assert digest(cache_identity(**{**args,field:value}))!=base


def test_precision_recall_and_citations_use_actual_answer_and_context():
    assert token_pr('red blue','red green yellow')==(.5,1/3)
    assert citation_validity('Fact.[1] Other.[0][9]',[{},{}])['citation_validity']==1/3
    assert citation_validity('No citations',[{}])['citation_validity'] is None
    assert mean_boolean([True,None]) is None


def setup_client(tmp_path,monkeypatch,handler,options=None):
    import rag.formal.client as module
    monkeypatch.setenv('OPENAI_API_KEY','unit-test-placeholder')
    monkeypatch.setattr(module.config,'LLM_PROVIDER','openai')
    monkeypatch.setattr(module.config,'LLM_MODEL','mock-model')
    monkeypatch.setattr(module.config,'OPENAI_BASE_URL','https://unit.invalid/v1')
    http=httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(module.httpx,'Client',lambda **kwargs:http)
    journal=Journal(tmp_path,options or Options(rpm=0))
    return FormalLLM(tmp_path,journal,'dataset','q','A','version'),journal


def good_response():
    return httpx.Response(200,json={'model':'mock-model','choices':[{'finish_reason':'stop','message':{'content':'A fact.[1]'}}],
        'usage':{'prompt_tokens':12,'completion_tokens':4}})


def test_real_client_cache_is_scoped_and_keeps_raw_request_without_secret(tmp_path,monkeypatch):
    calls=[]
    llm,journal=setup_client(tmp_path,monkeypatch,lambda request:(calls.append(request),good_response())[1])
    assert llm.generate('Question')=='A fact.[1]'
    assert llm.generate('Question')=='A fact.[1]' and len(calls)==1
    assert llm.stats.cache_hits==1
    llm.config_id='B'; assert llm.generate('Question')=='A fact.[1]' and len(calls)==2
    entries=[json.loads(line) for line in journal.log.read_text(encoding='utf-8').splitlines()]
    assert sum(e['status']=='success' for e in entries)==2
    assert all('Authorization' not in e['request'] for e in entries)
    assert 'unit-test-placeholder' not in journal.log.read_text(encoding='utf-8')
    assert entries[-1]['response']['usage']['prompt_tokens']==12
    llm.close()


def test_cost_cap_is_enforced_before_contacting_endpoint(tmp_path,monkeypatch):
    calls=[]
    options=Options(rpm=0,max_cost_usd=.000001,max_output_tokens=100,input_price=1.,output_price=1.,price_source='verified test tariff')
    llm,journal=setup_client(tmp_path,monkeypatch,lambda request:(calls.append(request),good_response())[1],options)
    with pytest.raises(BudgetLimit): llm.generate('Question')
    assert not calls
    llm.close()


def test_invalid_structured_reply_never_becomes_a_successful_cache(tmp_path,monkeypatch):
    import rag.formal.client as module
    monkeypatch.setattr(module.time,'sleep',lambda seconds:None)
    def bad(request):
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':'not JSON'}}],
                       'usage':{'prompt_tokens':2,'completion_tokens':2}})
    llm,journal=setup_client(tmp_path,monkeypatch,bad,Options(rpm=0,retries=2))
    with pytest.raises(RuntimeError): llm.generate_json('You are a strict fact-checker')
    assert not list((tmp_path/'generation/call_cache').glob('*/*.json'))
    final=[json.loads(p.read_text(encoding='utf-8')) for p in (tmp_path/'generation/raw_calls').glob('*.json')]
    assert len(final)==2 and all(r['status']=='failed' for r in final)
    llm.close()


def test_empty_judge_is_not_a_high_faithfulness_default():
    with pytest.raises(ValueError): validate_judge({},5)


def test_budget_with_unknown_prices_fails_closed(monkeypatch):
    monkeypatch.setenv('MAX_COST_USD','1')
    monkeypatch.delenv('INPUT_PRICE_PER_MTOK',raising=False)
    monkeypatch.delenv('OUTPUT_PRICE_PER_MTOK',raising=False)
    with pytest.raises(BudgetLimit): Options.from_env()
