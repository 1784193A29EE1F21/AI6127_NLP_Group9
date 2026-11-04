import csv
import json
from pathlib import Path
from types import SimpleNamespace
import httpx
import pytest
from rag.formal.contract import check_matrix,matrix
from rag.formal.evaluation import validate_judge
from rag.formal.statistics import compare,holm
from rag.formal.verification import expected_generation_prompt
from test_formal import setup_client,good_response


def test_title_filter_or_reversed_module_matrix_is_rejected():
    configs=matrix()
    for code in configs:
        if code!='0':configs[code]['retrieval']['title_filter']='gold source title'
    with pytest.raises(ValueError):check_matrix(configs)
    configs=matrix(); configs['A']['retrieval']['rerank']=True
    with pytest.raises(ValueError):check_matrix(configs)


def test_bootstrap_is_clustered_reproducible_and_zero_baseline_is_explicit():
    args=([0.,0.,0.,0.],[.1,.1,.1,.1],['article1','article1','article2','article2'])
    result=compare(*args,resamples=200)
    assert result==compare(*args,resamples=200)
    assert result['n_source_articles']==2 and result['relative_change'] is None
    assert result['ci95']==pytest.approx([.1,.1])
    null=compare([0,1],[0,1],['a','b'],resamples=200)
    assert null['p_value']==1 and null['ties']==2
    assert holm([.01,.04,.03])==pytest.approx([.03,.06,.06])


def test_reference_canaries_are_absent_from_every_generation_prompt():
    q={'question':'What happened?','answer':'CANARY_GOLD_A','source_title':'CANARY_GOLD_TITLE'}
    hits=[{'title':'Retrieved source','section':'','text':'Actual retrieved evidence.'}]
    row={'hits':hits,'draft_answer':'Draft.[1]','self_check':{'unsupported':['Unverified detail']},
         'retrieval_trace':[{'hits':hits}],'draft_answer_before_iteration':'Earlier draft.[1]',
         'self_check_before_iteration':{'unsupported':['Earlier detail']}}
    for stage in ['answer','closed_book','rewrite','hyde','evidence_assessment','self_check','revision',
                  'before_iteration/answer','before_iteration/self_check','before_iteration/revision']:
        prompt=expected_generation_prompt(stage,q,row)
        assert 'CANARY_GOLD' not in prompt


def test_corrupt_cache_is_rejected_without_an_extra_billable_request(tmp_path,monkeypatch):
    calls=[]
    llm,_=setup_client(tmp_path,monkeypatch,lambda req:(calls.append(req),good_response())[1])
    llm.generate('Question')
    path=next((tmp_path/'generation/call_cache').glob('*/*.json'))
    cache=json.loads(path.read_text(encoding='utf-8')); cache['text']='corrupt'
    path.write_text(json.dumps(cache),encoding='utf-8')
    with pytest.raises(ValueError,match='Corrupt'):llm.generate('Question')
    assert len(calls)==1
    llm.close()


def test_retry_after_is_honored_instead_of_rotating_keys(tmp_path,monkeypatch):
    import rag.formal.client as module
    clock=[0.]; sent=[]
    monkeypatch.setattr(module.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(module.time,'sleep',lambda seconds:clock.__setitem__(0,clock[0]+seconds))
    def handler(request):
        sent.append(clock[0])
        if len(sent)==1:return httpx.Response(429,headers={'Retry-After':'7'},json={'error':'rate limit'})
        return good_response()
    llm,journal=setup_client(tmp_path,monkeypatch,handler)
    llm.generate('Question')
    assert len(sent)==2 and sent[1]-sent[0]>=7
    records=[json.loads(p.read_text(encoding='utf-8')) for p in (tmp_path/'generation/raw_calls').glob('*.json')]
    assert any(r.get('error_category')=='rate_limit' for r in records)
    llm.close()


def test_model_suggestions_never_become_human_review(tmp_path,monkeypatch):
    import rag.formal.reviews as module
    folder=tmp_path/'data_validation'; folder.mkdir()
    columns=['question_id','question_type','question','answer_A','answer_B','source_url',
             'suggested_label','suggested_reason','human_label','human_reviewer','human_reason','review_date']
    path=folder/'annotation_adjudication_template.csv'
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        writer.writerows([{**{k:'' for k in columns},'question_id':qid} for qid in ['q1','q2']])
    (folder/'annotation_agreement_report.json').write_text('{}',encoding='utf-8')
    source=tmp_path/'legacy';(source/'agreement').mkdir(parents=True)
    for qid in ['q1','q2']:
        (source/'agreement'/(qid+'.json')).write_text(json.dumps({'id':qid,'AB':'agree','reason':'model suggestion'}),encoding='utf-8')
    monkeypatch.setattr(module,'Q4',source)
    monkeypatch.setattr(module,'load_jsonl',lambda path:[{'id':'q1'},{'id':'q2'}])
    result=module.annotation_report(tmp_path)
    assert result['independent_human_review_n']==0
    assert result['independent_human_answer_agreement'] is None
    assert result['LLM_assistance']['n']==2 and result['status']=='pending_independent_human_review'


def test_unstarted_formal_execution_cannot_pass_audit(tmp_path):
    from rag.formal.verification import verify
    result=verify(tmp_path)
    assert not result['automated_complete'] and not result['all_requirements_complete']


def test_published_tables_reject_changed_scores_duplicate_rows_and_null_as_zero(tmp_path):
    from rag.formal.verification import table_errors
    path=tmp_path/'metrics.csv'
    expected=[{'configuration':'A','F1':.5,'faithfulness':None}]
    path.write_text('configuration,F1,faithfulness\nA,0.5,\n',encoding='utf-8')
    assert table_errors(path,expected,['configuration'])==[]
    path.write_text('configuration,F1,faithfulness\nA,0.9,0\nA,0.9,0\n',encoding='utf-8')
    errors=table_errors(path,expected,['configuration'])
    assert len(errors)==3


def test_evidence_resolution_reconstructs_actual_prompt_without_gold():
    from rag.formal.verification import expected_resolution_prompt
    from rag.formal.runner import Experiment
    initial={'chunk_id':'c1','title':'Source','text':'First fact.','section':''}
    added={**initial,'chunk_id':'c2','text':'Second fact.'}
    q={'id':'q','question':'Compare the facts?','answer':'CANARY_GOLD'}
    row={'retrieval_trace':[{'hits':[initial]}],'hits':[initial,added],
         'iterative_rounds':[{'missing':'Second fact'}]}
    captured=[]
    class Client:
        validators={}; used=[]
        stats=SimpleNamespace(as_dict=lambda:{})
        def generate_json(self,prompt):
            captured.append(prompt)
            return {'resolved':True,'reason':'Added passage contains second fact.',
                    'supporting_added_passage_numbers':[1]}
        def close(self):pass
    experiment=object.__new__(Experiment)
    experiment.dataset_sha='data'; experiment.version='version'
    experiment.client=lambda *args:Client()
    experiment.resolve_iteration('F',q,row)
    assert captured==[expected_resolution_prompt(q,row)]
    assert 'CANARY_GOLD' not in captured[0]


def test_checkpoint_rename_retries_transient_windows_lock_without_truncation(tmp_path,monkeypatch):
    import rag.formal.contract as contract
    path=tmp_path/'status.json'; path.write_text('{"old":true}',encoding='utf-8')
    original=Path.replace; calls=[]
    def flaky(source,target):
        calls.append(source)
        if len(calls)<3:
            assert json.loads(path.read_text())=={'old':True}
            raise PermissionError('Simulated transient Windows reader lock')
        return original(source,target)
    monkeypatch.setattr(Path,'replace',flaky)
    monkeypatch.setattr(contract.time,'sleep',lambda _:None)
    contract.write_json(path,{'new':True})
    assert len(calls)==3 and json.loads(path.read_text())=={'new':True}
