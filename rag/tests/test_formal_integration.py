import json
from types import SimpleNamespace
import httpx
import pytest
from rag.llm import LLMStats
from rag.pipeline import RAGSystem
from rag.formal.client import Options,Journal,BudgetLimit,stage_for
from rag.formal.contract import matrix
from rag.formal.runner import Experiment
from rag.formal.analysis import metrics_summary,QA,RAG
from test_formal import setup_client


class RecordingModel:
    def __init__(self,iterate=False):
        self.stats=LLMStats(); self.used=[]; self.prompts=[]; self.scope_prefix=''; self.stage_override=None
        self.iterate=iterate
    def generate(self,prompt,**kwargs):
        self.prompts.append(prompt); self.stats.calls+=1
        self.used.append({'stage':self.scope_prefix+stage_for(prompt),'call_id':str(len(self.used))})
        if prompt.startswith('You are a strict fact-checker'):
            return json.dumps({'claims':[{'claim':'Unsupported fact','supported':not self.iterate,'passage':1}]})
        if prompt.startswith('You answer questions'):
            return json.dumps({'sufficient':False,'missing':'A second fact','follow_up_query':'followup'})
        if prompt.startswith('Rewrite the ANSWER'):return 'Supported fact.[1]'
        return 'Unsupported fact.[1]' if self.iterate else 'Supported fact.[1]'
    def generate_json(self,prompt,**kwargs):return json.loads(self.generate(prompt,**kwargs))
    def close(self):pass


def fixture_question():
    return {'id':'q','question':'What happened?','answer':'CANARY_GOLD_A','source_title':'CANARY_GOLD_TITLE',
        'question_type':'factual','reference_answers':['CANARY_GOLD_A','CANARY_GOLD_B'],
        'annotation_variants':{'human_reannotated':{'answer':'CANARY_GOLD_B'}}}


def fixture_experiment(model):
    experiment=object.__new__(Experiment)
    experiment.configs=matrix(); experiment.options=Options(); experiment.dataset_sha='dataset';experiment.version='version'
    experiment.system=object.__new__(RAGSystem)
    experiment.rankings={'G':{'q':['c1']}}
    experiment.chunks={'c1':{'chunk_id':'c1','doc_id':'d','title':'Retrieved source','section':'',
                           'url':'https://example.test','text':'Supported fact.'}}
    experiment.client=lambda qid,code:model
    return experiment


def test_actual_generation_path_never_sends_gold_metadata():
    model=RecordingModel(); experiment=fixture_experiment(model)
    row=experiment.generate_one('G',fixture_question())
    assert len(model.prompts)==2
    assert all('CANARY_GOLD' not in prompt for prompt in model.prompts)
    assert row['answer']=='Supported fact.[1]'
    assert row['online_end_to_end_seconds'] is None  # verified retrieval cache is not fresh inference


def test_iteration_counterfactual_keeps_self_check_enabled(monkeypatch):
    model=RecordingModel(iterate=True); experiment=fixture_experiment(model)
    initial=experiment.chunks['c1']
    def retrieve(system,query,cfg):
        hits=[dict(initial)] if query=='What happened?' else [dict(initial,chunk_id='c2'),dict(initial,chunk_id='c3')]
        for i,h in enumerate(hits,1):h['rank']=i
        return hits,{'timings':{'retrieval_total':.001},'config':{}}
    monkeypatch.setattr(RAGSystem,'retrieve',retrieve)
    row=experiment.generate_one('K',fixture_question())
    assert row['draft_answer_before_iteration']=='Unsupported fact.[1]'
    assert row['answer_before_iteration']==row['answer']=='Supported fact.[1]'
    assert row['self_check_before_iteration']['unsupported']
    assert len(row['additional_retrieved_chunks'])==2
    assert any(r['stage']=='before_iteration/self_check' for r in model.used)
    assert all('CANARY_GOLD' not in prompt for prompt in model.prompts)


def test_failure_and_unknown_metrics_are_not_silently_scored_zero():
    kinds=['definition','definition','why','cause_effect','factual','comparison']
    records={}
    for i,kind in enumerate(kinds):
        score=0. if i<2 else 1.
        records[str(i)]={'id':str(i),'question_type':kind,'execution_status':'success',
                        **{k:score for k in QA},'multi_reference':{k:score for k in QA}}
    records['failed']={'id':'failed','question_type':'factual','execution_status':'failed','F1':1.}
    summary,_=metrics_summary({'A':records},{'A':{}},set(records))
    s=summary['A']
    assert s['n_failed']==1 and s['n_success']==6 and s['coverage']==6/7
    assert s['primary_and_RAG_metrics']['F1']['mean']==pytest.approx(4/6)
    assert s['primary_and_RAG_metrics']['F1']['type_macro_mean']==pytest.approx(.8)
    assert s['primary_and_RAG_metrics']['faithfulness']=={'mean':None,'n_scored':0,'type_macro_mean':None}


def test_unknown_billed_usage_keeps_budget_reserved_across_restart(tmp_path,monkeypatch):
    options=Options(rpm=0,max_cost_usd=1.,max_output_tokens=100,input_price=1.,output_price=1.,price_source='test verified tariff')
    def unknown_usage(request):
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':'Actual model text'}}]})
    llm,journal=setup_client(tmp_path,monkeypatch,unknown_usage,options)
    llm.generate('Question');llm.close()
    assert journal.reservations
    with pytest.raises(BudgetLimit):Journal(tmp_path,options)


def test_empty_progress_reports_do_not_produce_fake_charts(tmp_path):
    from rag.formal.reporting import reports,figures
    summaries={c:{'n_records':0,'n_success':0,'n_failed':0,'n_not_run':1359,
                  'primary_and_RAG_metrics':{k:{'mean':None} for k in QA+RAG}} for c in matrix()}
    (tmp_path/'config').mkdir()
    (tmp_path/'config/model_config.json').write_text(json.dumps({'requested_model':'unit-test-model'}),encoding='utf-8')
    figures(tmp_path,summaries,[])
    reports(tmp_path,summaries,[],{},{},{'input_tokens_reported':0,'output_tokens_reported':0,'status':'unknown'},
            {'status':'pending','independent_human_review_n':0})
    assert not list((tmp_path/'figures').glob('*.png'))
    assert '尚未完成' in (tmp_path/'reports/Q4_FORMAL_REPORT.md').read_text(encoding='utf-8')
