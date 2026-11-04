from copy import deepcopy
import json
import pytest
from rag.six.scope import CODES,matrix,original_matrix,check_matrix,CONTRASTS


def test_six_scope_preserves_q4_and_adds_unconfounded_iteration_groups():
    configs=matrix();check_matrix(configs)
    assert list(configs)==CODES and len(configs)==6
    for c in ['0','A','B','G']:assert configs[c]==original_matrix()[c]
    assert configs['S']['generation']['iterative'] and not configs['S']['generation']['self_check']
    assert configs['T']['generation']['iterative'] and configs['T']['generation']['self_check']
    assert not configs['S']['retrieval']['rewrite'] and not configs['T']['retrieval']['rewrite']
    assert original_matrix()['F']['retrieval']['rewrite']
    wrong=deepcopy(configs);wrong['T']['retrieval']['hyde']=True
    with pytest.raises(ValueError):check_matrix(wrong)


def test_six_statistics_use_seven_contrasts_and_factorial_interaction(tmp_path):
    from rag.six.statistics import paired_statistics
    questions=[{'id':str(i),'gold_doc_id':str(i)} for i in range(3)]
    values={'0':0.,'A':.1,'B':.2,'G':.3,'S':.4,'T':.6}
    predictions={c:{q['id']:{'execution_status':'success','F1':value} for q in questions} for c,value in values.items()}
    results=paired_statistics(tmp_path,predictions,{c:{} for c in CODES},questions)
    assert {(r['control'],r['treatment']) for r in results}==set(CONTRASTS)
    data=json.loads((tmp_path/'ablation/bootstrap_confidence_intervals.json').read_text(encoding='utf-8'))
    assert data['interaction'][0]['delta']==pytest.approx(.1)
    assert all(r['n']==3 and 'p_holm' in r for r in results)


@pytest.mark.parametrize('code,expected_check',[('S',False),('T',True)])
def test_actual_generation_uses_no_rewrite_and_retains_requested_self_check(code,expected_check,monkeypatch):
    from test_formal_integration import RecordingModel,fixture_experiment,fixture_question
    from rag.pipeline import RAGSystem
    model=RecordingModel(iterate=True);experiment=fixture_experiment(model);experiment.configs=matrix()
    initial=experiment.chunks['c1'];seen=[]
    def retrieve(system,query,cfg):
        seen.append(cfg)
        hits=[dict(initial)] if query=='What happened?' else [dict(initial,chunk_id='c2')]
        for i,h in enumerate(hits,1):h['rank']=i
        return hits,{'timings':{'retrieval_total':.001},'config':{}}
    monkeypatch.setattr(RAGSystem,'retrieve',retrieve)
    row=experiment.generate_one(code,fixture_question())
    assert bool(row.get('self_check'))==expected_check
    assert len(row['retrieval_trace'])==2 and all(not cfg.rewrite for cfg in seen)
    assert all('CANARY_GOLD' not in p for p in model.prompts)


def test_six_report_describes_scope_and_does_not_invent_missing_results(tmp_path):
    from rag.six.reporting import reports,figures
    from rag.formal.analysis import QA,RAG
    config=tmp_path/'config';config.mkdir()
    (config/'model_config.json').write_text(json.dumps({'requested_model':'unit-test-model',
        'options':{'rpm':60,'workers':6,'timeout':240}}),encoding='utf-8')
    (config/'runtime_amendment.json').write_text('{}',encoding='utf-8')
    summaries={c:{'n_success':0,'n_judgments':0,'n_records':0,
        'primary_and_RAG_metrics':{k:{'mean':None,'n_scored':0} for k in QA+RAG}} for c in CODES}
    reports(tmp_path,summaries,[],{}, {},{'input_tokens_reported':0,'output_tokens_reported':0,'status':'unknown'},
            {'status':'pending','independent_human_review_n':0})
    text=(tmp_path/'reports/Q4_FORMAL_REPORT.md').read_text(encoding='utf-8')
    assert '8,154' in text and 'N/A (n=0)' in text and 'pending' in text
    figures(tmp_path,summaries,[])
    assert not list((tmp_path/'figures').glob('*.png'))


def test_unstarted_six_scope_cannot_claim_completion(tmp_path):
    from rag.six.verification import verify
    result=verify(tmp_path)
    assert not result['automated_complete'] and not result['all_requirements_complete']
