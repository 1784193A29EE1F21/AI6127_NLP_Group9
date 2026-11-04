from rag.formal.difficulty import challenge_slices,analyze_challenges
from rag.formal.analysis import QA


def test_challenge_slices_are_overlapping_fixed_rules_and_do_not_claim_semantic_disagreement():
    questions=[{'id':'a','question':' '.join(['word']*30),'question_type':'comparison',
                'gold_chunk_ids':['c1','c2','c2'],'reference_answers':['one fact','different claim']},
               {'id':'b','question':'Short question','question_type':'factual',
                'gold_chunk_ids':['c1','c1'],'reference_answers':['same answer','same answer']}]
    groups=challenge_slices(questions)
    assert all(ids=={'a'} for ids in groups.values())
    assert len(questions)==2


def test_empty_challenge_group_is_unmeasured_and_failures_remain_in_denominator(tmp_path):
    questions=[{'id':'a','question':'Short','question_type':'comparison','reference_answers':['x','x']},
               {'id':'b','question':'Short','question_type':'comparison','reference_answers':['x','x']}]
    predictions={'A':{'a':{'id':'a','question_type':'comparison','execution_status':'success',
                           **{key:1. for key in QA},'multi_reference':{key:1. for key in QA}},
                      'b':{'id':'b','question_type':'comparison','execution_status':'failed'}}}
    result=analyze_challenges(tmp_path,questions,predictions,{'A':{}})
    assert result['comparison']['A']['coverage']==.5
    assert result['comparison']['A']['primary_and_RAG_metrics']['F1']['mean']==1.
    empty=result['multiple_gold_chunks']['A']
    assert empty['n_target']==0 and empty['coverage'] is None
    assert empty['primary_and_RAG_metrics']['F1']['mean'] is None
