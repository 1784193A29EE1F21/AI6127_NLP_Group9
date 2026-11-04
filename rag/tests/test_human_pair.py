import pytest
from rag.human_pair import human_only, reference_scores, validate_pair, validate_reference_claims


def row():
    return {'id':'q1','question':'Which city?','answer':'AI contaminated main answer',
            'question_type_all_agree':False,'ai_needs_review':True,
            'annotation_variants':{
                'human_original':{'origin':'human','answer':'London','evidence':'In London.','question_type':'factual'},
                'human_reannotated':{'origin':'human','answer':'Paris','evidence':'In Paris.','question_type':'factual'},
                'ai_chat':{'origin':'AI','answer':'Berlin','question_type':'why'}}}


def test_human_only_reconstructs_reference_without_ai_or_ai_filtering():
    source=row()
    result=human_only([source])[0]
    assert result['answer']=='London'
    assert set(result['annotation_variants'])=={'human_original','human_reannotated'}
    assert result['reference_answers']==['London','Paris']
    assert 'ai_needs_review' not in result and 'question_type_all_agree' not in result
    result['annotation_variants']['human_original']['answer']='Changed'
    assert source['annotation_variants']['human_original']['answer']=='London'


def test_duplicate_questions_or_missing_human_group_are_rejected():
    with pytest.raises(ValueError,match='Duplicate'): human_only([row(),row()])
    item=row(); del item['annotation_variants']['human_reannotated']
    with pytest.raises(ValueError,match='Two human'): human_only([item])


def test_primary_and_two_reference_scores_are_distinct():
    q=human_only([row()])[0]
    scores=reference_scores('Paris',q)
    assert scores['EM']==0 and scores['F1']==0
    assert scores['multi_reference']['EM']==1 and scores['multi_reference']['F1']==1


def test_pair_labels_cannot_be_replaced_with_third_group_decisions():
    with pytest.raises(ValueError): validate_pair({'AC':'agree','BC':'agree','reason':'Wrong pairs'})
    assert validate_pair({'AB':'partial','reason':'An essential part is omitted'})['AB']=='partial'


@pytest.mark.parametrize('claims',[None,[],[{'claim':'x','supported':'false'}]])
def test_semantic_recall_requires_explicit_valid_reference_claims(claims):
    with pytest.raises(ValueError): validate_reference_claims({'reference_claims':claims})
