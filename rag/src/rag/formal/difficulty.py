"""Fixed, descriptive challenge slices; never exclude questions from the main set."""
from .contract import write_json
from .evaluation import qa_scores
from .preflight import csv_file


def challenge_slices(questions):
    groups={name:set() for name in ['comparison','multiple_gold_chunks','long_question_30_words',
                                   'human_reference_lexical_F1_below_0_5']}
    for q in questions:
        qid=q['id']
        if q.get('question_type')=='comparison': groups['comparison'].add(qid)
        if len(set(q.get('gold_chunk_ids',[])))>=2: groups['multiple_gold_chunks'].add(qid)
        if len(q['question'].split())>=30: groups['long_question_30_words'].add(qid)
        refs=q.get('reference_answers',[])
        if len(refs)>=2 and qa_scores(refs[0],{'answer':refs[1],'reference_answers':[refs[1]]})['F1']<.5:
            groups['human_reference_lexical_F1_below_0_5'].add(qid)
    return groups


def analyze_challenges(out,questions,predictions,evaluations):
    # Delayed import avoids a cycle with analysis.analyze().
    from .analysis import metrics_summary
    groups=challenge_slices(questions); summaries={}; rows=[]
    for name,ids in groups.items():
        summary,_=metrics_summary(predictions,evaluations,ids); summaries[name]=summary
        for code,s in summary.items():
            rows.append({'slice':name,'configuration':code,'n_target':len(ids),
                'n_success':s['n_success'],'coverage':s['coverage'],
                **{k:v['mean'] for k,v in s['primary_and_RAG_metrics'].items()},
                **{k+'_n_scored':v['n_scored'] for k,v in s['primary_and_RAG_metrics'].items()}})
    write_json(out/'ablation/difficult_question_analysis.json',{
        'selection':'Fixed metadata/reference rules, independent of generated answers and configuration scores.',
        'interpretation':'Descriptive challenge proxies, not human difficulty labels; overlapping slices; exploratory comparisons only.',
        'rules':{'comparison':'Canonical human question_type equals comparison.',
                 'multiple_gold_chunks':'At least two distinct annotated gold_chunk_ids; does not establish multi-hop reasoning.',
                 'long_question_30_words':'At least 30 whitespace-delimited words in the original question.',
                 'human_reference_lexical_F1_below_0_5':'Token F1 between the two human answers below 0.5; not semantic disagreement.'},
        'ids':{name:sorted(ids) for name,ids in groups.items()},'results':summaries})
    csv_file(out/'ablation/difficult_question_metrics.csv',list(rows[0]),rows)
    return summaries
