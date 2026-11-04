"""Human adjudication stays explicit; model suggestions never fill human fields."""
from collections import Counter,defaultdict
import csv
import json
import math
import random
from rag.corpus import load_jsonl
from .contract import DATA,Q4,sha,write_json
from .preflight import csv_file


LABELS={'agree','partial','disagree','uncertain'}


def wilson(success,n):
    if not n: return None
    z=1.959963984540054; p=success/n; d=1+z*z/n
    center=(p+z*z/(2*n))/d; half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0.,center-half),min(1.,center+half)]


def annotation_report(out):
    folder=out/'data_validation'; path=folder/'annotation_adjudication_template.csv'
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f); fields=reader.fieldnames; reviews=list(reader)
    expected={q['id'] for q in load_jsonl(DATA)}
    if len(reviews)!=len(expected) or {r['question_id'] for r in reviews}!=expected:
        raise ValueError('Human adjudication template IDs are duplicated, missing or extra')
    suggestions={}
    for file in (Q4/'agreement').glob('*.json'):
        row=json.loads(file.read_text(encoding='utf-8'))
        suggestions[row['id']]=row
    for row in reviews:
        suggestion=suggestions.get(row['question_id'])
        if suggestion:
            row['suggested_label']=suggestion['AB']; row['suggested_reason']=suggestion['reason']
    # Preserve all human-authored columns, including corrections, on refresh.
    csv_file(path,fields,reviews)
    csv_file(folder/'annotation_review_queue.csv',fields,
             sorted(reviews,key=lambda r:(r['human_label'] in LABELS,r['suggested_label']=='agree',r['question_id'])))
    accepted=[r for r in reviews if r['human_label'] in LABELS and all(r.get(k,'').strip()
                   for k in ['human_reviewer','human_reason','review_date'])]
    existing=json.loads((folder/'annotation_agreement_report.json').read_text(encoding='utf-8'))
    counts=Counter(r['human_label'] for r in accepted)
    existing.update(independent_human_review_n=len(accepted),human_label_counts=dict(counts),
        independent_human_answer_agreement=counts['agree']/1359 if len(accepted)==1359 else None,
        independent_human_ci95=wilson(counts['agree'],1359) if len(accepted)==1359 else None,
        reviewed_only_agreement=counts['agree']/len(accepted) if accepted else None,
        reviewed_only_ci95=wilson(counts['agree'],len(accepted)),
        status='complete' if len(accepted)==1359 else 'pending_independent_human_review',
        source_human_review_sha256=sha(path),
        LLM_assistance={'n':len(suggestions),'counts':dict(Counter(r['AB'] for r in suggestions.values())),
                       'source':'Existing Q4 independent stateless model comparisons; not human review'},
        confidence_interval_method='Wilson binomial interval; does not model article-level correlation',
        review_authenticity='Human reviewer identity and independent work are declared in user-supplied review file, not verified by software')
    write_json(folder/'annotation_agreement_report.json',existing)
    return existing


def judge_validation_sample(out,evaluations):
    folder=out/'data_validation'
    template=folder/'judge_human_validation_100.csv'
    if template.exists():
        with template.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
        reviewed=[r for r in rows if r.get('human_reviewer','').strip() and r.get('human_correctness','') in ['0','0.5','1']]
        write_json(folder/'judge_validation_status.json',{'n_sample':len(rows),'n_human_reviewed':len(reviewed),
            'status':'complete' if len(reviewed)==len(rows) and len(rows)>=100 else 'pending',
            'different_from_annotation_IAA':True,'source_sha256':sha(template)})
        return
    candidates=defaultdict(list)
    for code,rows in evaluations.items():
        for qid,row in rows.items():
            if row.get('execution_status')=='success':
                candidates[(row['question_type'],str(row.get('answer_correctness')))].append((code,qid))
    rng=random.Random(42)
    for entries in candidates.values(): rng.shuffle(entries)
    selected=[]
    while len(selected)<100 and any(candidates.values()):
        for bucket in sorted(candidates):
            if candidates[bucket] and len(selected)<100: selected.append(candidates[bucket].pop())
    questions={q['id']:q for q in load_jsonl(DATA)}
    rows=[]; key={}
    if len(selected)<100:
        write_json(folder/'judge_validation_status.json',{'n_sample':0,'n_available':len(selected),
            'n_human_reviewed':0,'status':'pending_at_least_100_real_judgments'})
        return
    for index,(code,qid) in enumerate(selected):
        pred=json.loads((out/'generation/records'/code/(qid+'.json')).read_text(encoding='utf-8'))
        task_id=f'blind_review_{index:03d}'; key[task_id]={'configuration_id':code,'question_id':qid}
        q=questions[qid]
        rows.append({'review_id':task_id,'question_type':q['question_type'],'question':q['question'],
            'answer':pred['answer'],'reference_A':q['answer'],
            'reference_B':q['annotation_variants']['human_reannotated']['answer'],
            'passages':json.dumps(pred.get('hits',[]),ensure_ascii=False),'human_correctness':'',
            'human_faithfulness':'','human_relevance':'','human_reviewer':'','human_reason':''})
    if rows:
        csv_file(template,list(rows[0]),rows,overwrite=False)
        write_json(folder/'judge_human_validation_blind_key.json',key)
    write_json(folder/'judge_validation_status.json',{'n_sample':len(rows),'n_human_reviewed':0,
        'status':'pending','selection':'seed42 round-robin question-type x model-correctness strata',
        'configuration_and_model_scores_hidden_from_review_template':True})
