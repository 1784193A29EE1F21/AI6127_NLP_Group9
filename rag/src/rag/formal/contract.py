from __future__ import annotations
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import time
from pathlib import Path
from rag.ablations import generation_configs
from rag.config import PATHS
from rag.corpus import load_jsonl
from rag.metrics import normalize_answer

ROOT=PATHS.results_dir/'q4_q5_formal_full'
DATA=PATHS.eval_dir/'human_pair/eval_qa.jsonl'
Q4=PATHS.results_dir/'q4_two_humans_2026-11-02'
CONTRASTS=[('0','A'),('A','B'),('A','C'),('A','D'),('B','D'),('C','D'),
           ('D','E'),('D','F'),('B','G'),('B','H'),('B','I'),('B','J')]+[(c,'K') for c in 'LMNOPQR']


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def write_json(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    for attempt in range(10):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt==9: raise
            time.sleep(min(.05*(attempt+1),.5))


def matrix():
    return {name.split()[0]:{'name':name,'retrieval':asdict(r) if r else None,
            'generation':asdict(g) if g else None} for name,r,g in generation_configs(True)}


def check_matrix(configs):
    if set(configs)!=set(['0']+list('ABCDEFGHIJKLMNOPQR')):
        raise ValueError('Exactly 19 expected configurations required')
    flags={'A':set(),'B':{'rerank'},'C':{'rewrite'},'D':{'rerank','rewrite'},
        'E':{'rerank','rewrite','self_check'},'F':{'rerank','rewrite','iterative'},
        'G':{'rerank','self_check'},'H':{'rerank','prf'},'I':{'rerank','hyde'},'J':{'rerank','diversity'}}
    all_flags={'rerank','rewrite','prf','hyde','diversity','self_check','iterative'}
    flags['K']=all_flags
    for code,removed in zip('LMNOPQR',['rerank','prf','rewrite','hyde','diversity','self_check','iterative']):
        flags[code]=all_flags-{removed}
    if configs['0']['retrieval'] is not None or configs['0']['generation'] is not None:
        raise ValueError('Closed-book must not use retrieval or RAG enhancements')
    for code,wanted in flags.items():
        r,g=configs[code]['retrieval'],configs[code]['generation']
        actual={key for key in ['rerank','rewrite','prf','hyde'] if r[key]}
        actual|={key for key in ['self_check','iterative'] if g[key]}
        if r['max_per_doc']:actual.add('diversity')
        if actual!=wanted:raise ValueError(f'{code}: wrong actual enabled modules')
        if r['title_filter'] or r['mode']!='hybrid' or r['top_k']!=5 or r['k_sparse']!=100 or r['k_dense']!=100 or r['rrf_k']!=60 or r['rerank_candidates']!=20:
            raise ValueError(f'{code}: unexpected retrieval parameters or title filtering')
        if r['max_per_doc'] not in (0,2) or g['style']!='concise' or g['max_rounds']!=2:
            raise ValueError(f'{code}: unexpected generation/diversity parameters')
    expected=[('A','B','rerank'),('A','C','rewrite'),('B','D','rewrite'),('C','D','rerank'),
              ('D','E','self_check'),('D','F','iterative'),('B','G','self_check'),
              ('B','H','prf'),('B','I','hyde'),('B','J','max_per_doc')]
    expected += [('K',c,f) for c,f in zip('LMNOPQR',
                  ['rerank','prf','rewrite','hyde','max_per_doc','self_check','iterative'])]
    def flatten(code):
        c=configs[code]; return {**c['retrieval'],**c['generation']}
    for a,b,key in expected:
        left,right=flatten(a),flatten(b)
        changed={k for k in left if left[k]!=right[k]}
        if changed!={key}: raise ValueError(f'{a}/{b} must differ only in {key}: {changed}')
    if configs['G']['generation']['iterative'] or not configs['G']['generation']['self_check']:
        raise ValueError('Q4 final system must remain G')
    return True


def validate_dataset():
    questions=load_jsonl(DATA)
    original=load_jsonl(PATHS.eval_file)
    subset=load_jsonl(PATHS.eval_dir/'consensus_type/eval_qa.jsonl')
    chunks={r['chunk_id']:r for r in load_jsonl(PATHS.chunks_file)}
    original_by_id={q['id']:q for q in original}
    errors=[]; seen=set(); text_ids={}; duplicate_questions=[]
    for q in questions:
        qid=q.get('id')
        if not qid or qid in seen: errors.append({'id':qid,'error':'missing_or_duplicate_id'})
        seen.add(qid)
        if not isinstance(q.get('question'),str) or not q['question'].strip():
            errors.append({'id':qid,'error':'missing_question'})
        key=normalize_answer(q.get('question',''))
        if key in text_ids: duplicate_questions.append([text_ids[key],qid])
        text_ids[key]=qid
        variants=q.get('annotation_variants',{})
        if set(variants)!={'human_original','human_reannotated'}:
            errors.append({'id':qid,'error':'not_exactly_two_human_groups'})
        for group in ['human_original','human_reannotated']:
            if not variants.get(group,{}).get('answer','').strip(): errors.append({'id':qid,'error':'empty_'+group})
            if variants.get(group)!=original_by_id.get(qid,{}).get('annotation_variants',{}).get(group):
                errors.append({'id':qid,'error':'human_record_mismatch_'+group})
        if q.get('question')!=original_by_id.get(qid,{}).get('question'):
            errors.append({'id':qid,'error':'original_question_mismatch'})
        if not q.get('gold_chunk_ids') or any(cid not in chunks for cid in q.get('gold_chunk_ids',[])):
            errors.append({'id':qid,'error':'invalid_gold_chunks'})
        if not any(c.get('doc_id')==q.get('gold_doc_id') for c in
                   (chunks[cid] for cid in q.get('gold_chunk_ids',[]) if cid in chunks)):
            errors.append({'id':qid,'error':'gold_source_missing'})
        if any(chunks[cid].get('doc_id')!=q.get('gold_doc_id') for cid in q.get('gold_chunk_ids',[]) if cid in chunks):
            errors.append({'id':qid,'error':'gold_chunk_from_wrong_source_article'})
    if len(questions)!=1359 or len(seen)!=1359: errors.append({'error':'wrong_full_dataset_size'})
    subset_ids={q['id'] for q in subset}
    if len(subset_ids)!=1070 or not subset_ids<=seen: errors.append({'error':'invalid_1070_subset'})
    report={'n':len(questions),'n_unique_ids':len(seen),'n_source_articles':len({q['gold_doc_id'] for q in questions}),
        'question_type_counts':dict(Counter(q['question_type'] for q in questions)),
        'duplicate_question_texts':duplicate_questions,'errors':errors,'valid':not errors,
        'subset_n':len(subset_ids),'human_groups':2,'AI_gold_used':False,
        'ai_archive_exists':(PATHS.root.parent/'6127-data/6127-data/Data Annotation - AI').exists(),
        'adjudication_status':'pending_independent_human_review',
        'gold_matching':'Inherited lexical evidence alignment, not independent passage adjudication'}
    return questions,report,subset_ids


def cache_identity(dataset_sha,qid,config_id,model,parameters,prompt,context,pipeline_version,stage):
    return {'dataset_sha256':dataset_sha,'question_id':qid,'configuration_id':config_id,
            'model':model,'parameters':parameters,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
            'retrieved_context_sha256':digest(context),'pipeline_version':pipeline_version,'stage':stage}
