"""Requirement-sized validation: inspect records, actual API inputs and derived tables."""
from __future__ import annotations
import hashlib
import csv
import json
from pathlib import Path
import numpy as np
from rag.config import PATHS
from rag.corpus import load_jsonl
from rag.generator import ANSWER_PROMPT,SELF_CHECK_PROMPT,REVISE_PROMPT,AGENT_PROMPT,STYLES,format_passages,normalize_citations
from rag.llm import parse_json
from rag.query import REWRITE_PROMPT,HYDE_PROMPT
from rag.formal.contract import DATA,ROOT,digest,sha,write_json
from .scope import matrix,check_matrix
from rag.formal.client import context_text
from rag.formal.evaluation import qa_scores,validate_judge,judge_metrics,judge_prompt,citation_validity


def expected_generation_prompt(stage,q,row):
    before=stage.startswith('before_iteration/')
    stage=stage.rsplit('/',1)[-1]
    initial=(row.get('retrieval_trace') or [{}])[0].get('hits',row.get('hits',[]))
    hits=initial if before else row.get('hits',[])
    passages=format_passages(hits)
    if stage=='closed_book':return f"Answer the question. {STYLES['concise']}\nQUESTION: {q['question']}\nANSWER:"
    if stage=='answer':return ANSWER_PROMPT.format(style=STYLES['concise'],passages=passages,question=q['question'])
    if stage=='rewrite':return REWRITE_PROMPT.format(query=q['question'])
    if stage=='hyde':return HYDE_PROMPT.format(query=q['question'])
    if stage=='evidence_assessment':return AGENT_PROMPT.format(passages=format_passages(initial),question=q['question'])
    draft=row.get('draft_answer_before_iteration') if before else row.get('draft_answer')
    if stage=='self_check':return SELF_CHECK_PROMPT.format(passages=passages,answer=draft)
    if stage=='revision':
        check=row.get('self_check_before_iteration') if before else row.get('self_check')
        return REVISE_PROMPT.format(passages=passages,answer=draft,unsupported='; '.join(check['unsupported']))
    raise ValueError('Unrecognized generation call stage: '+stage)


def equal_numeric(left,right,tolerance=1e-10):
    if left is None or right is None:return left is None and right is None
    return abs(float(left)-float(right))<=tolerance


def table_errors(path,expected_rows,identity_fields):
    """Check actual published CSV cells against recomputed per-question aggregates."""
    if not path.exists():return ['Missing published table: '+path.name]
    with path.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    def identity(row):return tuple(str(row[k]) for k in identity_fields)
    observed={identity(row):row for row in rows}
    expected={identity(row):row for row in expected_rows}
    errors=[]
    if len(observed)!=len(rows) or set(observed)!=set(expected):
        errors.append('Published table ID coverage/duplicates differ: '+path.name)
    for key,wanted in expected.items():
        actual=observed.get(key)
        if actual is None:continue
        for field,value in wanted.items():
            cell=actual.get(field)
            if value is None:valid=cell==''
            elif isinstance(value,(int,float)):
                try: valid=cell is not None and equal_numeric(float(cell),value)
                except (TypeError,ValueError): valid=False
            else:valid=cell==str(value)
            if not valid:errors.append(f'{path.name}/{key}: inconsistent {field}')
    return errors


def output_matches(stage,text,row):
    before=stage.startswith('before_iteration/'); stage=stage.rsplit('/',1)[-1]
    if stage=='closed_book':return text.strip()==row['answer']
    if stage=='answer':return normalize_citations(text)==row.get('draft_answer_before_iteration' if before else 'draft_answer')
    if stage=='self_check':
        check=row.get('self_check_before_iteration' if before else 'self_check') or {}
        return (parse_json(text) or {}).get('claims')==check.get('claims')
    if stage=='revision':return normalize_citations(text)==row.get('answer_before_iteration' if before else 'answer')
    if stage=='rewrite':
        raw=parse_json(text) or {}; rewritten=row.get('rewrite') or {}
        return raw.get('english_query','').strip()==rewritten.get('english_query') and raw.get('keywords')==rewritten.get('keywords')
    if stage=='hyde':return text.strip()==row.get('hyde')
    if stage=='evidence_assessment':
        raw=parse_json(text) or {}; actual=(row.get('iterative_rounds') or [{}])[0]
        return all(raw.get(k)==actual.get(k) for k in ['sufficient','missing']) and raw.get('follow_up_query','').strip()==actual.get('follow_up_query')
    return False


def expected_resolution_prompt(q,row):
    initial=row['retrieval_trace'][0]['hits']; initial_ids={h['chunk_id'] for h in initial}
    added=[h for h in row['hits'] if h['chunk_id'] not in initial_ids]
    missing=(row.get('iterative_rounds') or [{}])[0].get('missing','')
    return ('You are evaluating whether follow-up retrieval fills an actual stated evidence gap. '
        'Do not assume the follow-up succeeds. Use only the passages. Return JSON: '
        '{"resolved":true/false/null,"reason":"explanation","supporting_added_passage_numbers":[1,...]}. '
        'null means uncertain. Passage numbers refer to ADDED PASSAGES only.\n\n'
        f"QUESTION: {q['question']}\nSTATED GAP: {missing}\n\nINITIAL PASSAGES:\n{format_passages(initial)}\n\n"
        f"ADDED PASSAGES:\n{format_passages(added)}")


def verify(out=ROOT,encoder=None,scope='all'):
    configs=matrix(); check_matrix(configs)
    if scope not in ['all','Q4']:raise ValueError('Unknown verification scope')
    selected_codes=['0','A','G'] if scope=='Q4' else list(configs)
    audit_name='Q4_completion_audit.json' if scope=='Q4' else 'completion_audit.json'
    questions=load_jsonl(DATA); by_id={q['id']:q for q in questions}; expected=set(by_id)
    errors=[]; groups={}; proxy_checks=[]; trace_checked=set()
    frozen_path=out/'config/frozen_experiment.json'
    if not frozen_path.exists():
        result={'automated_complete':False,'all_requirements_complete':False,'errors':['Formal execution has not frozen its inputs yet']}
        write_json(out/'reports'/audit_name,result); return result
    frozen=json.loads(frozen_path.read_text(encoding='utf-8'))
    scope_path=out/'config/scope_manifest.json'
    if not scope_path.exists():errors.append('Missing six-group scope provenance')
    else:
        scope_manifest=json.loads(scope_path.read_text(encoding='utf-8'))
        if scope_manifest.get('configuration_ids')!=list(configs):errors.append('Unexpected six-group scope')
        for relative,checksum in scope_manifest.get('scope_code_sha256',{}).items():
            if sha(PATHS.root/relative)!=checksum:errors.append('Six-group adapter changed: '+relative)
    if len(expected)!=1359 or sha(DATA)!=frozen['dataset_sha256']: errors.append('Dataset missing, duplicated or changed')
    if configs!=frozen['matrix']:errors.append('Ablation matrix differs from frozen configuration')
    for relative,checksum in frozen['code_sha256'].items():
        if sha(PATHS.root/relative)!=checksum:errors.append('Code changed after freeze: '+relative)
    data_manifest=json.loads((out/'data_validation/dataset_manifest.json').read_text(encoding='utf-8'))
    if set(data_manifest['ids'])!=expected or len(data_manifest['ids'])!=1359:errors.append('Frozen ID set is not full dataset')
    human_root=PATHS.root.parent/'6127-data/6127-data'
    for relative,checksum in data_manifest['human_source_files_sha256'].items():
        if sha(human_root/relative)!=checksum:errors.append('Raw human annotation changed: '+relative)
    validation=json.loads((out/'data_validation/dataset_validation_report.json').read_text(encoding='utf-8'))
    for relative,checksum in validation.get('index_hashes_verified',{}).items():
        if sha(PATHS.root/'data'/relative)!=checksum:errors.append('Index changed: '+relative)
    content_fields=['chunk_id','doc_id','title','section','url','text']
    corpus_hashes={}
    with PATHS.chunks_file.open(encoding='utf-8') as f:
        for line in f:
            c=json.loads(line); corpus_hashes[c['chunk_id']]=digest({k:c.get(k,'') for k in content_fields})
    baseline={}
    legacy=PATHS.results_dir/'ablation_full_2026-11-02'
    for code,name in {'A':'Hybrid_RRF','B':'Hybrid_RRF_plus_rerank','G':'Hybrid_RRF_plus_rerank',
                       'H':'Hybrid_RRF_plus_PRF_plus_rerank','J':'Hybrid_RRF_plus_rerank_plus_diversity'}.items():
        baseline[code]={r['id']:r['ranked'][:5] for r in load_jsonl(legacy/(name+'.trec.jsonl'))}

    def trace(ref,code,qid,wanted_prompt=None):
        imported=ref.get('source')=='verified_existing_API'
        path=out/'generation'/('imported_calls' if imported else 'raw_calls')/(ref['call_id']+'.json')
        if not path.exists():errors.append(f'{code}/{qid}: missing raw call {ref["call_id"]}'); return
        call=json.loads(path.read_text(encoding='utf-8'))
        messages=call.get('request',{}).get('messages',[])
        if len(messages)!=1 or messages[0].get('role')!='user':errors.append(f'{code}/{qid}: stateful or unexpected request');return
        prompt=messages[0]['content']
        if wanted_prompt is not None and prompt!=wanted_prompt:errors.append(f'{code}/{qid}: prompt differs from reference-free generation/judge contract')
        if call.get('status')!='success':errors.append(f'{code}/{qid}: successful record points to failed request')
        if call['request'].get('model')!=frozen['model'] or call['request'].get('temperature')!=0.:
            errors.append(f'{code}/{qid}: mixed model or temperature')
        if call['request'].get('max_tokens')!=frozen['parameters']['max_tokens']:errors.append(f'{code}/{qid}: mixed max_tokens')
        if not imported:
            identity=call['identity']
            if identity['question_id']!=qid or identity['configuration_id']!=code or identity['dataset_sha256']!=frozen['dataset_sha256']:
                errors.append(f'{code}/{qid}: cache scope mismatch')
            if identity['pipeline_version']!=frozen['pipeline_version'] or digest(identity)!=call['cache_key']:
                errors.append(f'{code}/{qid}: cache version/key mismatch')
            if hashlib.sha256(prompt.encode()).hexdigest()!=identity['prompt_sha256'] or digest(context_text(prompt))!=identity['retrieved_context_sha256']:
                errors.append(f'{code}/{qid}: input/context hash mismatch')
        else:
            if call['question_id']!=qid or call['configuration_id']!=code:errors.append(f'{code}/{qid}: imported scope mismatch')
            cache=PATHS.root/call['source_cache']
            if not cache.exists() or sha(cache)!=call['raw_response_cache_sha256']:
                errors.append(f'{code}/{qid}: original imported response no longer verifies')
            else:
                actual=json.loads(cache.read_text(encoding='utf-8'))['text']
                if actual!=call['raw_text']:errors.append(f'{code}/{qid}: imported model text changed')
                is_json='response_format' in call['request']
                wanted=hashlib.sha256(f"openai|{frozen['endpoint'].rstrip('/')}|{frozen['model']}|0.0|{is_json}|{prompt}".encode()).hexdigest()
                if cache.stem!=wanted:errors.append(f'{code}/{qid}: legacy prompt/model cache key mismatch')
        trace_checked.add(ref['call_id'])
        return call

    for code in selected_codes:
        cfg=configs[code]
        seen=set(); success=failures=judged=0
        for path in (out/'generation/records'/code).glob('*.json'):
            row=json.loads(path.read_text(encoding='utf-8')); qid=row.get('id')
            if qid in seen:errors.append(f'{code}: duplicate ID {qid}')
            seen.add(qid)
            if qid not in expected:errors.append(f'{code}: extra ID {qid}');continue
            q=by_id[qid]
            if row.get('pipeline_version')!=frozen['pipeline_version'] or row.get('dataset_sha256')!=frozen['dataset_sha256']:
                errors.append(f'{code}/{qid}: mixed output version')
            if row.get('execution_status')=='failed':
                failures+=1
                if not row.get('error_details'):errors.append(f'{code}/{qid}: failed without a reason')
                continue
            success+=1
            if not row.get('answer','').strip() or not row.get('call_refs'):errors.append(f'{code}/{qid}: empty answer or no provenance')
            for hit in row.get('hits',[]):
                if corpus_hashes.get(hit.get('chunk_id'))!=digest({k:hit.get(k,'') for k in content_fields}):
                    errors.append(f'{code}/{qid}: retrieved evidence differs from frozen Wikipedia corpus')
            trace_rows=row.get('retrieval_trace',[])
            if code in baseline and [h['chunk_id'] for h in row.get('hits',[])]!=baseline[code].get(qid):
                errors.append(f'{code}/{qid}: reusable baseline ranking changed')
            if trace_rows:
                if trace_rows[0]['retrieval']!=cfg['retrieval'] or trace_rows[0]['query']!=q['question']:
                    errors.append(f'{code}/{qid}: initial retrieval configuration/query changed')
                if len(trace_rows)>2:errors.append(f'{code}/{qid}: exceeded two-round retrieval bound')
                if len(trace_rows)>1:
                    follow_cfg={**cfg['retrieval'],'rewrite':False,'hyde':False}
                    if not cfg['generation']['iterative'] or trace_rows[1]['retrieval']!=follow_cfg:
                        errors.append(f'{code}/{qid}: invalid follow-up retrieval parameters')
                    if trace_rows[1]['query']!=(row.get('iterative_rounds') or [{}])[0].get('follow_up_query'):
                        errors.append(f'{code}/{qid}: follow-up query does not match actual evidence assessment')
            if row.get('canonical_reference')!=q['answer'] or row.get('alternative_human_reference')!=q['annotation_variants']['human_reannotated']['answer']:
                errors.append(f'{code}/{qid}: human references changed')
            for ref in row.get('call_refs',[]):
                try: prompt=expected_generation_prompt(ref['stage'],q,row)
                except (ValueError,KeyError,TypeError) as error:
                    errors.append(f'{code}/{qid}: cannot reconstruct {ref["stage"]}: {error}');continue
                call=trace(ref,code,qid,prompt)
                if call:
                    text=call.get('raw_text') if ref.get('source')=='verified_existing_API' else call.get('response',{}).get('choices',[{}])[0].get('message',{}).get('content','')
                    if not output_matches(ref['stage'],text,row):errors.append(f'{code}/{qid}: saved model text does not match generated intermediate/final answer')
            if code!='0':
                if cfg['generation']['self_check']:
                    if row['answer']!=(row.get('self_check') or {}).get('revised_answer'):
                        errors.append(f'{code}/{qid}: final answer is not the actual self-check result')
                    if not (row.get('self_check') or {}).get('unsupported') and row['answer']!=row.get('draft_answer'):
                        errors.append(f'{code}/{qid}: answer changed without a recorded revision')
                elif row['answer']!=row.get('draft_answer'):errors.append(f'{code}/{qid}: final answer differs from the unmodified model draft')
            scores=qa_scores(row['answer'],q)
            for key in ['EM','token_precision','token_recall','F1','ROUGE_L']:
                if not equal_numeric(scores[key],row.get(key)) or not equal_numeric(scores['multi_reference'][key],row.get('multi_reference',{}).get(key)):
                    errors.append(f'{code}/{qid}: inconsistent {key}')
            valid=citation_validity(row['answer'],row.get('hits',[]),code=='0')
            if not equal_numeric(valid['citation_validity'],row.get('citation_validity')):errors.append(f'{code}/{qid}: citation validity inconsistent')
            variants=['final']
            if code!='0' and cfg['generation']['self_check']:variants.append('draft')
            if code!='0' and cfg['generation']['iterative']:variants.append('before_iteration')
            for variant in variants:
                ep=out/'evaluation/records'/code/variant/(qid+'.json')
                if not ep.exists():errors.append(f'{code}/{qid}: missing {variant} judgment');continue
                e=json.loads(ep.read_text(encoding='utf-8'))
                if e.get('execution_status')!='success':errors.append(f'{code}/{qid}: failed {variant} judgment');continue
                if variant=='final':answer,hits=row['answer'],row.get('hits',[]);judged+=1
                elif variant=='draft':answer,hits=row['draft_answer'],row['hits']
                else:answer,hits=row['answer_before_iteration'],row['retrieval_trace'][0]['hits']
                try:validate_judge(e['raw_judgment'],len(hits))
                except ValueError as error:errors.append(f'{code}/{qid}: invalid {variant}: {error}');continue
                derived=judge_metrics(e['raw_judgment'],q,answer,hits)
                for key,v in derived.items():
                    if isinstance(v,(list,str)) or key=='answer_relevance_reverse_question_proxy':continue
                    if not equal_numeric(v,e.get(key)):errors.append(f'{code}/{qid}: derived judge metric mismatch: {key}')
                expected_qa=qa_scores(answer,q)
                for key in ['EM','token_precision','token_recall','F1','ROUGE_L']:
                    if not equal_numeric(expected_qa[key],e.get(key)) or not equal_numeric(expected_qa['multi_reference'][key],e.get('multi_reference',{}).get(key)):
                        errors.append(f'{code}/{qid}: {variant} QA metric mismatch: {key}')
                for ref in e.get('call_refs',[]):
                    call=trace(ref,code,qid,judge_prompt(q,answer,hits))
                    if call:
                        text=call.get('raw_text') if ref.get('source')=='verified_existing_API' else call.get('response',{}).get('choices',[{}])[0].get('message',{}).get('content','')
                        if parse_json(text)!=e['raw_judgment']:errors.append(f'{code}/{qid}: judgment differs from actual model response')
                if not e.get('call_refs'):errors.append(f'{code}/{qid}: untraceable {variant} judge')
                if variant=='final':proxy_checks.append((q['question'],e['raw_judgment'],e.get('answer_relevance_reverse_question_proxy'),code,qid))
            if code!='0' and cfg['generation']['iterative'] and len(row.get('retrieval_trace',[]))>1:
                rp=out/'evaluation/records'/code/'iteration_resolution'/(qid+'.json')
                if not rp.exists():errors.append(f'{code}/{qid}: missing follow-up evidence resolution')
                else:
                    resolution=json.loads(rp.read_text(encoding='utf-8'))
                    if resolution.get('execution_status')!='success':errors.append(f'{code}/{qid}: failed follow-up evidence resolution')
                    initial_ids={h['chunk_id'] for h in row['retrieval_trace'][0]['hits']}
                    added=[h for h in row['hits'] if h['chunk_id'] not in initial_ids]
                    refs=resolution.get('call_refs',[])
                    if not added:
                        if resolution.get('resolved') is not False or refs or resolution.get('supporting_added_passage_numbers')!=[]:
                            errors.append(f'{code}/{qid}: no added evidence cannot resolve the gap')
                    else:
                        if not refs:errors.append(f'{code}/{qid}: evidence resolution lacks actual judge call')
                        for ref in refs:
                            call=trace(ref,code,qid,expected_resolution_prompt(q,row))
                            if call:
                                raw=parse_json(call.get('response',{}).get('choices',[{}])[0].get('message',{}).get('content','')) or {}
                                if any(raw.get(k)!=resolution.get(k) for k in ['resolved','reason','supporting_added_passage_numbers']):
                                    errors.append(f'{code}/{qid}: evidence resolution differs from actual model response')
                        numbers=resolution.get('supporting_added_passage_numbers')
                        if not isinstance(numbers,list) or any(type(n) is not int or n<1 or n>len(added) for n in numbers):
                            errors.append(f'{code}/{qid}: invalid added-passage evidence attribution')
                        elif resolution.get('resolved') is True and not numbers:
                            errors.append(f'{code}/{qid}: resolved gap has no supporting additional evidence')
        if seen!=expected:errors.append(f'{code}: missing {len(expected-seen)} IDs; extra {len(seen-expected)}')
        if failures:errors.append(f'{code}: {failures} generation failures still require recovery')
        groups[code]={'records':len(seen),'success':success,'failed':failures,'judged':judged}

    # Reproduce the reverse-question embedding proxy in batches; no additional model API calls.
    if proxy_checks:
        if encoder is None:
            from rag.dense import DenseEncoder
            encoder=DenseEncoder(frozen['embedding_model'])
        texts=list(dict.fromkeys(text for q,raw,_,_,_ in proxy_checks if not raw['noncommittal'] for text in [q]+raw['generated_questions']))
        vectors=encoder.encode_queries(texts,batch_size=32) if texts else []
        positions={text:i for i,text in enumerate(texts)}
        for q,raw,actual,code,qid in proxy_checks:
            wanted=0. if raw['noncommittal'] else float(np.mean([vectors[positions[t]]@vectors[positions[q]] for t in raw['generated_questions']]))
            if not equal_numeric(wanted,actual,1e-5):errors.append(f'{code}/{qid}: reverse-question proxy cannot be reproduced')
    from rag.formal.analysis import load_light,metrics_summary
    predictions,evaluations=load_light(out)
    full,type_rows=metrics_summary(predictions,evaluations,expected)
    subset=set(data_manifest['subset_ids']); sub,_=metrics_summary(predictions,evaluations,subset)
    summary_path=out/'evaluation/metrics_summary.json'
    if summary_path.exists():
        saved=json.loads(summary_path.read_text(encoding='utf-8'))
        if digest(saved['full_1359'])!=digest(full) or digest(saved['secondary_1070'])!=digest(sub):
            errors.append('Aggregate full/subset tables differ from recomputed per-question records')
    else:errors.append('Missing aggregate summary')
    errors.extend(table_errors(out/'evaluation/metrics_by_question_type.csv',type_rows,['configuration','question_type']))
    from rag.formal.analysis import QA,RAG
    ablation_rows=[{'configuration':code,'n_target':1359,'success':s['n_success'],'failed':s['n_failed'],
        'not_run':s['n_not_run'],'coverage':s['coverage'],
        **{key:s['primary_and_RAG_metrics'][key]['mean'] for key in QA+RAG},
        **{'multi_'+key:value for key,value in s['multi_reference'].items()}} for code,s in full.items()]
    errors.extend(table_errors(out/'ablation/ablation_full_results.csv',ablation_rows,['configuration']))
    profiles=out/'performance/fresh_profiles.json'
    if not profiles.exists():errors.append('Fresh end-to-end/scaling benchmark not completed')
    else:
        conditions=json.loads(profiles.read_text(encoding='utf-8'))['conditions']
        observed={(r['configuration'],r['workers']) for r in conditions}
        required={(c,4) for c in (['0','A','G'] if scope=='Q4' else ['0','A','B','G','S','T'])}|{('G',1),('G',2)}
        if not required<=observed or any(r['n']!=10 for r in conditions):errors.append('Incomplete performance conditions')
    def optional(path,default):return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default
    human=optional(out/'data_validation/annotation_agreement_report.json',{'status':'missing'})
    judge=optional(out/'data_validation/judge_validation_status.json',{'status':'missing'})
    human_ok=human.get('status')=='complete' and (human.get('independent_human_answer_agreement') or 0)>=.8
    judge_ok=judge.get('status')=='complete' and judge.get('n_human_reviewed',0)>=100
    result={'automated_complete':not errors,'all_requirements_complete':not errors and human_ok and judge_ok,
        'errors':errors,'groups':groups,'n_API_traces_checked':len(trace_checked),
        'scope':scope,'expected_configuration_questions':1359*len(selected_codes),'independent_human_annotation_review':human.get('status'),
        'answer_human_agreement_threshold_met':human_ok,'judge_human_validation':judge.get('status'),
        'human_pending_tasks':[] if human_ok and judge_ok else ['Independent human answer equivalence review and/or 100-answer judge validation'],
        'historical_chat_pilot_used':False,'secondary_subset_recomputed_from_full':True}
    write_json(out/'reports'/audit_name,result)
    lines=['# 完整性核查','',f'自动实验校验通过：{result["automated_complete"]}',
           f'全部任务完成：{result["all_requirements_complete"]}','',
           '独立人工审核与judge人工核验未完成时，不标记全部完成。','',
           '| 配置 | 记录 | 真实成功 | 失败 | 已评分 |','|---|---:|---:|---:|---:|']
    for c,r in groups.items():lines.append(f'| {c} | {r["records"]} | {r["success"]} | {r["failed"]} | {r["judged"]} |')
    lines+=['','## 未通过项目']+[f'- {e}' for e in errors]+['','## 人工待办']+[f'- {e}' for e in result['human_pending_tasks']]
    checklist='Q4_COMPLETION_CHECKLIST.md' if scope=='Q4' else 'EXPERIMENT_COMPLETION_CHECKLIST.md'
    (out/'reports'/checklist).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
