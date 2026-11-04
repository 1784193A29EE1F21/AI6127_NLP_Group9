"""Recompute formal tables, uncertainty, cases and integrity checks from saved records."""
from __future__ import annotations
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from rag.config import PATHS
from rag.corpus import load_jsonl
from .contract import DATA,Q4,CONTRASTS,ROOT,digest,sha,write_json,matrix,check_matrix
from .evaluation import qa_scores,validate_judge,citation_validity
from .preflight import csv_file
from .reviews import annotation_report,judge_validation_sample
from .statistics import compare,holm

QA=['EM','token_precision','token_recall','F1','ROUGE_L']
RAG=['faithfulness','answer_relevance','answer_relevance_reverse_question_proxy','context_precision',
     'context_relevance','context_recall','context_recall_lexical','answer_correctness',
     'answer_correctness_multi','citation_validity','citation_support','citation_claim_coverage','answer_completeness','noncommittal']


def load_light(out):
    predictions={}; evaluations={}
    for code in matrix():
        predictions[code]={}; evaluations[code]={}
        for path in (out/'generation/records'/code).glob('*.json'):
            row=json.loads(path.read_text(encoding='utf-8'))
            predictions[code][row['id']]={k:row.get(k) for k in
                ['id','question_type','execution_status','error_details','answer','multi_reference','usage',
                 'bulk_wall_seconds','online_end_to_end_seconds','timings','imported_from']+QA}
        for path in (out/'evaluation/records'/code/'final').glob('*.json'):
            row=json.loads(path.read_text(encoding='utf-8'))
            evaluations[code][row['id']]={k:row.get(k) for k in
                ['id','question_type','execution_status','wall_seconds','usage','noncommittal']+RAG}
    return predictions,evaluations


def metrics_summary(predictions,evaluations,ids):
    summaries={}; by_type=[]
    for code,records in predictions.items():
        present={qid:r for qid,r in records.items() if qid in ids}
        success={qid:r for qid,r in present.items() if r['execution_status']=='success'}
        rows={qid:{**r,**(evaluations[code].get(qid,{}) if evaluations[code].get(qid,{}).get('execution_status')=='success' else {})}
              for qid,r in success.items()}
        metric={}
        for key in QA+RAG:
            values=[r[key] for r in rows.values() if r.get(key) is not None]
            type_means=[]
            for kind in ['definition','why','cause_effect','factual','comparison']:
                group=[r[key] for r in rows.values() if r['question_type']==kind and r.get(key) is not None]
                if group: type_means.append(float(np.mean(group)))
            metric[key]={'mean':float(np.mean(values)) if values else None,'n_scored':len(values),
                         'type_macro_mean':float(np.mean(type_means)) if len(type_means)==5 else None}
        multi={k:float(np.mean([r['multi_reference'][k] for r in success.values()])) if success else None for k in QA}
        summaries[code]={'n_target':len(ids),'n_records':len(present),'n_success':len(success),
            'n_judgments':sum(evaluations[code].get(qid,{}).get('execution_status')=='success' for qid in success),
            'n_failed':sum(r['execution_status']=='failed' for r in present.values()),'n_not_run':len(ids)-len(present),
            'coverage':len(success)/len(ids) if ids else None,'primary_and_RAG_metrics':metric,'multi_reference':multi}
        for kind in ['definition','why','cause_effect','factual','comparison']:
            group=[r for r in rows.values() if r['question_type']==kind]
            by_type.append({'configuration':code,'question_type':kind,'n_success':len(group),
                **{k:float(np.mean([r[k] for r in group if r.get(k) is not None])) if any(r.get(k) is not None for r in group) else None
                   for k in QA+RAG},**{k+'_n_scored':sum(r.get(k) is not None for r in group) for k in QA+RAG}})
    return summaries,by_type


def paired_statistics(out,predictions,evaluations,questions):
    by_id={q['id']:q for q in questions}; results=[]; distributions=[]
    for left,right in CONTRASTS:
        for key in QA+RAG:
            source=predictions if key in QA else evaluations
            ids=sorted(qid for qid in set(source[left])&set(source[right]) if
                source[left][qid].get('execution_status')=='success' and source[right][qid].get('execution_status')=='success'
                and source[left][qid].get(key) is not None and source[right][qid].get(key) is not None)
            if not ids: continue
            control=[source[left][qid][key] for qid in ids]; treatment=[source[right][qid][key] for qid in ids]
            result=compare(control,treatment,[by_id[qid]['gold_doc_id'] for qid in ids])
            results.append({'control':left,'treatment':right,'metric':key,**result})
            distributions.extend({'control':left,'treatment':right,'metric':key,'id':qid,
                 'source_article':by_id[qid]['gold_doc_id'],'delta':b-a} for qid,a,b in zip(ids,control,treatment))
    for key in QA+RAG:
        family=[r for r in results if r['metric']==key]
        for row,adjusted in zip(family,holm([r['p_value'] for r in family])):
            row['p_holm']=adjusted; row['holm_family']='all planned contrasts for '+key
            row['significant_holm_005']=adjusted<.05
    interaction=[]
    for key in QA+RAG:
        source=predictions if key in QA else evaluations
        ids=sorted(qid for qid in set.intersection(*(set(source[c]) for c in 'ABCD')) if all(
            source[c][qid].get('execution_status')=='success' and source[c][qid].get(key) is not None for c in 'ABCD'))
        if not ids: continue
        values=[source['D'][i][key]-source['B'][i][key]-source['C'][i][key]+source['A'][i][key] for i in ids]
        interaction.append({'metric':key,'definition':'D - B - C + A','exploratory':True,
            **compare([0.]*len(values),values,[by_id[i]['gold_doc_id'] for i in ids])})
    if results:
        flat=[{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in results]
        csv_file(out/'ablation/paired_comparisons.csv',list(flat[0]),flat)
    write_json(out/'ablation/bootstrap_confidence_intervals.json',{
        'comparisons':results,'interaction':interaction,'resamples':2000,'seed':42,
        'resampling_unit':'source article','Holm_family':'Per metric across all planned contrasts; not joint correction over outcomes',
        'primary_outcome':'canonical-reference F1','CI_note':'Percentile 95% intervals are unadjusted; p-values additionally corrected within outcome'})
    path=out/'ablation/paired_differences.jsonl'; path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        for row in distributions:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    return results


def mechanisms(out,usage=None):
    self_rows=[]; iteration_rows=[]; configs=matrix()
    for code,configuration in configs.items():
        if code=='0': continue
        for path in (out/'generation/records'/code).glob('*.json'):
            row=json.loads(path.read_text(encoding='utf-8')); qid=row['id']
            if row['execution_status']!='success': continue
            def evaluated(variant):
                p=out/'evaluation/records'/code/variant/(qid+'.json')
                return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
            final=evaluated('final')
            if configuration['generation']['self_check']:
                draft=evaluated('draft'); sc=row.get('self_check',{})
                before,after=draft.get('answer_correctness'),final.get('answer_correctness')
                self_rows.append({'id':qid,'configuration':code,'executed':bool(sc),
                    'unsupported_detected':bool(sc.get('unsupported')),'revision_requested':bool(sc.get('unsupported')),
                    'answer_changed':row.get('draft_answer')!=row['answer'],
                    'correctness_before':before,'correctness_after':after,
                    'faithfulness_before':draft.get('faithfulness'),'faithfulness_after':final.get('faithfulness'),
                    'repaired_to_correct':before is not None and after is not None and before<1 and after==1,
                    'correct_answer_corrupted':before==1 and after is not None and after<1,
                    'still_incorrect':after is not None and after<1,
                    'self_check_seconds':(row.get('timings') or {}).get('self_check'),
                    'draft_answer':row.get('draft_answer'),'final_answer':row['answer']})
            if configuration['generation']['iterative']:
                before=evaluated('before_iteration'); rounds=row.get('iterative_rounds',[])
                resolution=evaluated('iteration_resolution')
                trace=row.get('retrieval_trace',[])
                iteration_rows.append({'id':qid,'configuration':code,
                    'first_sufficient':rounds[0].get('sufficient') if rounds else None,
                    'first_missing':rounds[0].get('missing') if rounds else None,
                    'followup_query':rounds[0].get('follow_up_query') if rounds else None,
                    'followup_retrieval_count':max(0,len(trace)-1),
                    'missing_gap_resolved':resolution.get('resolved'),'resolution_reason':resolution.get('reason'),
                    'new_unique_chunks':len(row.get('additional_retrieved_chunks',[])),
                    'context_recall_before':before.get('context_recall'),'context_recall_after':final.get('context_recall'),
                    'F1_before':before.get('F1'),'F1_after':final.get('F1'),
                    'faithfulness_before':before.get('faithfulness'),'faithfulness_after':final.get('faithfulness'),
                    'correctness_before':before.get('answer_correctness'),'correctness_after':final.get('answer_correctness'),
                    'initial_query':row['original_question'],'initial_retrieved_chunks':row.get('initial_retrieved_chunks'),
                    'additional_retrieved_chunks':row.get('additional_retrieved_chunks'),
                    'final_context_ids':[h['chunk_id'] for h in row['hits']],
                    'answer_before_iteration':row.get('answer_before_iteration'),'answer_after_iteration':row['answer']})
    def export(folder,name,rows):
        path=out/folder/(name+'_per_question.jsonl'); path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w',encoding='utf-8') as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    export('self_check','self_check',self_rows); export('iterative','iterative_retrieval',iteration_rows)
    self_summary={}; iteration_summary={}
    for code in configs:
        group=[r for r in self_rows if r['configuration']==code]
        if group:
            self_summary[code]={'n':len(group),**{k:sum(bool(r[k]) for r in group) for k in
                ['executed','unsupported_detected','revision_requested','answer_changed','repaired_to_correct','correct_answer_corrupted']},
                'triggered_still_incorrect':sum(r['revision_requested'] and r['still_incorrect'] for r in group),
                'correctness_unknown_n':sum(r['correctness_before'] is None or r['correctness_after'] is None for r in group)}
            for key in ['faithfulness','correctness']:
                differences=[r[key+'_after']-r[key+'_before'] for r in group if r[key+'_before'] is not None and r[key+'_after'] is not None]
                self_summary[code][key+'_delta_mean']=float(np.mean(differences)) if differences else None
                self_summary[code][key+'_paired_n']=len(differences)
        group=[r for r in iteration_rows if r['configuration']==code]
        if group:
            iteration_summary[code]={'n':len(group),'first_sufficient_n':sum(r['first_sufficient'] is True for r in group),
                'first_insufficient_n':sum(r['first_sufficient'] is False for r in group),
                'followup_triggered_n':sum(r['followup_retrieval_count']>0 for r in group),
                'followup_retrieval_calls':sum(r['followup_retrieval_count'] for r in group),
                'new_unique_chunks':sum(r['new_unique_chunks'] for r in group)}
            iteration_summary[code]['new_evidence_resolved_gap_n']=sum(r['missing_gap_resolved'] is True for r in group)
            iteration_summary[code]['new_evidence_failed_to_resolve_n']=sum(r['missing_gap_resolved'] is False for r in group)
            iteration_summary[code]['gap_resolution_unknown_triggered_n']=sum(r['followup_retrieval_count']>0 and r['missing_gap_resolved'] is None for r in group)
            for key in ['context_recall','F1','faithfulness','correctness']:
                differences=[r[key+'_after']-r[key+'_before'] for r in group if r[key+'_before'] is not None and r[key+'_after'] is not None]
                iteration_summary[code][key+'_delta_mean']=float(np.mean(differences)) if differences else None
                iteration_summary[code][key+'_paired_n']=len(differences)
    for summaries,stages in [(self_summary,{'self_check','revision'}),(iteration_summary,{'evidence_assessment'})]:
        for code,entry in summaries.items():
            selected=[r for r in (usage or []) if r['configuration']==code and r['stage'] in stages]
            entry['direct_module_calls']=sum(r['calls'] for r in selected)
            entry['direct_module_input_tokens']=sum(r['input_tokens'] for r in selected)
            entry['direct_module_output_tokens']=sum(r['output_tokens'] for r in selected)
            entry['direct_module_cost_usd']=sum(r['cost_usd'] for r in selected) if selected and all(r['cost_usd'] is not None for r in selected) else None
            entry['cost_scope']='Direct module calls; counterfactual answers and judges are separate offline research costs.'
    write_json(out/'self_check/self_check_summary.json',self_summary)
    write_json(out/'iterative/iterative_retrieval_summary.json',iteration_summary)
    for name,folder,items in [('self_check','self_check',self_rows),('iterative_retrieval','iterative',iteration_rows)]:
        lines=['# '+name+'：真实案例','\n所有案例来自正式记录，不注入虚构错误。']
        if name=='self_check':
            categories=[('修复为正确答案',lambda r:r['repaired_to_correct']),
                        ('正确答案被误改',lambda r:r['correct_answer_corrupted']),
                        ('请求修订后仍有错误',lambda r:r['revision_requested'] and r['still_incorrect']),
                        ('答案改变但正确性未知',lambda r:r['answer_changed'] and (r['correctness_before'] is None or r['correctness_after'] is None))]
        else:
            categories=[('新增证据补齐缺失事实',lambda r:r['followup_retrieval_count']>0 and r['missing_gap_resolved'] is True),
                        ('新增证据未能补齐缺失事实',lambda r:r['followup_retrieval_count']>0 and r['missing_gap_resolved'] is False),
                        ('追加检索后的效果不确定',lambda r:r['followup_retrieval_count']>0 and r['missing_gap_resolved'] is None)]
        selected=[]
        for label,predicate in categories:
            candidates=sorted((r for r in items if predicate(r)),key=lambda r:(r['configuration'],r['id']))
            lines.append(f'\n## {label}：观察到{len(candidates)}条，按配置和ID展示前3条')
            for row in candidates[:3]:
                lines.append('\n```json\n'+json.dumps(row,ensure_ascii=False,indent=2)+'\n```')
            selected.extend(candidates[:3])
        if not selected:lines.append('当前没有观察到符合条件的真实案例。')
        (out/folder/(name+'_cases.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return self_summary,iteration_summary


def usage_report(out):
    counts=defaultdict(lambda:{'calls':0,'failed_attempts':0,'input_tokens':0,'output_tokens':0,'usage_unknown_calls':0,'cost_usd':0.})
    latest={}
    journal=out/'generation/api_calls.jsonl'
    if journal.exists():
        with journal.open(encoding='utf-8') as f:
            for line in f:
                entry=json.loads(line); identity=entry['identity']; usage=entry.get('usage') or {}
                latest[entry['call_id']]={'configuration':identity['configuration_id'],'stage':identity['stage'],
                    'status':entry['status'],'input_tokens':usage.get('prompt_tokens'),
                    'output_tokens':usage.get('completion_tokens'),'cost_usd':entry.get('cost_usd'),
                    'seconds':entry.get('seconds'),'billing_scope':'new_formal_requests',
                    'response_model':(entry.get('response') or {}).get('model'),
                    'system_fingerprint':(entry.get('response') or {}).get('system_fingerprint')}
    for path in (out/'generation/imported_calls').glob('*.json'):
        r=json.loads(path.read_text(encoding='utf-8'))
        latest[r['call_id']]={'configuration':r['configuration_id'],'stage':r['stage'],'status':r['status'],
            'input_tokens':r.get('prompt_tokens'),'output_tokens':r.get('output_tokens'),
            'cost_usd':None,'seconds':r.get('seconds'),'billing_scope':'historical_import',
            'response_model':None,'system_fingerprint':None}
    version_counts=Counter((r['configuration'],r['response_model'] or 'unavailable',r['system_fingerprint'] or 'unavailable') for r in latest.values())
    write_json(out/'performance/model_versions.json',{'observed':[{'configuration':c,'response_model':m,
        'system_fingerprint':f,'calls':n} for (c,m,f),n in version_counts.items()],
        'note':'Requested model is frozen; provider aliases or legacy missing metadata do not prove a pinned underlying weights version.'})
    for row in latest.values():
        group=counts[row['configuration'],row['stage'],row['billing_scope']]
        group['calls']+=1; group['failed_attempts']+=row['status']=='failed'
        group['input_tokens']+=row['input_tokens'] or 0; group['output_tokens']+=row['output_tokens'] or 0
        group['usage_unknown_calls']+=row['input_tokens'] is None or row['output_tokens'] is None
        group['cost_usd']+=row['cost_usd'] or 0.
    rows=[{'configuration':c,'stage':s,'billing_scope':b,**values} for (c,s,b),values in counts.items()]
    model=json.loads((out/'config/model_config.json').read_text(encoding='utf-8'))
    if not model['options'].get('price_source'):
        for row in rows:row['cost_usd']=None
    else:
        for row in rows:
            if row['usage_unknown_calls'] or row['billing_scope']=='historical_import':row['cost_usd']=None
    if rows:csv_file(out/'performance/token_usage.csv',list(rows[0]),rows)
    cost={'status':'unknown' if any(r['cost_usd'] is None for r in rows) or not rows else 'computed_from_configured_tariff',
        'cost_usd':None if any(r['cost_usd'] is None for r in rows) or not rows else sum(r['cost_usd'] for r in rows),
        'price_source':model['options'].get('price_source') or None,'raw_call_records':len(latest),
        'input_tokens_reported':sum(r['input_tokens'] for r in rows),'output_tokens_reported':sum(r['output_tokens'] for r in rows),
        'calls_with_unknown_usage':sum(r['usage_unknown_calls'] for r in rows),
        'historical_vs_new_separated':True,'formula':'input_tokens/1e6*verified_input_price + output_tokens/1e6*verified_output_price'}
    write_json(out/'performance/cost_report.json',cost)
    return rows,cost


def latency_report(out,predictions):
    result=[]
    for code,records in predictions.items():
        rows=[r for r in records.values() if r['execution_status']=='success']
        series={'bulk_generation': [r['bulk_wall_seconds'] for r in rows if r.get('bulk_wall_seconds') is not None],
                'fresh_online_end_to_end':[r['online_end_to_end_seconds'] for r in rows if r.get('online_end_to_end_seconds') is not None]}
        for stage in ['bm25','dense','fusion','rerank','rewrite','hyde','generate','self_check','assess','followup_retrieval']:
            series[stage]=[(r.get('timings') or {})[stage] for r in rows if stage in (r.get('timings') or {})
                and (stage not in ['bm25','dense','fusion','rerank'] or not r.get('imported_from'))]
        for stage,values in series.items():
            result.append({'configuration':code,'stage':stage,'n_measured':len(values),
                'mean_seconds':float(np.mean(values)) if values else None,
                **{f'p{p}_seconds':float(np.percentile(values,p)) if values else None for p in [50,90,95,99]},
                'note':'N/A if disabled, not executed, cached, or unavailable. Bulk execution is not fresh online inference.'})
    csv_file(out/'performance/latency_summary.csv',list(result[0]),result)
    return result


def analyze(out=ROOT):
    questions=load_jsonl(DATA); ids={q['id'] for q in questions}
    subset={q['id'] for q in load_jsonl(PATHS.eval_dir/'consensus_type/eval_qa.jsonl')}
    predictions,evaluations=load_light(out)
    summaries,types=metrics_summary(predictions,evaluations,ids)
    secondary,_=metrics_summary(predictions,evaluations,subset)
    write_json(out/'evaluation/metrics_summary.json',{'full_1359':summaries,'secondary_1070':secondary,
        'subset_source':'ID filtering of the same full-data outputs; no new calls','primary_reference':'human_original',
        'multi_reference':'Maximum per metric over two human references; auxiliary only'})
    csv_file(out/'evaluation/metrics_by_question_type.csv',list(types[0]),types)
    rows=[]
    for code,s in summaries.items():
        rows.append({'configuration':code,'n_target':1359,'success':s['n_success'],'failed':s['n_failed'],
            'not_run':s['n_not_run'],'coverage':s['coverage'],
            **{k:s['primary_and_RAG_metrics'][k]['mean'] for k in QA+RAG},
            **{'multi_'+k:v for k,v in s['multi_reference'].items()}})
    csv_file(out/'ablation/ablation_full_results.csv',list(rows[0]),rows)
    comparisons=paired_statistics(out,predictions,evaluations,questions)
    from .difficulty import analyze_challenges
    analyze_challenges(out,questions,predictions,evaluations)
    usage,cost=usage_report(out)
    self_summary,iteration_summary=mechanisms(out,usage)
    latency_report(out,predictions)
    agreement=annotation_report(out); judge_validation_sample(out,evaluations)
    # Streaming exports: do not read all long contexts and raw replies into memory.
    for name,folder in [('predictions','generation'),('qa_metrics_per_question','evaluation')]:
        target=out/('generation' if folder=='generation' else 'evaluation')/(name+'.jsonl')
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('w',encoding='utf-8') as f:
            for code in matrix():
                source=out/'generation/records'/code if folder=='generation' else out/'evaluation/records'/code/'final'
                for path in sorted(source.glob('*.json')):
                    row=json.loads(path.read_text(encoding='utf-8')); f.write(json.dumps(row,ensure_ascii=False)+'\n')
    from .reporting import reports,figures
    figures(out,summaries,types)
    reports(out,summaries,comparisons,self_summary,iteration_summary,cost,agreement)
    return summaries


def audit(out=ROOT,encoder=None,scope='all'):
    from .verification import verify
    return verify(out,encoder=encoder,scope=scope)
