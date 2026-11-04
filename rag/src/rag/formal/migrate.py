"""Import only API outputs whose prompts and saved raw responses can be verified."""
from __future__ import annotations
import hashlib
import json
from rag import config
from rag.generator import ANSWER_PROMPT,SELF_CHECK_PROMPT,REVISE_PROMPT,STYLES,format_passages,normalize_citations
from rag.llm import parse_json
from .contract import Q4,PATHS,digest,sha,write_json
from .evaluation import qa_scores,citation_validity


def prompts_for(q,row,code):
    if code=='0':
        return [('closed_book',f"Answer the question. {STYLES['concise']}\nQUESTION: {q['question']}\nANSWER:",False)]
    passages=format_passages(row['hits'])
    prompts=[('answer',ANSWER_PROMPT.format(style=STYLES['concise'],passages=passages,question=q['question']),False),
             ('self_check',SELF_CHECK_PROMPT.format(passages=passages,answer=row['draft_answer']),True)]
    if row['self_check']['unsupported']:
        prompts.append(('revision',REVISE_PROMPT.format(unsupported='; '.join(row['self_check']['unsupported']),
                                                      passages=passages,answer=row['draft_answer']),False))
    return prompts


def verify_source(q,row,code):
    if row.get('reused_from'): raise ValueError('Earlier subset-scoped source excluded from strict formal import')
    if row['usage'].get('cache_hits',0): raise ValueError('Historical cached requests lack complete per-call usage provenance')
    if code=='G' and row['query']!=q['question']: raise ValueError('Historical query mismatch')
    prompts=prompts_for(q,row,code)
    requests=row['api_requests']
    if len(requests)!=len(prompts): raise ValueError('Historical call trace is incomplete')
    frames=[]
    for (stage,prompt,json_mode),request in zip(prompts,requests):
        prompt_sha=hashlib.sha256(prompt.encode()).hexdigest()
        if request['prompt_sha256']!=prompt_sha: raise ValueError('Saved API prompt hash does not match reconstructed input')
        cache_sha=hashlib.sha256(f'openai|{config.OPENAI_BASE_URL.rstrip("/")}|{config.LLM_MODEL}|0.0|{json_mode}|{prompt}'.encode()).hexdigest()
        path=Q4/'llm_cache'/cache_sha[:2]/(cache_sha+'.json')
        if not path.exists(): raise ValueError('Missing original raw API response cache')
        text=json.loads(path.read_text(encoding='utf-8'))['text']
        if stage=='closed_book' and text.strip()!=row['answer']: raise ValueError('Closed-book raw response mismatch')
        if stage=='answer' and normalize_citations(text)!=row['draft_answer']: raise ValueError('Draft raw response mismatch')
        if stage=='self_check' and parse_json(text).get('claims')!=row['self_check']['claims']: raise ValueError('Self-check raw response mismatch')
        if stage=='revision' and normalize_citations(text)!=row['answer']: raise ValueError('Revision raw response mismatch')
        body={'model':config.LLM_MODEL,'temperature':0.,'messages':[{'role':'user','content':prompt}]}
        if json_mode: body['response_format']={'type':'json_object'}
        frames.append({'stage':stage,'request':body,'raw_text':text,'raw_response_cache_sha256':sha(path),
                       'source_cache':str(path.relative_to(PATHS.root)),**request,
                       'response_metadata':'Original HTTP response id, finish_reason and resolved model were not retained; unavailable'})
    return frames


def import_compatible(out,questions,options,version):
    manifest=json.loads((Q4/'input_manifest.json').read_text(encoding='utf-8'))
    dataset_sha=sha(PATHS.eval_dir/'human_pair/eval_qa.jsonl')
    if (manifest['dataset_sha256']!=dataset_sha or manifest['model']!=config.LLM_MODEL or
          manifest['base_url']!=config.OPENAI_BASE_URL or manifest['provider']!=config.LLM_PROVIDER or
          manifest['temperature']!=0. or options.max_output_tokens is not None):
        return {'imported':0,'reason':'Source dataset/model/parameters incompatible; native API execution required'}
    from .contract import matrix
    configs=matrix()
    if manifest['retrieval']!=configs['G']['retrieval'] or manifest['generation']!=configs['G']['generation']:
        raise ValueError('Current Q4 G configuration no longer matches formal G')
    report={'imported':0,'rejected':[],'source_manifest_sha256':sha(Q4/'input_manifest.json')}
    by_id={q['id']:q for q in questions}
    for code,folder in [('G','generation'),('0','closed_book')]:
        target=out/'generation/records'/code; target.mkdir(parents=True,exist_ok=True)
        for source in sorted((Q4/folder).glob('*.json')):
            row=json.loads(source.read_text(encoding='utf-8')); qid=row['id']
            if qid not in by_id or (target/source.name).exists(): continue
            try: frames=verify_source(by_id[qid],row,code)
            except (ValueError,KeyError,TypeError) as e:
                report['rejected'].append({'id':qid,'configuration':code,'reason':str(e)})
                continue
            q=by_id[qid]; call_refs=[]
            for index,frame in enumerate(frames):
                call_id='import_'+digest({'source_sha256':sha(source),'index':index,'prompt_sha256':frame['prompt_sha256']})
                call_refs.append({'call_id':call_id,'stage':frame['stage'],'cache_hit':True,'source':'verified_existing_API'})
                write_json(out/'generation/imported_calls'/(call_id+'.json'),{
                    'call_id':call_id,'question_id':qid,'configuration_id':code,'dataset_sha256':dataset_sha,
                    'pipeline_version':version,'status':'success','source':'verified_existing_API',
                    'historical_source_file':str(source.relative_to(PATHS.root)),'historical_source_sha256':sha(source),
                    'source_manifest_sha256':report['source_manifest_sha256'],'billing_scope':'historical_not_new_request',**frame})
            result={**row,**qa_scores(row['answer'],q),**citation_validity(row['answer'],row.get('hits',[]),code=='0'),
                'id':qid,'question_id':qid,'configuration_id':code,'question_type':q['question_type'],
                'original_question':q['question'],'canonical_reference':q['answer'],
                'alternative_human_reference':q['annotation_variants']['human_reannotated']['answer'],
                'execution_status':'success','dataset_sha256':dataset_sha,'pipeline_version':version,
                'model':config.LLM_MODEL,'parameters':{'temperature':0.,'top_p':None,'max_tokens':None},
                'call_refs':call_refs,'retrieval_trace':[],'imported_from':str(source.relative_to(PATHS.root)),
                'import_verification':'All prompt hashes and raw model texts checked; cached/subset-origin outputs excluded',
                'online_end_to_end_seconds':None,'bulk_wall_seconds':row['wall_seconds']}
            write_json(target/source.name,result); report['imported']+=1
    write_json(out/'generation/import_report.json',report)
    return report
