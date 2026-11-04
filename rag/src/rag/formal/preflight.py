from __future__ import annotations
from collections import Counter
from dataclasses import asdict
from datetime import datetime,timezone
import csv
import platform
import shutil
import sys
import importlib.metadata
from rag import config
from rag.config import PATHS
from rag.corpus import load_jsonl
from rag.metrics import exact_match,token_f1
from .client import Options
from .contract import ROOT,DATA,Q4,check_matrix,digest,matrix,sha,validate_dataset,write_json


def environment_versions():
    versions={}
    for name in ['numpy','scipy','torch','transformers','sentence-transformers','httpx','python-dotenv',
                 'psutil','matplotlib','scikit-learn','pytest','tokenizers','huggingface-hub']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    return versions


def csv_file(path,fields,rows,overwrite=True):
    if path.exists() and not overwrite: return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def run(out=ROOT):
    out.mkdir(parents=True,exist_ok=True)
    questions,validation,subset=validate_dataset()
    configs=matrix(); check_matrix(configs); options=Options.from_env()
    checks=out/'data_validation'; checks.mkdir(exist_ok=True)
    write_json(checks/'dataset_validation_report.json',validation)
    if not validation['valid']: raise ValueError('Dataset preflight failed: see data_validation/dataset_validation_report.json')
    source_root=PATHS.root.parent/'6127-data/6127-data'
    source_hashes={}
    source_groups={g:{} for g in ['human_original','human_reannotated']}
    for name in ['Data Annotation - original 00-49','Data Annotation - original 50-99','Data Annotation - Reannotated']:
        for path in sorted((source_root/name).rglob('*_qa.json')):
            source_hashes[str(path.relative_to(source_root))]=sha(path)
            import json
            group='human_reannotated' if name=='Data Annotation - Reannotated' else 'human_original'
            for row in json.loads(path.read_text(encoding='utf-8-sig')):
                if row['id'] in source_groups[group]: raise ValueError('Duplicate original human ID')
                source_groups[group][row['id']]=row
    if len(source_hashes)!=200: raise ValueError('Expected 200 original human annotation files')
    for q in questions:
        for group,raw in source_groups.items():
            row=raw.get(q['id'],{})
            if row.get('question','').strip()!=q['question'].strip() or any(
                row.get(k)!=q['annotation_variants'][group][k] for k in ['answer','evidence','question_type']):
                raise ValueError('Human-only dataset no longer matches raw human source files')
    legacy_manifest=json.loads((PATHS.eval_dir/'dataset_manifest.json').read_text(encoding='utf-8'))
    index_hashes={}
    for relative,expected in legacy_manifest['files_sha256'].items():
        if relative.replace('\\','/').startswith('index/'):
            actual=sha(PATHS.root/'data'/relative)
            if actual!=expected: raise ValueError('Index input changed: '+relative)
            index_hashes[relative]=actual
    retrieval_manifest=json.loads((PATHS.results_dir/'retrieval_full_2026-11-02/run_manifest.json').read_text(encoding='utf-8'))
    retrieval_core={'src/rag/'+name for name in ['pipeline.py','bm25.py','dense.py','fusion_rerank.py',
                                                 'query.py','text.py','metrics.py','config.py']}
    unrelated_changes=[]
    for relative,expected in retrieval_manifest['code_sha256'].items():
        path=PATHS.root/relative
        if path.exists() and sha(path)!=expected:
            if relative.replace('\\','/') in retrieval_core:
                raise ValueError('Historical retrieval implementation changed: '+relative)
            unrelated_changes.append(relative)
    validation['original_human_files_verified']=200
    validation['original_human_alignment_verified']=True
    validation['index_hashes_verified']=index_hashes
    validation['changed_non_retrieval_files']=unrelated_changes
    write_json(checks/'dataset_validation_report.json',validation)
    manifest={'dataset_sha256':sha(DATA),'n':1359,'ids':[q['id'] for q in questions],
              'source_articles':331,'human_groups':2,'subset_ids':sorted(subset),
              'corpus_sha256':sha(PATHS.chunks_file),'human_source_files_sha256':source_hashes,
              'source_dataset_sha256':sha(PATHS.eval_file),'legacy_dataset_manifest_sha256':sha(PATHS.eval_dir/'dataset_manifest.json')}
    manifest_path=checks/'dataset_manifest.json'
    if manifest_path.exists():
        import json
        if json.loads(manifest_path.read_text(encoding='utf-8'))!=manifest: raise ValueError('Frozen data changed')
    else: write_json(manifest_path,manifest)
    write_json(out/'config/ablation_matrix.json',configs)
    write_json(out/'config/model_config.json',{'provider':config.LLM_PROVIDER,'requested_model':config.LLM_MODEL,
        'endpoint':config.OPENAI_BASE_URL,'temperature':0.,'top_p':None,'max_tokens':options.max_output_tokens,
        'parameter_note':'Unspecified top_p, seed and max_tokens use provider defaults. Actual response metadata is archived.',
        'options':asdict(options),'key_present':bool(__import__('os').environ.get('OPENAI_API_KEY')),
        'key_source':'local environment/.env; never copied to experiment artifacts',
        'price_status':'unknown' if not options.price_source else 'configured; source recorded',
        'independent_judge_model_available':False,'judge':'Same model, anonymous independent stateless calls'})
    write_json(out/'config/generation_config.json',{'final_Q4_system':'G','Q4_comparison':['0','A','G'],
        'primary_reference':'human_original','secondary_reference':'human_reannotated',
        'AI_reference_used':False,'temperature':0.,'all_questions_required':True})
    agreement=[]; review=[]
    for q in questions:
        a,b=[q['annotation_variants'][g] for g in ['human_original','human_reannotated']]
        agreement.append({'id':q['id'],'EM':exact_match(a['answer'],b['answer']),
                          'F1':token_f1(a['answer'],b['answer']),'type_agree':a['question_type']==b['question_type']})
        review.append({'question_id':q['id'],'question_type':q['question_type'],'question':q['question'],
            'answer_A':a['answer'],'answer_B':b['answer'],'source_url':q['source_url'],
            'suggested_label':'','suggested_reason':'','human_label':'','human_reviewer':'','human_reason':'','review_date':''})
    csv_file(checks/'annotation_review_queue.csv',list(review[0]),review)
    csv_file(checks/'annotation_adjudication_template.csv',list(review[0])+['canonical_correction','adjudication_status'],
             [{**r,'canonical_correction':'','adjudication_status':'pending'} for r in review],overwrite=False)
    write_json(checks/'annotation_agreement_report.json',{
        'n':1359,'type_agreement_n':sum(r['type_agree'] for r in agreement),
        'type_agreement':sum(r['type_agree'] for r in agreement)/1359,
        'answer_EM':sum(r['EM'] for r in agreement)/1359,'answer_token_F1':sum(r['F1'] for r in agreement)/1359,
        'independent_human_review_n':0,'independent_human_answer_agreement':None,'independent_human_ci95':None,
        'status':'pending_independent_human_review','LLM_assistance':'reported separately when available',
        'warning':'Type agreement and lexical overlap do not establish semantic answer agreement >=80%'})
    supplied=source_root/'report_en.md'
    if supplied.exists():
        write_json(checks/'supplied_agreement_claim_provenance.json',{
            'source':str(supplied.relative_to(PATHS.root.parent)),'sha256':sha(supplied),
            'claimed_semantic_agreement':1331/1359,'claimed_n_agree':1331,'claimed_n_total':1359,
            'item_level_independent_human_review_records_found':False,
            'verification_status':'Unverified supplied aggregate; not adopted as verified answer IAA',
            'note':'The report states semantic agreement but does not contain per-question labels or reviewer provenance.'})
    (checks/'annotation_guidelines.md').write_text(
        '# 两个人工答案的独立审核规则\n\n保留原始答案；不将AI辅助判断当作人工审核。\n\n'
        '- agree：核心意义一致，覆盖问题所需部分；兼容的额外细节不影响一致。\n'
        '- partial：有共同核心，但遗漏必要条件或关键部分。\n'
        '- disagree：核心事实矛盾，或有效答案与拒答不同。\n'
        '- uncertain：不足以确定；不强行归类。\n\n'
        '独立审核者填写human_label、human_reviewer、human_reason、review_date。'
        '如需裁决，填写canonical_correction和adjudication_status，不能直接覆盖源标注。'
        '全量独立审核完成前，答案语义一致率和置信区间保持pending。模型建议只填suggested列。\n',encoding='utf-8')
    baseline_calls=sum([1,1,1,2,2,3,3,2,1,2,1,5,5,5,4,4,5,4,4])*1359
    estimate={'generation_calls_base':baseline_calls,'final_judge_calls':19*1359,
        'additional_calls':'Revisions, before-iteration answers/judges and changed-draft judges add calls',
        'total_call_estimate':[110000,145000],'estimated_input_tokens':[120000000,250000000],
        'estimated_output_tokens':[20000000,70000000],'cost_usd':None,'price_status':'unknown',
        'rate_limit_rpm':options.rpm,'rate_limit_only_hours':[(110000/(options.rpm or 30))/60,(145000/(options.rpm or 30))/60],
        'wall_clock_estimate_hours':[96,144], 'note':'Estimate only, not measured or billed usage; reuse reduces calls',
        'created_at':datetime.now(timezone.utc).isoformat()}
    write_json(out/'performance/preflight_estimate.json',estimate)
    functions={
        'implemented':['BM25','dense multilingual encoder','Hybrid RRF','cross-encoder rerank','PRF',
            'query rewriting','HyDE','diversity','citation generation','self-check','iterative retrieval','19-config matrix'],
        'existing_full_experiments':'10 retrieval configurations, all 1359; hash validation required before reuse',
        'historical_generation':'19x10 chat pilot excluded; current API Q4 tracked independently',
        'missing_before_this_goal':['Full 19x1359 generation/judging','complete citation-support evaluation',
            'independent human reviews','formal before/after mechanism analyses','full paired statistical analysis'],
        'new_work':['Strict per-input API provenance/cache','formal validation and adjudication templates',
            'full-coverage runner','anonymous structured scoring','cluster bootstrap/Holm','reports and completeness audit'],
        'environment':{'python':sys.version,'platform':platform.platform(),'disk_free_bytes':shutil.disk_usage(out).free,
                       'library_versions':environment_versions()},
        'execution_order':['Finish current Q4 task','Smoke preflight','Import only proven compatible API runs',
            'Complete 19 full generation groups','Evaluate all outputs and mechanism intermediates',
            'Paired statistics and cases','Strict audit and report'],
        'human_review':'Pending; never substituted by AI','api':'Existing endpoint is live in current Q4 process; no extra preflight call made'}
    write_json(out/'preflight_report.json',functions)
    write_json(out/'config/environment_versions.json',functions['environment'])
    (out/'IMPLEMENTATION_PLAN.md').write_text(
        '# 正式Q4/Q5执行计划\n\n'
        '1. 验证冻结的1359题、1070题子集、200个人工源文件与索引。\n'
        '2. 等待当前Q4进程终止并核查结果，不重复启动。\n'
        '3. 小规模端到端验证后，复用可严格追溯的0/G真实API结果。\n'
        '4. 全部19组各运行1359题；失败保留原因并断点重试，不填占位答案。\n'
        '5. 匿名逐题评分；补齐引用支持、完整性、自检前后和迭代前后评审。\n'
        '6. 按文章聚类bootstrap 2000次，seed42，进行多重比较校正。\n'
        '7. 核验全部ID、调用来源、配置与统计一致，生成中文Q4/Q5报告。\n'
        '8. 人工审核模板保留pending，不宣称模型辅助判断等于独立人工裁决。\n\n'
        '成本单价未知；估计11万至14.5万次请求，约4至6天，实际取决于缓存、自检及追问触发率。\n',encoding='utf-8')
    return {'dataset':validation,'cost_estimate':estimate,'configurations':len(configs),'independent_human_review':'pending'}
