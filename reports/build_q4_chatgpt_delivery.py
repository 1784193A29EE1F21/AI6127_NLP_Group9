"""Deliver Q4 writing materials as soon as its real measured results are available."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
RAG=ROOT/'rag'
LEGACY=RAG/'results/q4_two_humans_2026-11-02'
FORMAL=RAG/'results/q4_q5_formal_full'
DEST=ROOT/'reports/Q4_ChatGPT写作材料'


def read(path,default=None):return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def available(source):
    if source=='formal':
        if not read(FORMAL/'reports/Q4_completion_audit.json',{}).get('automated_complete'):return False
        summary=read(FORMAL/'evaluation/metrics_summary.json',{}).get('full_1359',{})
        if not all(summary.get(c,{}).get('n_success')==1359 and summary[c].get('n_judgments')==1359 for c in ['0','A','G']):return False
        profile=read(FORMAL/'performance/fresh_profiles.json',{}).get('conditions',[])
        return {(c,4) for c in ['0','A','G']}|{('G',1),('G',2)} <= {(r['configuration'],r['workers']) for r in profile if r.get('n')==10}
    status=read(LEGACY/'status.json',{})
    return (all(status.get('counts',{}).get(p)==1359 for p in ['generation','closed_book','judge','agreement'])
            and (LEGACY/'fresh_end_to_end_benchmark.json').exists()
            and read(LEGACY/'summary.json',{}).get('RAG_metrics') is not None)


def build(source='formal',formal_root=None):
    global FORMAL
    if formal_root is not None:FORMAL=Path(formal_root)
    if not available(source):
        raise RuntimeError('Q4 source is not fully measured yet; no final writing bundle created')
    now=datetime.now(ZoneInfo('Asia/Shanghai')); stamp=now.strftime('%Y%m%d_%H%M%S')
    label='正式0_A_G版' if source=='formal' else '基础0_G版'
    folder=DEST/(label+'_'+stamp); folder.mkdir(parents=True,exist_ok=False)
    origin=FORMAL if source=='formal' else LEGACY
    summary=read(origin/('evaluation/metrics_summary.json' if source=='formal' else 'summary.json'))
    corpus=read(RAG/'data/corpus/stats.json')
    dataset=read(LEGACY/'data_manifest.json')
    metrics=[]
    if source=='formal':
        for code in ['0','A','G']:
            row=summary['full_1359'][code]
            metrics.append({'configuration':code,'n':row['n_success'],'n_judged':row['n_judgments'],
                'primary':row['primary_and_RAG_metrics'],'multi_reference':row['multi_reference']})
        annotation=read(FORMAL/'data_validation/annotation_agreement_report.json')
        performance=read(FORMAL/'performance/fresh_profiles.json')
        draft=FORMAL/'reports/Q4_FORMAL_REPORT.md'
        figures=[FORMAL/'figures'/name for name in ['generation_metrics_comparison.png','rag_metrics_comparison.png',
                                                  'latency_comparison.png','question_type_performance.png']]
        extra_files=[FORMAL/'evaluation/metrics_by_question_type.csv',FORMAL/'performance/latency_summary.csv',
                     FORMAL/'performance/token_usage.csv',FORMAL/'performance/cost_report.json']
        caveat='0/A/G三组均为1,359题；Q5其余组可能仍在运行，不影响本材料中的Q4全量数据。'
        summary={**summary,'full_1359':{c:summary['full_1359'][c] for c in ['0','A','G']},
                 'secondary_1070':{c:summary.get('secondary_1070',{}).get(c) for c in ['0','A','G']}}
    else:
        for code,key in [('0','closed_book_quality'),('G','generation_quality')]:
            quality=summary[key]
            metrics.append({'configuration':code,'n':1359,'primary':quality['primary_reference'],
                            'multi_reference':quality['two_human_references']})
        annotation={'question_type_agreement':dataset['question_type_agreement'],
            'model_assisted_answer_equivalence':summary.get('human_answer_agreement'),
            'independent_human_answer_equivalence':'pending; model comparison is not an independent human review'}
        performance=read(LEGACY/'fresh_end_to_end_benchmark.json')
        draft=ROOT/'reports/Q4_两组人工标注_中文报告.md'
        figures=[LEGACY/'RAG_vs_Closed_book.png']
        extra_files=[LEGACY/'quality_by_type_partial_or_complete.csv',LEGACY/'retrieval_summary.json']
        caveat='此为当前Q4基础流程的0/G全量材料。task1.md额外要求的A组和更完整的引用评分，将在正式0/A/G版中补齐；不要把A写成已经完成。'
    model_name=(read(FORMAL/'config/model_config.json',{}).get('requested_model','unknown') if source=='formal' else summary.get('model','unknown'))
    data={'source_mode':source,'created_at_beijing':now.isoformat(),'version_label':label,
        'dataset':{'n':1359,'source_articles':331,'human_annotators':2,'question_types':dataset['question_type_counts']},
        'corpus':corpus,'model':model_name,'temperature':0.,'reference_protocol':
        'Human original canonical; max over two human references auxiliary; AI answers never used as gold',
        'configuration_metrics':metrics,'full_source_summary':summary,'annotation':annotation,'performance':performance,
        'independent_human_review':'pending unless the supplied review status explicitly proves completion',
        'monetary_cost':'unknown unless a verified tariff is recorded','scope_note':caveat,
        'source_path':str(origin.relative_to(ROOT)),'all_assignment_requirements_claimed_complete':False}
    data['measurement_protocol_version']='formal_Q4Q5_v2' if source=='formal' else 'initial_Q4_0G_v1'
    if source=='formal':
        data['runtime_scheduling_amendment']=read(FORMAL/'config/runtime_amendment.json')
        data['effective_runtime']=read(FORMAL/'config/runtime_active.json')
        if data['runtime_scheduling_amendment']:
            caveat+=' 运行中按用户要求将调度从30 RPM/4并发调整为60 RPM/6并发；模型、采样参数、提示词和评分规则未变，原冻结配置与变更历史均保留。'
            data['scope_note']=caveat
            if data['runtime_scheduling_amendment'].get('endpoint_override'):
                caveat+=' 后续请求切换至DeepSeek官方接口，已完成记录保留；同名模型不证明代理与官方底层版本完全相同，评价来源变更是比较局限，实际接口见逐次调用记录。'
                data['scope_note']=caveat
            extra_files.append(FORMAL/'config/runtime_amendment.json')
    data['metric_definitions']={
        'EM':'Project SQuAD-style normalization, then exact equality',
        'Token_F1':'Normalized word-multiset overlap; precision and recall combined by harmonic mean',
        'ROUGE_L':'Normalized token longest-common-subsequence F1',
        'answer_relevance':('Anonymous direct 0/0.5/1 rubric; reverse-question E5 similarity stored separately'
                            if source=='formal' else 'Reverse-question E5 similarity proxy, not a direct relevance judgment'),
        'context_recall_lexical':'Human-original evidence word-set coverage; not semantic recall',
        'context_recall_semantic':'Judge assessment of required human reference facts supported by retrieved context',
        'faithfulness':('Fraction of atomic claims supported; no factual claims/uncertainty may yield null'
                        if source=='formal' else 'Supported atomic claims; an empty-claim refusal is assigned 1 in this initial protocol'),
        'correctness_reference':('Canonical human A; separate multi-reference correctness also available'
                                 if source=='formal' else 'Either supplied human reference accepted where consistent with retrieved evidence'),
        'comparability_warning':'Do not mix metrics from the initial and formal protocol versions or infer effects from their differences.'}
    (folder/'03_实验数据.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    prompt='''请作为NTU AI6127 NLP小组作业的报告写作助手，只依据附件的真实实验数据，撰写Question 4中文报告。

结构：4.1 Generation Approach；4.2 Evaluation Dataset and Annotation；4.3 Evaluation Metrics and Results；4.4 Efficiency and Scalability。

写清做了什么、为什么这样做、如何评价、结果及局限。使用简洁学术语言，表格承载主要数值，不写成长篇工程说明。

必须遵守：
1. 主评测集为全部1,359题、331篇来源文章，仅两组人工答案作为参考。不要使用1,070题筛选子集替代主集。
2. 模型以03_实验数据.json中的model字段为准，不写成当前聊天模型。没有微调或端到端训练。
3. 主参考和双人工参考取最大值的辅助结果分开报告。引用中位数、P95等时注明样本规模和是否含新检索。
4. 题型83.89%不等于答案语义一致性。模型比较两个人工答案不是第三个人工标注者，也不能代替独立人工审核。
5. 未核实的历史97.94%汇总数字不作为答案一致性达标证明。人工审核pending时明确写pending。
6. 各指标使用数据中的真实定义；反向问题相似度是代理指标，自定义评分不是官方RAGAS执行结果。
7. 不将高Faithfulness等同于回答正确或完整。未知评分不填0。注意每项指标的n_scored及拒答率。
8. 未知API价格不编造金额；只报告实际Token和成本公式。纯检索耗时不冒充端到端生成时延。
9. 不使用旧十题聊天先导数据，不凭均值差宣称显著提升，不把Q5尚未完成的配置写成已完成。
10. 基础0/G版本缺少的A组与引用支持评分，不自行补数字；正式0/A/G版本优先于基础版。

先列出可用实验证据和仍缺的证据，再给出可放入课程报告的正文与表格。'''
    (folder/'01_直接粘贴给ChatGPT的提示词.md').write_text(prompt+'\n',encoding='utf-8')
    text=draft.read_text(encoding='utf-8')
    (folder/'02_Q4中文写作底稿.md').write_text('> 材料版本：'+label+'。'+caveat+'\n\n'+text,encoding='utf-8')
    rows=[]
    for item in metrics:
        for key,value in item['primary'].items():
            mean=value.get('mean') if isinstance(value,dict) else value
            n=value.get('n_scored',item['n']) if isinstance(value,dict) else item['n']
            rows.append({'configuration':item['configuration'],'protocol':'canonical_or_explicit_RAG_definition',
                         'metric':key,'value':mean,'n_scored':n})
        for key,value in item['multi_reference'].items():
            rows.append({'configuration':item['configuration'],'protocol':'max_over_two_human_references',
                         'metric':key,'value':value,'n_scored':item['n']})
    with (folder/'04_核心结果表.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if source=='legacy':
        (folder/'05_RAG指标_基础版.json').write_text(json.dumps(summary['RAG_metrics'],ensure_ascii=False,indent=2),encoding='utf-8')
    for path in figures+extra_files:
        if path.exists():
            target=folder/('figures' if path.suffix=='.png' else '原始统计')/path.name
            target.parent.mkdir(exist_ok=True)
            if source=='formal' and path.suffix=='.csv':
                with path.open(encoding='utf-8-sig',newline='') as f:
                    reader=csv.DictReader(f); fields=reader.fieldnames
                    selected=[r for r in reader if r.get('configuration') in ['0','A','G']]
                with target.open('w',encoding='utf-8-sig',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(selected)
            else:shutil.copy2(path,target)
    limits='''# 写作边界

材料仅复用本次真实API输出和真实统计。原始人工文件未修改，第三组AI答案不作为gold。
独立人工答案等价审核与100条judge人工核验尚未完成时，不能宣称所有作业门槛已验证。
生成器和judge来自同一接口模型，存在自评偏差。题集曾用于检索配置选择，不是未接触过的held-out test。
真实来源、参数及版本见03_实验数据.json和来源目录的输入manifest；本包不包含API密钥。
'''
    (folder/'06_限制与引用说明.md').write_text(limits+'\n'+caveat+'\n',encoding='utf-8')
    files={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file()}
    (folder/'SHA256.json').write_text(json.dumps(files,ensure_ascii=False,indent=2),encoding='utf-8')
    archive=DEST/(folder.name+'.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in folder.rglob('*'):
            if path.is_file():z.write(path,path.relative_to(folder))
    pointer={'source':source,'version':label,'created_at_beijing':now.isoformat(),'materials_directory':str(folder.relative_to(ROOT)),
             'zip':str(archive.relative_to(ROOT)),'automated_Q4_0_A_G_ready':source=='formal',
             'independent_human_review':'pending unless explicitly completed','scope_note':caveat}
    (DEST/'最新材料位置.json').write_text(json.dumps(pointer,ensure_ascii=False,indent=2),encoding='utf-8')
    delivery=FORMAL/'deliveries';delivery.mkdir(exist_ok=True)
    (delivery/('Q4_formal_ready.json' if source=='formal' else 'Q4_initial_ready.json')).write_text(json.dumps(pointer,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Q4 writing materials ready:',archive,flush=True)
    return pointer


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',choices=['legacy','formal'],default='formal')
    args=ap.parse_args();build(args.source)
