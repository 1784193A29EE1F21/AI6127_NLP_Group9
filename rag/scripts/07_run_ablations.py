"""Run complete retrieval ablations and record the generation experiment matrix."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tqdm import tqdm
from rag import config
from rag.config import PATHS
from rag.corpus import load_jsonl, save_jsonl
from rag.metrics import retrieval_metrics, hit_at_k
from rag.pipeline import RAGSystem
from rag.llm import LLM
from rag.ablations import retrieval_configs, generation_configs, RETRIEVAL_CONTRASTS, GENERATION_CONTRASTS, paired_delta


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def filename(name):
    return name.replace(' ','_').replace('+','plus').replace('(','').replace(')','')+'.trec'


def read_run(path, ids):
    ranks = defaultdict(list)
    for line in path.read_text(encoding='utf-8').splitlines():
        qid, _, chunk, rank, score, tag = line.split()
        assert qid in ids
        ranks[qid].append((int(rank),chunk))
    result = {}
    for qid in ids:
        ranked = sorted(ranks[qid])
        assert len({cid for _,cid in ranked}) == len(ranked)
        assert [rank for rank,_ in ranked] == list(range(1,len(ranked)+1))
        result[qid] = [cid for _,cid in ranked]
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=PATHS.results_dir/'ablation_full_2026-11-02')
    ap.add_argument('--reuse',type=Path,default=PATHS.results_dir/'retrieval_full_2026-11-02')
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'runs').mkdir(exist_ok=True)
    questions = load_jsonl(PATHS.eval_file)
    assert len(questions)==1359 and len({q['id'] for q in questions})==len(questions)
    assert all(q['gold_chunk_ids'] for q in questions)
    ids = [q['id'] for q in questions]
    corpus_hash, eval_hash = sha(PATHS.chunks_file),sha(PATHS.eval_file)
    generation_status={'status':'ready','provider':config.LLM_PROVIDER,'model':config.LLM_MODEL}
    try:
        LLM()
    except Exception as error:
        generation_status.update(status='blocked',reason=str(error))
    generation_status.update(configurations=[{'name':name,'retrieval':asdict(r) if r else None,
                                               'generation':asdict(g) if g else None}
        for name,r,g in generation_configs(True)],contrasts=GENERATION_CONTRASTS,
        n_questions=len(ids),sample_ids=ids,
        command='powershell -ExecutionPolicy Bypass -File rag/run_local.ps1 scripts/06_eval_generation.py --all --n 1359 --workers 1 --out results/ablation_full_2026-11-02/generation')
    (args.out/'generation_status.json').write_text(json.dumps(generation_status,ensure_ascii=False,indent=2),encoding='utf-8')

    reuse_manifest=json.loads((args.reuse/'run_manifest.json').read_text(encoding='utf-8'))
    assert reuse_manifest['dataset_sha256']==eval_hash and reuse_manifest['corpus_sha256']==corpus_hash
    old_code_hashes={name.replace('\\','/'):value for name,value in reuse_manifest['code_sha256'].items()}
    for code in ['src/rag/pipeline.py','src/rag/bm25.py','src/rag/dense.py','src/rag/fusion_rerank.py','src/rag/query.py','src/rag/text.py','src/rag/metrics.py','src/rag/config.py']:
        assert old_code_hashes[code]==sha(PATHS.root/code), f'Stale reusable code: {code}'
    dataset_manifest=json.loads((PATHS.eval_dir/'dataset_manifest.json').read_text(encoding='utf-8'))
    for path, expected in dataset_manifest['files_sha256'].items():
        if path.replace('\\','/').startswith('index/'):
            assert sha(PATHS.root/'data'/path)==expected, f'Stale index: {path}'
    with (args.reuse/'summary.csv').open(encoding='utf-8',newline='') as f:
        old_summaries={r['config']:r for r in csv.DictReader(f)}
    old_questions=json.loads((args.reuse/'per_query.json').read_text(encoding='utf-8'))
    system=RAGSystem()
    system.warmup()
    doc_of_chunk={c.chunk_id:c.doc_id for c in system.chunks}
    configs=retrieval_configs()
    summaries=[]
    per_query={}
    with (args.out/'qrels.trec').open('w',encoding='utf-8') as f:
        for q in questions:
            for cid in q['gold_chunk_ids']:
                f.write(f"{q['id']} 0 {cid} 1\n")
    for name,cfg in configs.items():
        runfile=args.out/'runs'/filename(name)
        if name in old_summaries:
            assert int(old_summaries[name]['n'])==len(ids)
            assert set(old_questions[name])==set(ids)
            ranked=read_run(args.reuse/'runs'/filename(name),set(ids))
            runfile.write_bytes((args.reuse/'runs'/filename(name)).read_bytes())
            origin='reused_verified_run'
            latency=float(old_summaries[name]['latency_ms_mean'])
        else:
            origin='fresh_run'
            ranked={}
            times=[]
            with runfile.open('w',encoding='utf-8') as f:
                for q in tqdm(questions,desc=name):
                    start=time.perf_counter()
                    hits,info=system.retrieve(q['question'],cfg)
                    times.append((time.perf_counter()-start)*1000)
                    ranked[q['id']]=[h['chunk_id'] for h in hits]
                    for h in hits:
                        score=h.get('rerank',h.get('rrf',0))
                        f.write(f"{q['id']} Q0 {h['chunk_id']} {h['rank']} {score} ablation\n")
            latency=float(np.mean(times))
        metrics=[]
        records=[]
        for q in questions:
            ranking=ranked[q['id']]
            assert len(ranking)==len(set(ranking)) and all(cid in doc_of_chunk for cid in ranking)
            m=retrieval_metrics(ranking,set(q['gold_chunk_ids']))
            m['DocHit@5']=hit_at_k([doc_of_chunk[cid] for cid in ranking],{q['gold_doc_id']},5)
            m['unique_docs@5']=len({doc_of_chunk[cid] for cid in ranking[:5]})
            m['returned@10']=len(ranking)
            metrics.append(m)
            records.append({'id':q['id'],'question_type':q['question_type'],'gold_doc_id':q['gold_doc_id'],'ranked':ranking,'metrics':m})
        summary={'config':name,'n':len(ids),'origin':origin,'latency_ms_mean':latency,
                 **{key:float(np.mean([m[key] for m in metrics])) for key in metrics[0]}}
        summaries.append(summary)
        per_query[name]=records
        print(json.dumps(summary,ensure_ascii=False))
        save_jsonl(records,args.out/(filename(name)+'.jsonl'))

    paired=[]
    docs=[q['gold_doc_id'] for q in questions]
    for label,control,treatment in RETRIEVAL_CONTRASTS:
        for metric in ['MRR@10','nDCG@10','Hit@5','R@5','DocHit@5']:
            delta=paired_delta([r['metrics'][metric] for r in per_query[control]],
                               [r['metrics'][metric] for r in per_query[treatment]],docs,seed=args.seed)
            paired.append({'feature':label,'control':control,'treatment':treatment,'metric':metric,**delta})
    # 2x2 interaction: does PRF change the gain attributable to reranking?
    for metric in ['MRR@10','nDCG@10','Hit@5','R@5']:
        vector=[per_query['Hybrid RRF + PRF + rerank'][i]['metrics'][metric]
                -per_query['Hybrid RRF + PRF'][i]['metrics'][metric]
                -per_query['Hybrid RRF + rerank'][i]['metrics'][metric]
                +per_query['Hybrid RRF'][i]['metrics'][metric] for i in range(len(ids))]
        paired.append({'feature':'PRF_x_rerank_interaction','control':'additive effects','treatment':'combined effect',
                       'metric':metric,**paired_delta([0]*len(ids),vector,docs,seed=args.seed)})
    by_type={name:{typ:{'n':sum(r['question_type']==typ for r in rs),
                    'MRR@10':float(np.mean([r['metrics']['MRR@10'] for r in rs if r['question_type']==typ]))}
                    for typ in sorted({q['question_type'] for q in questions})} for name,rs in per_query.items()}
    report={'n_questions':len(ids),'n_source_articles':len(set(docs)),'dataset_sha256':eval_hash,
            'corpus_sha256':corpus_hash,'retrieval_configs':{n:asdict(c) for n,c in configs.items()},
            'summaries':summaries,'paired_comparisons':paired,'by_question_type':by_type,
            'generation':generation_status,'bootstrap':'paired source-article cluster bootstrap, 2000 draws, seed 42; exploratory, no multiplicity correction',
            'code_sha256':{str(p.relative_to(PATHS.root)):sha(p) for p in [Path(__file__),PATHS.root/'src/rag/ablations.py']}}
    (args.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 全创新点消融实验（2026-11-02）','',
        '固定完整 1,359 题、相同 gold 和 115,325 个语料片段。检索 top_k=10、候选数保持一致；文档多样性上限为每篇文章 2 个片段。', '',
        '七组已有结果在核对题集、语料、索引和检索代码哈希后复用，三组多样性配置新跑。时延来自不同轮次，不能把时延差解释为严格配对的性能开销。','',
        '| 配置 | n | MRR@10 | nDCG@10 | Hit@5 | R@5 | 平均文档数@5 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f"| {r['config']} | {r['n']} | {r['MRR@10']:.4f} | {r['nDCG@10']:.4f} | {r['Hit@5']:.4f} | {r['R@5']:.4f} | {r['unique_docs@5']:.2f} |")
    lines+=['','## 配对消融差值','',
            '差值 = treatment − control。区间按来源文章聚类重采样 2,000 次，保留同题配对；区间为探索性结果，未做多重比较校正。','',
            '| 因素/对照 | ΔMRR@10 | 95% 区间 | 胜/平/负题数 |',
            '|---|---:|---|---|']
    for r in paired:
        if r['metric']=='MRR@10':
            lo,hi=r['ci95']
            lines.append(f"| {r['feature']} | {r['delta']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {r['wins']}/{r['ties']}/{r['losses']} |")
    lines+=['','详细 control/treatment 名称、其余指标及分题型结果见 report.json。', '',
            '## 生成侧实验','',
            f"状态：{generation_status['status']}。{generation_status.get('reason','仅客户端初始化成功，尚未执行真实请求。')}",
            '19 组配置及 17 项对照已保存于 generation_status.json，包括闭卷、基线、逐项添加、完整系统以及七项完整系统移除实验。',
            '固定相同题目、主人工参考、简洁回答、temperature=0、workers=1；生成检索 top_k=5，迭代检索最多两轮。',
            '需要模型接口的改写、HyDE、自检、迭代检索没有有效运行结果，不能把配置清单或代码测试当成实验结果。', '',
            '## 解释边界','',
            '- 多样性上限增加来源覆盖，但本数据 gold 基于单篇来源文章，可能牺牲相关片段密度；不能仅凭该指标判断跨文档任务的价值。',
            '- 当前题集为英文题；多语言编码器固定，中文查询只有之前的演示测试，不能据此宣称完成多语言消融。',
            '- 标题过滤是用户输入功能；使用 gold 标题作为过滤条件会泄漏答案来源，所以不作为默认创新消融。',
            '- 分块与清洗是固定的语料构建条件，本次未比较其他分块策略。',
            '- gold 相关性为词覆盖匹配，仍需人工核验；三组标注存在分歧，当前主参考未做人工裁决。',
            '- 自定义 judge 不是官方 RAGAS；没有接口时不报告生成质量、faithfulness 或 API 成本。']
    (args.out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Saved',args.out/'report.md')


if __name__=='__main__':
    main()
