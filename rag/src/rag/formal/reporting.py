"""Concise Chinese reports and labeled figures from actual formal artifacts only."""
from datetime import datetime
import json
from zoneinfo import ZoneInfo
import numpy as np
from .contract import matrix,write_json


def fmt(v): return 'N/A' if v is None else f'{v:.4f}'


def metric(summaries,code,key): return summaries[code]['primary_and_RAG_metrics'][key]['mean']


def metric_cell(summaries,code,key):
    cell=summaries[code]['primary_and_RAG_metrics'][key]
    text=fmt(cell['mean'])
    return text+f" (n={cell.get('n_scored',0)})" if key in ['faithfulness','answer_relevance','answer_correctness'] else text


def figures(out,summaries,types):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=out/'figures'; folder.mkdir(exist_ok=True)
    ready=[code for code,s in summaries.items() if s['n_records']==1359 and s['n_success']>0]
    if not ready:return
    def bars(filename,keys,codes,title):
        codes=[c for c in codes if c in ready]
        if not codes:return
        if not any(metric(summaries,c,k) is not None for c in codes for k in keys):return
        fig,ax=plt.subplots(figsize=(max(6,len(codes)*.52),4.2),layout='constrained')
        x=np.arange(len(codes)); width=.8/len(keys)
        for j,key in enumerate(keys):
            values=[metric(summaries,c,key) for c in codes]
            mask=[i for i,v in enumerate(values) if v is not None]
            ax.bar(x[mask]+(j-(len(keys)-1)/2)*width,[values[i] for i in mask],width,label=key)
        ax.set(xticks=x,xticklabels=codes,ylim=(0,1),ylabel='Mean over scored questions',
               title=title+'\nTarget n=1,359 per configuration; see coverage table')
        ax.legend(fontsize=8); fig.savefig(folder/filename,dpi=220); fig.savefig((folder/filename).with_suffix('.svg')); plt.close(fig)
    bars('generation_metrics_comparison.png',['EM','F1','ROUGE_L'],['0','A','G'],'Formal QA metrics (canonical human reference)')
    bars('rag_metrics_comparison.png',['faithfulness','answer_relevance','answer_correctness'],['A','G','K'],'Custom RAG judge metrics')
    if len(ready)==19:
        bars('ablation_f1.png',['F1'],ready,'Full-dataset ablation: token F1')
        bars('ablation_faithfulness.png',['faithfulness'],[c for c in ready if c!='0'],'Full-dataset ablation: faithfulness')
    g=[r for r in types if r['configuration']=='G' and r.get('F1') is not None]
    if g:
        fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
        ax.bar([r['question_type'] for r in g],[r['F1'] for r in g])
        ax.set(ylim=(0,1),ylabel='Token F1',title='Final system G by question type (target n=1,359)')
        ax.tick_params(axis='x',labelrotation=15)
        fig.savefig(folder/'question_type_performance.png',dpi=220); plt.close(fig)
    profiles=out/'performance/fresh_profiles.json'
    if profiles.exists():
        conditions=[r for r in json.loads(profiles.read_text(encoding='utf-8'))['conditions'] if r['workers']==4]
        if conditions:
            fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
            ax.bar([r['configuration'] for r in conditions],[r['latency_p50'] for r in conditions])
            ax.set(ylabel='Seconds',title='Fresh end-to-end latency: median, n=10/config, 4 workers')
            fig.savefig(folder/'latency_comparison.png',dpi=220); plt.close(fig)


def reports(out,summaries,comparisons,self_summary,iteration_summary,cost,agreement):
    folder=out/'reports'; folder.mkdir(exist_ok=True)
    now=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M')
    model=json.loads((out/'config/model_config.json').read_text(encoding='utf-8'))
    total=sum(s['n_success'] for s in summaries.values())
    full=all(s['n_records']==1359 for s in summaries.values())
    common=[f'整理时间：{now}（北京时间）。生成模型：{model["requested_model"]}，温度0。',
       '**状态：正式实验尚未完成，以下保留真实进度与缺失项。**' if total<25821 else
       '全量系统回答已保存；是否完成评分、人工审核及性能验证，以completion_audit.json为准。',
       f'原始主评测集1,359题、331篇文章，仅两组人工答案作参考；19组正式成功回答共{total}/25,821条。'
       '1,070题仅为历史题型一致子集，其结果从同一全量输出按ID过滤。十题聊天先导单独归档，不参与以下数值。',
       '先以第一组人工答案计算主参考分数，再单列两个人工参考取最大值的辅助结果。'
       '分歧问题不删除，人工原始标注不覆盖。生成请求只含问题和实际检索证据；评审是匿名、独立、无历史会话的调用。']
    q4_ready=all(summaries[c]['n_success']==1359 and summaries[c].get('n_judgments')==1359 for c in ['0','A','G'])
    q4_common=common[:]
    q4_common[1]=('Q4的0/A/G全量回答与评分已完成；独立人工审核仍按真实状态单列。' if q4_ready else common[1])
    q4=['# Question 4：下游生成与评测']+q4_common+[
       '## 4.1 Generation Approach',
       '最终系统固定为G：BM25与多语言E5各取100个候选，经RRF（k=60）融合，交叉编码器重排前20项，'
       '取前5段生成简洁引用回答，并执行事实自检。G与Q5使用同一批预测。K只是全模块消融对照，不替换G。'
       '这一设计用外部证据支持可追溯问答，与[检索增强生成](https://arxiv.org/abs/2005.11401)和'
       '[稠密段落检索](https://aclanthology.org/2020.emnlp-main.550/)的研究动机相关；未复现端到端训练。',
       '百科正文进行HTML实体还原、空白清理、章节划分和去重，按句子切块，目标180词、最大300词、重叠一句。'
       '已有4,608篇文档、115,325个片段。保留出处，参考答案不写入索引；短段及公式清理可能丢失信息。',
       '## 4.2 Evaluation Dataset and Annotation',
       '两组人工题型一致率83.89%，Cohen κ约0.7947；这不等于答案语义一致率。'
       f"独立人工答案审核状态：**{agreement['status']}**，已审核{agreement['independent_human_review_n']}/1,359题。"
       '模型辅助建议与人工审核分开保存，未完成时不宣称答案语义一致率达到80%。'
       '原始题集也曾用于检索配置选择，不能称为未接触过的独立测试集。',
       '## 4.3 Evaluation Metrics and Results',
       '| 配置 | 成功/目标 | EM | F1 | ROUGE-L | Faithfulness | Relevance | Correctness |',
       '|---|---:|---:|---:|---:|---:|---:|---:|']
    for code in ['0','A','G']:
        q4.append(f"| {code} | {summaries[code]['n_success']}/1359 | "+' | '.join(metric_cell(summaries,code,k) for k in
             ['EM','F1','ROUGE_L','faithfulness','answer_relevance','answer_correctness'])+' |')
    q4+=['主表为主参考逐题均值。多参考结果、分题型均值、等权题型宏平均和每项有效评分数见evaluation/。'
       '语义Context Recall、词项覆盖代理及反向问题相关性代理各自标识，不冒充官方RAGAS。'
       '引用编号合法性由程序检查，引用是否支持事实由实际引用段落单独判定；未知评分保留null。',
       '评审使用同一供应商模型，可能自偏好；独立人工核验文件包含不少于100个真实评审样本，未填写部分保持pending。'
       '忠实性高不表示完整或正确，需要结合拒答、正确性、完整性和覆盖率。',
       '## 4.4 Efficiency and Scalability',
       '批量生成中复用了经哈希和实际抽查验证的检索结果，相关批量耗时不等同于在线端到端时延。'
       '另以五类各两题、共10题对主要配置测量全新检索与API生成，并在G上比较并发1/2/4。'
       '延迟均值、P50/P90/P95/P99、吞吐、分阶段计时见performance/，每项都标注有效样本数。',
       f"本次正式实验记录（含已执行的Q5部分）累计输入Token：{cost['input_tokens_reported']:,}，输出Token：{cost['output_tokens_reported']:,}；"
       f"货币成本状态：{cost['status']}。未知价格时保留Token，不编造金额。历史复用与新增调用分开计费统计。",
       '精确向量内积随语料增加提升计算和内存，重排前20项限制成本。可考虑近似近邻、批处理和分片，'
       f"但未实测的扩展能力不作容量承诺。正式客户端配置限速为{model.get('options',{}).get('rpm','未记录')} RPM，"
       '这是本地运行参数，不代表供应商公开配额；实际并发还受网络和服务端吞吐约束。']
    q5=['# Question 5：下游增强与全量消融']+common+[
       '## 5.1 Downstream Enhancement Methods',
       '比较重排序、查询改写、PRF、HyDE、多样性、自检和最多两轮的迭代检索。'
       '自检通过提示核对断言并修订；迭代先判断证据充分性，再按缺失信息追加检索，不访问人工答案或gold段落。'
       '这些是现有预训练组件与提示式增强，不声称训练了Self-RAG反思token。',
       '## 5.2 Ablation Study Design',
       '固定同一1,359题、模型及生成参数，配置为0、A至R共19组。A/B只差重排，A/C只差改写，D/E只差自检，'
       'D/F只差迭代，K/L–R每次只移除一个模块。完整矩阵和单模块校验见config/ablation_matrix.json。'
       '所有失败均记录，不用先导答案补齐。',
       '主指标为主参考F1，按来源文章聚类配对bootstrap 2,000次，seed=42。'
       '对每个指标的全部计划对照分别进行Holm校正；未联合控制不同指标之间的家族错误。'
       '置信区间为未校正95%区间，显著性使用同指标家族校正后的p值；不只看均值差。',
       '## 5.3 Experimental Results and Analysis',
       '| 配置 | 成功 | 失败 | 未执行 | F1 | ROUGE-L | Faithfulness | Correctness |',
       '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for code,s in summaries.items():
        q5.append(f"| {code} | {s['n_success']} | {s['n_failed']} | {s['n_not_run']} | "+
            ' | '.join(metric_cell(summaries,code,k) for k in ['F1','ROUGE_L','faithfulness','answer_correctness'])+' |')
    q5+=['未测指标记N/A，不记0。均值只覆盖有真实有效输出或评分的题，覆盖率与有效n在原表中另列。',
       '| 对照 | 配对n | F1差值 | 95% CI | Holm p |',
       '|---|---:|---:|---|---:|']
    for row in comparisons:
        if row['metric']=='F1': q5.append(f"| {row['control']}→{row['treatment']} | {row['n']} | {row['delta']:+.4f} | {row['ci95']} | {row['p_holm']:.4f} |")
    significant=[r for r in comparisons if r['metric']=='F1' and r['significant_holm_005']]
    if significant:
        q5.append('具有主指标统计支持的对照：'+ '；'.join(f"{r['control']}→{r['treatment']}（{r['delta']:+.4f}）" for r in significant)+'。正负方向均完整保留。')
    else:q5.append('当前未获得经过上述多重比较校正后的F1改善证据；不能将均值上升直接写成显著有效。')
    if all(s['n_success']==1359 for s in summaries.values()):
        ranked=sorted(summaries,key=lambda c:(-metric(summaries,c,'F1'),c))
        best=ranked[0]
        q5.append(f'按预先指定的主参考Token F1，19组中观察到的最高均值配置为{best}（{metric(summaries,best,"F1"):.4f}）。'
                  '这是同一评测集上的描述性排序，不是独立验证集调参或显著优于其余所有配置的证明；并列值应结合原始表查看。'
                  'Q4预先固定的最终系统仍为G，选择部署配置还需结合忠实性、完整性及实测时延。')
    q5.append('困难问题的描述性分组见ablation/difficult_question_analysis.json及difficult_question_metrics.csv：'
              '比较题、至少两个标注证据块、至少30词的长问题，以及两个人工答案词面F1低于0.5的问题。'
              '规则固定且不依赖模型得分；组间可重叠，均不从1,359题主评测中删除。'
              '多个证据块不等于多跳推理，词面差异不等于语义分歧；这些分组用于探索机制，不单独作显著性结论。')
    q5+=['重排与改写的组合还计算D−B−C+A的逐题交互差值，结果见bootstrap_confidence_intervals.json。'
       '该分析为探索性，不能凭平均分就断言协同增益。',
       '## 5.4 Case Study',
       '| 自检配置 | 执行n | 检出不支持主张 | 请求修订 | 答案实际改变 | 修好为全对 | 正确答案被误改 |',
       '|---|---:|---:|---:|---:|---:|---:|']
    for code,r in self_summary.items():
        q5.append(f"| {code} | {r['n']} | {r['unsupported_detected']} | {r['revision_requested']} | {r['answer_changed']} | {r['repaired_to_correct']} | {r['correct_answer_corrupted']} |")
    q5+=['修订前后均评价相同真实草稿与最终答案；未知判断不算成功修复。'
       '若未触发修订，真实记录为0，不构造改善案例。',
       '| 迭代配置 | n | 首轮不足 | 追加检索题数 | 新片段总数 | 补齐缺失证据 |',
       '|---|---:|---:|---:|---:|---:|']
    for code,r in iteration_summary.items():
        q5.append(f"| {code} | {r['n']} | {r['first_insufficient_n']} | {r['followup_triggered_n']} | {r['new_unique_chunks']} | {r['new_evidence_resolved_gap_n']} |")
    q5+=['迭代前后分别保留真实回答、上下文和评分；自检设置保持一致，额外的前状态回答和评审计为离线研究开销。'
       '成功补齐与未补齐的真实案例见iterative/，自检正负案例见self_check/。'
       '若模块无增益或退化，可能涉及不必要改写、假设文档噪声、分散证据或错误修订，但这些解释属于待检验机制，不能替代实验数据。']
    def save(name,lines):
        blocks=[]
        for line in lines:
            if blocks and not(line.startswith('|') and blocks[-1].startswith('|')):blocks.append('')
            blocks.append(line)
        (folder/name).write_text('\n'.join(blocks)+'\n',encoding='utf-8')
    save('Q4_FORMAL_REPORT.md',q4); save('Q5_FORMAL_REPORT.md',q5)
    save('Q4_Q5_COMBINED_REPORT.md',q4+['\n---\n']+q5)
    (folder/'REPRODUCIBILITY.md').write_text(
        '# 复现\n\n在项目根目录执行：\n\n'
        '`powershell -ExecutionPolicy Bypass -File rag/run_local.ps1 scripts/16_formal_full.py preflight`\n\n'
        '`powershell -ExecutionPolicy Bypass -File rag/run_local.ps1 scripts/16_formal_full.py run`\n\n'
        '`powershell -ExecutionPolicy Bypass -File rag/run_local.ps1 scripts/16_formal_full.py analyze`\n\n'
        '`powershell -ExecutionPolicy Bypass -File rag/run_local.ps1 scripts/16_formal_full.py validate`\n\n'
        '模型密钥从本地.env读取，不包含在输出中。冻结数据、参数和代码发生变化时拒绝混跑。'
        '源码快照见config/code_snapshot；1359题及1070子集ID见data_validation/dataset_manifest.json。'
        '人工审核完成前，annotation_agreement_report和judge_validation_status保持pending。\n',encoding='utf-8')
    (out/'README.md').write_text('# Q4/Q5正式实验\n\n本目录仅包含真实API全量实验及其验证、派生统计。'
       f'当前成功回答{total}/25,821；记录范围完整：{full}。实时进度见generation/execution_status.json，'
       '最终证明以reports/completion_audit.json为准。独立人工审核及未知价格不得写成完成。\n',encoding='utf-8')
    return folder
