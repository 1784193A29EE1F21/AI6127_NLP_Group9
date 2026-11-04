"""Reports for the approved six groups; never label them a completed nineteen-group study."""
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from rag.formal.reporting import figures as original_figures
from rag.formal.contract import write_json
from .scope import CODES,matrix


def fmt(v):return 'N/A' if v is None else f'{v:.4f}'


def figures(out,summaries,types):
    original_figures(out,summaries,types)
    if not all(summaries[c]['n_success']==1359 for c in CODES):return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for key,name in [('F1','ablation_f1'),('faithfulness','ablation_faithfulness')]:
        codes=[c for c in CODES if summaries[c]['primary_and_RAG_metrics'][key]['mean'] is not None]
        if not codes:continue
        fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
        ax.bar(codes,[summaries[c]['primary_and_RAG_metrics'][key]['mean'] for c in codes])
        ax.set(ylim=(0,1),ylabel=key,title='Six-group ablation; target n=1,359 per configuration')
        fig.savefig(out/'figures'/(name+'.png'),dpi=220)
        fig.savefig(out/'figures'/(name+'.svg'));plt.close(fig)


def reports(out,summaries,comparisons,self_summary,iteration_summary,cost,agreement):
    folder=out/'reports';folder.mkdir(exist_ok=True)
    model=json.loads((out/'config/model_config.json').read_text(encoding='utf-8'))
    amendment=json.loads((out/'config/runtime_amendment.json').read_text(encoding='utf-8'))
    total=sum(s['n_success'] for s in summaries.values())
    common=[f"北京时间 {datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M')}；生成与评审请求模型：{model['requested_model']}。",
        f'本次范围为用户确认的六组0/A/B/G/S/T，共8,154条目标回答，目前成功生成{total:,}条。原19组方案已取消，历史记录单独保留。',
        '主集1,359题、331篇文章；1,070题仅为历史题型一致子集，从相同全量输出按ID过滤。原始题集曾用于检索配置选择，不能称为完全未接触的独立测试集。',
        '以第一组人工答案为主参考；双人工参考逐指标取最大值只作辅助。AI答案不作为gold，分歧题不删除。生成阶段不接触参考答案和gold标签。',
        '同名deepseek-flash请求曾经由代理及官方接口执行。逐次调用保留真实接口；代理底层版本未固定，供应商切换可能引入评价来源混杂。请求温度为0，但未指定的思考模式及供应商实际参数支持不能视为受控。']
    def table(codes,keys):
        rows=['| 配置 | 生成成功 | 评分成功 | '+' | '.join(keys)+' |',
              '|---|---:|---:|'+'---:|'*len(keys)]
        for c in codes:
            s=summaries[c]
            cells=[]
            for k in keys:
                v=s['primary_and_RAG_metrics'][k]
                cells.append(fmt(v['mean'])+f" (n={v['n_scored']})")
            rows.append(f"| {c} | {s['n_success']}/1359 | {s['n_judgments']}/1359 | "+' | '.join(cells)+' |')
        return rows
    q4=['# Question 4 中文报告初稿']+common+[
        '## 4.1 Generation Approach',
        '最终系统G固定为Hybrid RRF + Cross-Encoder重排序 + 事实自检。BM25及多语言E5分别取100候选，RRF k=60，重排20项并取前5段生成简洁编号引用回答。'
        'G与Q5完全复用同一批预测。对照为不检索的0组和基础Hybrid RAG的A组。外部证据为可追溯问答提供依据，但不保证回答完整或正确。',
        '百科正文经HTML实体还原、空白清理、分节和去重，按句子切块，目标180词、最大300词、重叠一句。知识库保留4,608篇文档、115,325个片段及出处。'
        '这便于控制上下文长度，但清理及切块仍可能丢失必要信息。',
        '## 4.2 Evaluation Dataset and Annotation',
        f"两组人工题型一致率83.89%、Cohen κ约0.7947，不等于答案语义一致率。答案独立人工审核状态：{agreement['status']}，"
        f"已审核{agreement['independent_human_review_n']}/1359。未完成时不宣称80%答案一致性门槛已经验证。人工原始答案、模型辅助建议和人工裁决分开保存。",
        '## 4.3 Evaluation Metrics and Results']+table(['0','A','G'],['EM','F1','ROUGE_L','faithfulness','answer_relevance','answer_correctness'])+[
        'EM、Token F1和ROUGE-L先去除纯引用标记并按项目规范化；多参考评分单独保存。RAG评审匿名、独立调用，保存原始JSON、断言、理由与不确定判断。',
        'Faithfulness按事实断言支持计算；Context Precision按排名和有用段落判断，语义Context Recall按必要参考事实覆盖判断。'
        '词项覆盖及反向问题向量相关性代理另列，不称为官方RAGAS。无事实断言或未知判断保持null，并报告有效n。']+table(['A','G'],['context_precision','context_recall','citation_validity','citation_support','answer_completeness'])+[
        '生成器与评审来自同一模型，可能自偏好；自动分数不代替人工核验。100条分层judge核验文件及分题型均值、等权题型宏平均见data_validation/和evaluation/。',
        '## 4.4 Efficiency and Scalability',
        '全量生成复用了经校验的检索与API结果，批量耗时不能冒充在线推理时延。另对五类各两题、共10题执行全新检索及生成，G比较并发1/2/4。'
        'performance/保留均值、P50/P90/P95/P99、吞吐、调用数、重试、Token和未知用量；固定执行顺序及小样本限制单列。',
        f"调度现为{model['options']['rpm']} RPM、{model['options']['workers']}并发、{model['options']['timeout']}秒超时；这些是客户端参数，不是供应商容量。"
        '历史运行配置及接口变更见runtime_amendment.json。精确向量检索随知识库规模增加开销，重排候选上限控制成本；未实测的分片或近似检索能力只作为扩展方向。',
        f"当前已记录输入Token {cost['input_tokens_reported']:,}、输出Token {cost['output_tokens_reported']:,}（含本次已执行Q5部分）；费用状态{cost['status']}。"
        '没有经过核实的代理计费单价时不编造实际金额；用户提供的账户支出与预算预测另列，不当作逐组结算。']
    q5=['# Question 5 中文报告初稿']+common+[
        '## 5.1 Downstream Enhancement Methods',
        '本轮Q5聚焦事实自检与最多两轮迭代检索。自检核查断言支持并按需修订，迭代先评估证据充分性，再按缺失事实补充检索。'
        '二者均不能读取人工参考答案或gold证据来决定如何生成。改写、PRF、HyDE及多样性作为既有检索侧研究，不在此六组生成消融中宣称已验证贡献。',
        '## 5.2 Ablation Study Design',
        'B为Hybrid+重排序基线，G=B+自检，S=B+迭代，T=B+自检+迭代。四组除对应开关外完全一致，不开启查询改写、PRF、HyDE或多样性。'
        'S并非原F（F还包含改写），T也并非原K（K含所有模块）。0/A保留Q4对照，A/B额外检验重排序。',
        '六组各评全部1,359题，不用小样本冒充全量。主指标为主参考Token F1；按来源文章聚类配对bootstrap 2,000次、seed42；'
        '每个指标对七个预先指定对照作Holm校正，95%区间本身未作多重校正。交互T−G−S+B为探索性分析。',
        '## 5.3 Experimental Results and Analysis']+table(CODES,['F1','ROUGE_L','faithfulness','answer_correctness'])+[
        '| 对照 | 配对n | F1差值 | 95%区间 | Holm p |','|---|---:|---:|---|---:|']
    for row in comparisons:
        if row['metric']=='F1':q5.append(f"| {row['control']}→{row['treatment']} | {row['n']} | {row['delta']:+.4f} | {row['ci95']} | {row['p_holm']:.4f} |")
    q5.append('均值上升不等于显著提升；负结果、无增益与缺失评分均保留。困难分组按固定规则统计比较题、多证据块题、长问题及参考答案词面差异，不称为人工难度标签。')
    if all(summaries[c]['n_success']==1359 for c in CODES):
        best=max(CODES,key=lambda c:summaries[c]['primary_and_RAG_metrics']['F1']['mean'])
        q5.append(f'六组中观察到的最高主参考F1配置为{best}；这是同一题集的描述性排序，不证明独立泛化或显著优于所有其余配置。Q4预先固定最终系统仍为G。')
    q5+=['## 5.4 Case Study','自检前后、迭代前后均保存真实答案和实际上下文；S/T迭代前反事实回答保持相同自检设置。'
        '自检修复与误改、追加证据补齐与失败分别统计，不虚构成功案例。见self_check/及iterative/的逐题记录、汇总和案例。',
        '自检真实统计：\n\n```json\n'+json.dumps(self_summary,ensure_ascii=False,indent=2)+'\n```',
        '迭代真实统计：\n\n```json\n'+json.dumps(iteration_summary,ensure_ascii=False,indent=2)+'\n```']
    def save(name,lines):
        blocks=[]
        for line in lines:
            if blocks and not(line.startswith('|') and blocks[-1].startswith('|')):blocks.append('')
            blocks.append(line)
        (folder/name).write_text('\n'.join(blocks)+'\n',encoding='utf-8')
    save('Q4_FORMAL_REPORT.md',q4);save('Q5_FORMAL_REPORT.md',q5);save('Q4_Q5_COMBINED_REPORT.md',q4+['---']+q5)
    (folder/'REPRODUCIBILITY.md').write_text('# 六组复现\n\n`powershell -ExecutionPolicy Bypass -File rag/run_local.ps1 scripts/22_six_group_study.py run`\n\n'
        '入口先校验原始核心代码、数据和scope_manifest中的六组适配器哈希。复用结果保留原始调用，新增S/T有独立配置ID。'
        '密钥只从本地环境读取。旧19组历史目录不再作为本轮运行目录。\n',encoding='utf-8')
    (out/'README.md').write_text(f'# 六组全量实验\n\n0/A/B/G/S/T各1359题，共8154条目标回答。当前成功{total}条。'
        'Q4材料优先交付；最终自动及人工完成状态见reports/completion_audit.json。旧19组要求已由用户指令替代。\n',encoding='utf-8')
    return folder
