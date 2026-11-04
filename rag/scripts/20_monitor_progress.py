"""Read-only progress monitor. No model calls and no credential output."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]/'results/q4_q5_formal_full'
ACTIVE_POINTER=ROOT.parent/'active_experiment.json'
CODES=['0']+list('ABCDEFGHIJKLMNOPQR')


def read(path):
    try:return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError,PermissionError,json.JSONDecodeError):return {}


def files(path):return sum(1 for _ in path.glob('*.json'))


def processes():
    try:import psutil
    except ImportError:return ['无法读取进程：psutil 不可用；不能据此判断任务停止。']
    result=[]; seen=set()
    candidates=[(read(ROOT/'config/runtime_active.json'),'pid'),
                (read(ROOT/'supervisor_status.json'),'child_pid'),
                (read(ROOT/'supervisor_status.json'),'supervisor_pid')]
    for info,key in candidates:
        pid=info.get(key)
        if not pid or pid in seen:continue
        seen.add(pid)
        try:
            process=psutil.Process(pid)
            if process.is_running() and any('formal' in arg or '19_resume_fast.py' in arg or '22_six_group_study.py' in arg for arg in process.cmdline()):
                result.append(f'运行中 PID={pid} ({process.name()})')
        except psutil.NoSuchProcess:pass
        except psutil.AccessDenied:result.append(f'PID={pid} 状态不可读')
    return result or ['未发现已记录的运行进程；请检查日志，不能只根据旧检查点判断仍在运行。']


def show():
    global ROOT
    pointer=read(ACTIVE_POINTER)
    if pointer.get('directory'):ROOT=Path(pointer['directory'])
    state=read(ROOT/'generation/execution_status.json')
    codes=read(ROOT/'config/scope_manifest.json').get('configuration_ids',CODES)
    target=1359*len(codes)
    active=read(ROOT/'config/runtime_active.json')
    now=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
    print(f'\nWikiRAG 全量实验进度 | {now} 北京时间')
    print('；'.join(processes()))
    print(f"当前检查点阶段：{state.get('phase','尚无检查点')} | 配置：{state.get('active_configuration') or '-'}")
    print(f"检查点更新时间：{state.get('updated_at','未知')}（原始 UTC 时间）")
    options=active.get('effective_options') or read(ROOT/'config/model_config.json').get('options',{})
    print(f"运行参数：{options.get('rpm','未知')} RPM；{options.get('workers','未知')} 并发")
    print('\n配置     生成成功/1359   生成失败   评分成功/1359   评分已存文件')
    print('-'*70)
    total_generated=total_judged=0
    for code in codes:
        count=state.get('counts',{}).get(code,{})
        generated=count.get('success'); judged=count.get('judged')
        saved=files(ROOT/'evaluation/records'/code/'final')
        total_generated+=generated or 0;total_judged+=judged or 0
        print(f"{code:<8} {str(generated) if generated is not None else '-':>5}/1359     {str(count.get('failed','-')):>5}       {str(judged) if judged is not None else '-':>5}/1359         {saved:>5}")
    if state:
        print(f'\n全部{len(codes)}组：生成成功 {total_generated:,}/{target:,}；最终评分成功 {total_judged:,}/{target:,}')
    else:
        print(f'\n全部{len(codes)}组，目标{target:,}条：正在恢复检查点，成功数量暂不显示；已保存文件仍保留。')
    print('成功数来自检查点，通常最多滞后约10条；已存文件可能含失败记录，不等于全部成功。')
    ready=read(ROOT/'deliveries/Q4_formal_ready.json')
    if ready:
        print('\nQ4 正式报告与数据：已就绪')
        print('材料目录：'+ready.get('materials_directory','未记录'))
        print('ZIP：'+ready.get('zip','未记录'))
    else:
        print('\nQ4 正式材料：尚未就绪（需0/A/G评分、自检前后评价、性能测试及校验完成）')
    audit=read(ROOT/'reports/completion_audit.json')
    if audit:print('全量自动校验：'+str(audit.get('automated_complete'))+'；独立人工审核见报告。')
    print('日志：'+str(ROOT/'fast.stdout.log'))
    print('错误日志：'+str(ROOT/'fast.stderr.log'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch',type=int,default=0,metavar='SECONDS',help='定时刷新；0为只显示一次')
    args=parser.parse_args()
    if args.watch<0:parser.error('--watch 必须大于等于0')
    try:
        while True:
            show()
            if not args.watch:break
            print(f'\n{args.watch}秒后刷新；按 Ctrl+C 退出监控，不会停止实验。',flush=True)
            time.sleep(args.watch)
    except KeyboardInterrupt:print('\n监控已退出；实验继续运行。')


if __name__=='__main__':main()
