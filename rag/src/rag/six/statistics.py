import json
from pathlib import Path
from rag.formal.analysis import QA,RAG
from rag.formal.statistics import compare,holm
from rag.formal.preflight import csv_file
from rag.formal.contract import write_json
from .scope import CONTRASTS


def paired_statistics(out,predictions,evaluations,questions):
    by_id={q['id']:q for q in questions};results=[];differences=[];interactions=[]
    for key in QA+RAG:
        source=predictions if key in QA else evaluations
        for left,right in CONTRASTS:
            ids=sorted(i for i in set(source[left])&set(source[right]) if all(
                source[c][i].get('execution_status')=='success' and source[c][i].get(key) is not None for c in [left,right]))
            if not ids:continue
            a=[source[left][i][key] for i in ids];b=[source[right][i][key] for i in ids]
            results.append({'control':left,'treatment':right,'metric':key,
                **compare(a,b,[by_id[i]['gold_doc_id'] for i in ids])})
            differences.extend({'control':left,'treatment':right,'metric':key,'id':i,
                'source_article':by_id[i]['gold_doc_id'],'delta':y-x} for i,x,y in zip(ids,a,b))
        family=[row for row in results if row['metric']==key]
        for row,p in zip(family,holm([row['p_value'] for row in family])):
            row.update(p_holm=p,holm_family='Seven planned six-group contrasts per outcome',significant_holm_005=p<.05)
        ids=sorted(i for i in set.intersection(*(set(source[c]) for c in ['B','G','S','T'])) if all(
            source[c][i].get('execution_status')=='success' and source[c][i].get(key) is not None for c in ['B','G','S','T']))
        if ids:
            values=[source['T'][i][key]-source['G'][i][key]-source['S'][i][key]+source['B'][i][key] for i in ids]
            interactions.append({'metric':key,'definition':'T - G - S + B','exploratory':True,
                **compare([0.]*len(ids),values,[by_id[i]['gold_doc_id'] for i in ids])})
    if results:
        flat=[{k:v for k,v in row.items() if not isinstance(v,(list,dict))} for row in results]
        csv_file(out/'ablation/paired_comparisons.csv',list(flat[0]),flat)
    write_json(out/'ablation/bootstrap_confidence_intervals.json',{'comparisons':results,'interaction':interactions,
        'resamples':2000,'seed':42,'resampling_unit':'source article','primary_outcome':'canonical-reference F1',
        'scope':'0/A/B/G/S/T only; not the cancelled nineteen-group protocol',
        'CI_note':'Unadjusted percentile 95% intervals; Holm p-values across seven contrasts within each outcome.'})
    path=out/'ablation/paired_differences.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        for row in differences:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    return results
