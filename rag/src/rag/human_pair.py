"""Human-only evaluation records and explicit two-reference scoring."""
from copy import deepcopy
from .metrics import exact_match, token_f1, rouge_l

GROUPS = ('human_original','human_reannotated')


def human_only(rows):
    seen = set()
    result = []
    allowed = ('id','question','source_title','source_url','gold_doc_id','gold_chunk_ids',
               'best_evidence_recall','annotation_file')
    for row in rows:
        if row['id'] in seen:
            raise ValueError('Duplicate question ID')
        seen.add(row['id'])
        variants = row.get('annotation_variants',{})
        if any(g not in variants or variants[g].get('origin')!='human' for g in GROUPS):
            raise ValueError('Two human annotation groups required')
        item = {k:deepcopy(row[k]) for k in allowed if k in row}
        item['annotation_variants'] = {g:deepcopy(variants[g]) for g in GROUPS}
        for key in ('answer','evidence','question_type'):
            item[key] = variants[GROUPS[0]][key]
        if not item['answer'].strip() or any(not variants[g]['answer'].strip() for g in GROUPS):
            raise ValueError('Empty human answer')
        item['reference_answers'] = list(dict.fromkeys(variants[g]['answer'] for g in GROUPS))
        item['question_type_human_agree'] = variants[GROUPS[0]]['question_type']==variants[GROUPS[1]]['question_type']
        item['canonical_annotation'] = GROUPS[0]
        item['adjudication_status'] = 'two_human_references_not_adjudicated'
        result.append(item)
    return result


def reference_scores(answer, question):
    functions = {'EM':exact_match,'F1':token_f1,'ROUGE_L':rouge_l}
    result = {key:fn(answer,question['answer']) for key,fn in functions.items()}
    result['multi_reference'] = {key:max(fn(answer,ref) for ref in question['reference_answers'])
                                 for key,fn in functions.items()}
    return result


def validate_pair(data):
    if not isinstance(data,dict) or data.get('AB') not in {'agree','partial','disagree','uncertain'}:
        raise ValueError('Invalid two-human agreement decision')
    if not isinstance(data.get('reason'),str) or not data['reason'].strip():
        raise ValueError('Missing two-human agreement reason')
    return data


def validate_reference_claims(data):
    claims = data.get('reference_claims') if isinstance(data,dict) else None
    if not isinstance(claims,list) or not claims:
        raise ValueError('Missing reference claims for semantic context recall')
    if any(not isinstance(c,dict) or not isinstance(c.get('claim'),str) or not c['claim'].strip()
           or type(c.get('supported')) is not bool for c in claims):
        raise ValueError('Invalid reference claim support judgment')
    return claims
