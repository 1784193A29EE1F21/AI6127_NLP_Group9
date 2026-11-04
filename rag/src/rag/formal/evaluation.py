"""Explicit primary/multi-reference scores and anonymous, uncertainty-aware judging."""
from collections import Counter
import re
from rag.generator import format_passages
from rag.human_pair import reference_scores
from rag.metrics import normalize_answer
from rag.judge import context_precision,context_recall


def token_pr(prediction,reference):
    p,g=normalize_answer(prediction).split(),normalize_answer(reference).split()
    if not p or not g:
        return (float(p==g),float(p==g))
    overlap=sum((Counter(p)&Counter(g)).values())
    return overlap/len(p),overlap/len(g)


def qa_scores(prediction,question):
    result=reference_scores(prediction,question)
    result['token_precision'],result['token_recall']=token_pr(prediction,question['answer'])
    for i,key in enumerate(['token_precision','token_recall']):
        result['multi_reference'][key]=max(token_pr(prediction,r)[i] for r in question['reference_answers'])
    return result


def citation_validity(answer,hits,closed=False):
    if closed: return {'citation_validity':None,'citation_numbers':[],'invalid_citations':[]}
    numbers=[int(n) for n in re.findall(r'\[(\d+)\]',answer)]
    invalid=[n for n in numbers if n<1 or n>len(hits)]
    return {'citation_validity':(len(numbers)-len(invalid))/len(numbers) if numbers else None,
            'citation_numbers':numbers,'invalid_citations':invalid}


def judge_prompt(question,answer,hits):
    variants=question['annotation_variants']
    return ('You are evaluating one anonymous question-answering result. Do not infer which experimental '
       'configuration produced it. Use only the supplied answer, references, and passages. '
       'Never assume an expected improvement. Compare against canonical human reference A for correctness; '
       'also provide correctness_multi accepting either human reference. If references conflict, explain it.\n\n'
       f"QUESTION: {question['question']}\n\nCANONICAL HUMAN REFERENCE A: {variants['human_original']['answer']}\n"
       f"ALTERNATIVE HUMAN REFERENCE B: {variants['human_reannotated']['answer']}\n\n"
       f"PASSAGES:\n{format_passages(hits)}\n\nSYSTEM ANSWER:\n{answer}\n\n"
       f'There are exactly {len(hits)} retrieved passages. passage_useful must contain exactly {len(hits)} elements, in passage order.\n'
       'Return JSON only, with exactly these required fields:\n'
       'claims: list of atomic factual claims with claim (string), supported (true/false/null for uncertain), '
       'reason (string), citations (list of citation numbers actually attached to this claim), '
       'citation_supported (true/false/null). Citation support means the specifically CITED passages support '
       'the claim, not merely that any retrieved passage supports it.\n'
       'passage_useful: one true/false/null per passage, whether needed for an essential reference answer fact.\n'
       'reference_claims: deduplicated essential canonical reference facts required by the question, '
       'each with claim, supported (true/false/null), reason indicating coverage by the retrieved context.\n'
       'correctness and correctness_multi: 1 (correct), 0.5 (partial), 0 (wrong or missing), or null (uncertain). '
       'correctness_reason: explain the assessment.\n'
       'answer_relevance: 1/0.5/0/null for directly/partly/not responsive/uncertain; relevance_reason.\n'
       'completeness: 1/0.5/0/null; completeness_reason; required_points: list of {point, covered:true/false/null}.\n'
       'noncommittal: boolean, true for refusal or missing-information response.\n'
       'generated_questions: three different questions that the SYSTEM ANSWER would answer well; '
       'may be empty for a refusal.\n'
       'If PASSAGES is empty, this is ungrounded QA: claims may describe the answer but supported and '
       'citation_supported must be null, passage_useful and reference_claims must be empty. '
       'Continue to assess correctness, relevance and completeness against the human references. '
       'Never silently convert uncertainty into a positive score.')


def validate_judge(raw,n_hits,answer=None):
    if not isinstance(raw,dict): raise ValueError('Judge must return a JSON object')
    for field in ['correctness','correctness_multi','answer_relevance','completeness']:
        v=raw.get(field,'missing')
        if v is not None and (type(v) not in (int,float) or v not in (0,.5,1)):
            raise ValueError('Invalid or missing '+field)
    for field in ['correctness_reason','relevance_reason','completeness_reason']:
        if not isinstance(raw.get(field),str) or not raw[field].strip(): raise ValueError('Missing '+field)
    if type(raw.get('noncommittal')) is not bool: raise ValueError('Missing noncommittal boolean')
    claims=raw.get('claims')
    if not isinstance(claims,list) or (not claims and not raw['noncommittal']): raise ValueError('Missing factual claims')
    for c in claims:
        if not isinstance(c,dict) or not isinstance(c.get('claim'),str) or not c['claim'].strip(): raise ValueError('Invalid claim text')
        for field in ['supported','citation_supported']:
            if field not in c or (c[field] is not None and type(c[field]) is not bool): raise ValueError('Invalid claim support')
        if not isinstance(c.get('reason'),str) or not c['reason'].strip(): raise ValueError('Missing claim reason')
        if not isinstance(c.get('citations'),list) or any(type(n) is not int for n in c['citations']): raise ValueError('Invalid claim citations')
        if answer is not None and not set(c['citations'])<=set(int(n) for n in re.findall(r'\[(\d+)\]',answer)):
            raise ValueError('Judge attributed a citation that does not occur in the actual answer')
    useful=raw.get('passage_useful')
    if not isinstance(useful,list) or len(useful)!=n_hits or any(x is not None and type(x) is not bool for x in useful):
        raise ValueError('Passage judgments must match the actual returned passages')
    refs=raw.get('reference_claims')
    if not isinstance(refs,list) or (n_hits and not refs): raise ValueError('Missing semantic reference claims')
    for c in refs:
        if not isinstance(c,dict) or not isinstance(c.get('claim'),str) or not c['claim'].strip(): raise ValueError('Invalid reference claim')
        if 'supported' not in c or (c['supported'] is not None and type(c['supported']) is not bool): raise ValueError('Invalid reference support')
        if not isinstance(c.get('reason'),str) or not c['reason'].strip(): raise ValueError('Missing reference support reason')
    points=raw.get('required_points')
    if not isinstance(points,list) or not points: raise ValueError('Required answer points must be explicit')
    for p in points:
        if not isinstance(p,dict) or not p.get('point') or 'covered' not in p or (p['covered'] is not None and type(p['covered']) is not bool):
            raise ValueError('Invalid required-point completeness judgment')
    gq=raw.get('generated_questions')
    if not isinstance(gq,list) or (len(gq)!=3 and not(raw['noncommittal'] and not gq)) or any(not isinstance(q,str) or not q.strip() for q in gq):
        raise ValueError('Invalid reverse questions')
    return raw


def mean_boolean(values):
    return sum(values)/len(values) if values and all(type(x) is bool for x in values) else None


def judge_metrics(raw,question,answer,hits):
    closed=not hits; claims=raw['claims']; cited=[c for c in claims if c['citations']]
    useful=raw['passage_useful']
    return {**citation_validity(answer,hits,closed),
        'faithfulness':None if closed else mean_boolean([c['supported'] for c in claims]),
        'answer_relevance':raw['answer_relevance'],'answer_relevance_reverse_question_proxy':None,
        'context_precision':context_precision(useful) if useful and all(type(x) is bool for x in useful) else None,
        'context_relevance':None if closed else mean_boolean(useful),
        'context_recall':None if closed else mean_boolean([c['supported'] for c in raw['reference_claims']]),
        'context_recall_lexical':None if closed else context_recall(question['evidence'],[h['text'] for h in hits]),
        'answer_correctness':raw['correctness'],'answer_correctness_multi':raw['correctness_multi'],
        'answer_completeness':raw['completeness'],'noncommittal':raw['noncommittal'],
        'citation_support':None if closed else mean_boolean([c['citation_supported'] if all(1<=n<=len(hits) for n in c['citations']) else False for c in cited]),
        'citation_claim_coverage':None if closed or not claims else len(cited)/len(claims),
        'n_claims':len(claims),'uncertain_support_claims':sum(c['supported'] is None for c in claims)}
