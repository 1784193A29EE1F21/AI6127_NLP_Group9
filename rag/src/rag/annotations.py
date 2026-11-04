"""Align independent annotations without counting them as separate questions."""
from __future__ import annotations

import json
from pathlib import Path

DATASET_BASE = Path(__file__).resolve().parents[3] / '6127-data' / '6127-data'
HUMAN_DIRS = ['Data Annotation - original 00-49', 'Data Annotation - original 50-99']


def default_annotation_dirs():
    complete = [DATASET_BASE / name for name in HUMAN_DIRS]
    if all(p.is_dir() for p in complete):
        return complete
    return [Path(__file__).resolve().parents[3] / 'Data Annotation - 00-49']


def read_annotation_group(paths):
    records = {}
    for directory in paths:
        for file in sorted(Path(directory).rglob('*_qa.json')):
            for record in json.loads(file.read_text(encoding='utf-8-sig')):
                if record['id'] in records:
                    raise ValueError(f"Duplicate annotation ID {record['id']} in {file}")
                records[record['id']] = {**record, 'annotation_file': str(file.relative_to(DATASET_BASE))}
    return records


def attach_annotation_variants(rows, annotation_dirs):
    expected = {p.resolve() for p in default_annotation_dirs()}
    complete = { (DATASET_BASE/name).resolve() for name in HUMAN_DIRS }
    if {Path(p).resolve() for p in annotation_dirs} != complete or expected != complete:
        return {'attached': False, 'reason': 'custom annotation input'}
    groups = {'human_original': read_annotation_group([DATASET_BASE/name for name in HUMAN_DIRS]),
              'human_reannotated': read_annotation_group([DATASET_BASE/'Data Annotation - Reannotated']),
              'ai_chat': read_annotation_group([DATASET_BASE/'Data Annotation - AI'])}
    ids = {r['id'] for r in rows}
    for name, records in groups.items():
        if set(records) != ids:
            raise ValueError(f'{name}: annotation IDs do not match integrated questions')
    uncertain = disagreements = 0
    for row in rows:
        variants = {}
        for name, records in groups.items():
            r = records[row['id']]
            if r['question'].strip() != row['question']:
                raise ValueError(f"Question mismatch: {row['id']} / {name}")
            variants[name] = {k: r[k] for k in ['answer', 'evidence', 'question_type', 'annotation_file']}
            variants[name]['origin'] = 'AI' if name == 'ai_chat' else 'human'
            variants[name]['review_status'] = r.get('review_status', 'human_annotation')
        row['annotation_variants'] = variants
        row['canonical_annotation'] = 'human_original'
        row['adjudication_status'] = 'not_adjudicated'
        row['reference_answers'] = list(dict.fromkeys(variants[n]['answer'] for n in ['human_original','human_reannotated']))
        row['question_type_all_agree'] = len({v['question_type'] for v in variants.values()}) == 1
        row['ai_needs_review'] = variants['ai_chat']['review_status'] != 'reviewed'
        uncertain += row['ai_needs_review']
        disagreements += not row['question_type_all_agree']
    return {'attached': True, 'groups': {k:len(v) for k,v in groups.items()},
            'canonical': 'human_original', 'ai_needs_review': uncertain,
            'question_type_disagreements': disagreements,
            'note': 'Human references retained separately; AI answers are not canonical gold. No adjudication performed.'}
