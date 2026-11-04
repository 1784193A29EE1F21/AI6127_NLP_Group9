import json

from rag import annotations


def test_two_human_annotations_without_ai(tmp_path, monkeypatch):
    monkeypatch.setattr(annotations, 'DATASET_BASE', tmp_path)
    original = tmp_path / annotations.HUMAN_DIRS[0]
    second_batch = tmp_path / annotations.HUMAN_DIRS[1]
    reannotated = tmp_path / 'Data Annotation - Reannotated' / '00-49'
    for directory in (original, second_batch, reannotated):
        directory.mkdir(parents=True)
    first = {'id': 'q1', 'question': 'When?', 'answer': '1911',
             'evidence': 'In 1911.', 'question_type': 'factual'}
    second = {**first, 'answer': 'in 1911'}
    (original / 'wiki_00_qa.json').write_text(json.dumps([first]), encoding='utf-8')
    (reannotated / 'wiki_00_qa.json').write_text(json.dumps([second]), encoding='utf-8')
    rows = [dict(first)]
    result = annotations.attach_annotation_variants(rows, [original, second_batch])
    assert result['attached']
    assert set(rows[0]['annotation_variants']) == {'human_original', 'human_reannotated'}
    assert rows[0]['reference_answers'] == ['1911', 'in 1911']
    assert rows[0]['ai_needs_review'] is False
