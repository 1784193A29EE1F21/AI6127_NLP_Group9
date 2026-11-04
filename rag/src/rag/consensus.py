"""Select agreement cohorts without inventing semantic answer agreement."""
GROUPS={'human_original','human_reannotated','ai_chat'}


def select_type_consensus(rows, require_ai_reviewed=False):
    seen=set()
    selected=[]
    for row in rows:
        if row['id'] in seen:
            raise ValueError(f"Duplicate question ID: {row['id']}")
        seen.add(row['id'])
        variants=row['annotation_variants']
        if set(variants)!=GROUPS:
            raise ValueError(f"Three aligned annotation variants required: {row['id']}")
        agree=len({v['question_type'] for v in variants.values()})==1
        if agree and (not require_ai_reviewed or variants['ai_chat'].get('review_status')=='reviewed'):
            selected.append(row)
    return selected
