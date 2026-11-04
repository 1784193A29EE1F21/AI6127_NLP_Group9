from dataclasses import asdict
import pytest
from rag.ablations import generation_configs, retrieval_configs, paired_delta


def test_full_system_ablation_changes_exactly_one_feature():
    configs={name.split()[0]:(r,g) for name,r,g in generation_configs(True)}
    assert len(configs)==19
    full_r,full_g=configs['K']
    expected={'L':'rerank','M':'prf','N':'rewrite','O':'hyde','P':'max_per_doc',
              'Q':'self_check','R':'iterative'}
    full={**asdict(full_r),**asdict(full_g)}
    for code,feature in expected.items():
        r,g=configs[code]
        changed={k for k,v in {**asdict(r),**asdict(g)}.items() if v!=full[k]}
        assert changed=={feature}


def test_diversity_controls_keep_other_retrieval_settings_fixed():
    configs=retrieval_configs()
    assert len(configs)==10
    for name in ['Hybrid RRF','Hybrid RRF + rerank','Hybrid RRF + PRF + rerank']:
        a,b=asdict(configs[name]),asdict(configs[name+' + diversity'])
        assert a.pop('max_per_doc')==0
        assert b.pop('max_per_doc')==2
        assert a==b


def test_article_bootstrap_preserves_constant_paired_gain():
    result=paired_delta([0,0.5,0.25],[0.1,0.6,0.35],['a','a','b'],resamples=200)
    assert result['delta']==pytest.approx(0.1)
    assert result['ci95']==pytest.approx([0.1,0.1])
    assert result['n_source_articles']==2
    assert (result['wins'],result['ties'],result['losses'])==(3,0,0)


def test_paired_analysis_rejects_unmatched_or_empty_samples():
    for a,b,docs in [([],[],[]),([1],[1,2],['a']),([float('nan')],[0],['a'])]:
        with pytest.raises(ValueError):
            paired_delta(a,b,docs)
