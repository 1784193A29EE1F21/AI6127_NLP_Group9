"""Controlled ablations for every optional retrieval/generation enhancement."""
from dataclasses import replace
import numpy as np
from .pipeline import GenerationConfig, RetrievalConfig


def generation_configs(include_all=False):
    base = RetrievalConfig(rerank=False)
    rerank = replace(base, rerank=True)
    rewrite = replace(base, rewrite=True)
    combined = replace(rerank, rewrite=True)
    generation = GenerationConfig(style='concise')
    configs = [
        ('0 Closed-book LLM (no retrieval)', None, None),
        ('A Hybrid RAG (base)', base, generation),
        ('B + re-rank', rerank, generation),
        ('C + query rewrite', rewrite, generation),
        ('D + re-rank + rewrite', combined, generation),
        ('E D + self-check', combined, replace(generation,self_check=True)),
    ]
    if include_all:
        full_r = replace(rerank,prf=True,rewrite=True,hyde=True,max_per_doc=2)
        full_g = replace(generation,self_check=True,iterative=True,max_rounds=2)
        configs += [
            ('F D + iterative retrieval', combined, replace(generation,iterative=True,max_rounds=2)),
            ('G B + self-check', rerank, replace(generation,self_check=True)),
            ('H B + PRF', replace(rerank,prf=True), generation),
            ('I B + HyDE', replace(rerank,hyde=True), generation),
            ('J B + diversity', replace(rerank,max_per_doc=2), generation),
            ('K Full system', full_r, full_g),
            ('L Full minus rerank', replace(full_r,rerank=False), full_g),
            ('M Full minus PRF', replace(full_r,prf=False), full_g),
            ('N Full minus rewrite', replace(full_r,rewrite=False), full_g),
            ('O Full minus HyDE', replace(full_r,hyde=False), full_g),
            ('P Full minus diversity', replace(full_r,max_per_doc=0), full_g),
            ('Q Full minus self-check', full_r, replace(full_g,self_check=False)),
            ('R Full minus iterative', full_r, replace(full_g,iterative=False)),
        ]
    return configs


GENERATION_CONTRASTS = [
    ('RAG','0','A'),('rerank','A','B'),('rewrite','A','C'),
    ('rewrite_given_rerank','B','D'),('self_check','D','E'),
    ('iterative','D','F'),('self_check_without_rewrite','B','G'),
    ('PRF','B','H'),('HyDE','B','I'),('diversity','B','J'),
] + [('full_contribution_'+feature,code,'K') for feature,code in
      [('rerank','L'),('PRF','M'),('rewrite','N'),('HyDE','O'),
       ('diversity','P'),('self_check','Q'),('iterative','R')]]


def retrieval_configs():
    def cfg(**kwargs):
        return RetrievalConfig(top_k=10,rerank=False,**kwargs)
    base = cfg()
    return {
        'BM25': cfg(mode='bm25'),
        'BM25 + PRF': cfg(mode='bm25',prf=True),
        'Dense (multilingual-e5-small)':cfg(mode='dense'),
        'Hybrid RRF':base,
        'Hybrid RRF + PRF':replace(base,prf=True),
        'Hybrid RRF + rerank':replace(base,rerank=True),
        'Hybrid RRF + PRF + rerank':replace(base,rerank=True,prf=True),
        'Hybrid RRF + diversity':replace(base,max_per_doc=2),
        'Hybrid RRF + rerank + diversity':replace(base,rerank=True,max_per_doc=2),
        'Hybrid RRF + PRF + rerank + diversity':replace(base,rerank=True,prf=True,max_per_doc=2),
    }


RETRIEVAL_CONTRASTS = [
    ('hybrid_vs_sparse','BM25','Hybrid RRF'),
    ('hybrid_vs_dense','Dense (multilingual-e5-small)','Hybrid RRF'),
    ('rerank','Hybrid RRF','Hybrid RRF + rerank'),
    ('PRF','Hybrid RRF','Hybrid RRF + PRF'),
    ('PRF_given_rerank','Hybrid RRF + rerank','Hybrid RRF + PRF + rerank'),
    ('rerank_given_PRF','Hybrid RRF + PRF','Hybrid RRF + PRF + rerank'),
    ('diversity','Hybrid RRF','Hybrid RRF + diversity'),
    ('diversity_given_rerank','Hybrid RRF + rerank','Hybrid RRF + rerank + diversity'),
    ('diversity_given_PRF_rerank','Hybrid RRF + PRF + rerank','Hybrid RRF + PRF + rerank + diversity'),
    ('PRF_given_rerank_diversity','Hybrid RRF + rerank + diversity','Hybrid RRF + PRF + rerank + diversity'),
]


def paired_delta(control, treatment, doc_ids, seed=42, resamples=2000):
    """Paired differences, with source-article cluster bootstrap uncertainty."""
    if not (len(control) == len(treatment) == len(doc_ids)) or not len(control):
        raise ValueError('Matched nonempty control, treatment and document IDs required')
    delta = np.asarray(treatment,dtype=float) - np.asarray(control,dtype=float)
    if not np.isfinite(delta).all():
        raise ValueError('Non-finite paired metric')
    _, groups = np.unique(np.asarray(doc_ids,dtype=str),return_inverse=True)
    sizes = np.bincount(groups)
    sums = np.bincount(groups,weights=delta)
    draws = np.random.default_rng(seed).integers(0,len(sizes),size=(resamples,len(sizes)))
    means = sums[draws].sum(axis=1)/sizes[draws].sum(axis=1)
    return {'n':len(delta),'n_source_articles':len(sizes),'delta':float(delta.mean()),
            'ci95':[float(v) for v in np.quantile(means,[0.025,0.975])],
            'wins':int((delta>1e-12).sum()),'ties':int((np.abs(delta)<=1e-12).sum()),
            'losses':int((delta < -1e-12).sum()),'bootstrap_resamples':resamples,'seed':seed}
