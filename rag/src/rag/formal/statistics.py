"""Paired source-article bootstrap with a centered-null test and Holm adjustment."""
import numpy as np
from rag.ablations import paired_delta


def compare(control,treatment,articles,resamples=2000,seed=42):
    result=paired_delta(control,treatment,articles,seed=seed,resamples=resamples)
    left,right=np.asarray(control,dtype=float),np.asarray(treatment,dtype=float)
    delta=right-left
    _,groups=np.unique(np.asarray(articles,dtype=str),return_inverse=True)
    counts=np.bincount(groups); sums=np.bincount(groups,weights=delta)
    draws=np.random.default_rng(seed).integers(0,len(counts),size=(resamples,len(counts)))
    boot=sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
    observed=float(delta.mean())
    # Center bootstrap distribution on the null; finite Monte Carlo correction avoids p=0.
    centered=boot-observed
    p=(1+int(np.sum(np.abs(centered)>=abs(observed)-1e-15)))/(resamples+1)
    result.update(control_mean=float(left.mean()),treatment_mean=float(right.mean()),
        relative_change=observed/float(left.mean()) if abs(float(left.mean()))>1e-15 else None,
        p_value=min(1.,p),test='two-sided centered article-cluster bootstrap; approximate',
        delta_quantiles=[float(x) for x in np.quantile(delta,[0,.25,.5,.75,1])])
    return result


def holm(p_values):
    p=np.asarray(p_values,dtype=float)
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any(): raise ValueError('Invalid p-values')
    order=np.argsort(p); adjusted=np.empty(len(p)); running=0.
    for rank,index in enumerate(order):
        running=max(running,float(p[index])*(len(p)-rank))
        adjusted[index]=min(1.,running)
    return adjusted.tolist()
