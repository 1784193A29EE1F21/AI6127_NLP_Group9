from copy import deepcopy
from rag.formal.contract import matrix as original_matrix

CODES=['0','A','B','G','S','T']
CONTRASTS=[('0','A'),('A','B'),('B','G'),('B','S'),('G','T'),('S','T'),('B','T')]


def matrix():
    old=original_matrix()
    result={c:deepcopy(old[c]) for c in ['0','A','B','G']}
    result['S']=deepcopy(old['B'])
    result['S']['name']='S Hybrid + rerank + iterative retrieval'
    result['S']['generation']['iterative']=True
    result['T']=deepcopy(result['S'])
    result['T']['name']='T Hybrid + rerank + self-check + iterative retrieval'
    result['T']['generation']['self_check']=True
    return result


def check_matrix(configs):
    expected=matrix()
    if configs!=expected:raise ValueError('Six-group definitions differ from approved 0/A/B/G/S/T scope')
    def flat(c):return {**configs[c]['retrieval'],**configs[c]['generation']}
    for left,right,key in [('A','B','rerank'),('B','G','self_check'),('B','S','iterative'),
                           ('G','T','iterative'),('S','T','self_check')]:
        a,b=flat(left),flat(right)
        if {k for k in a if a[k]!=b[k]}!={key}:raise ValueError('Confounded six-group contrast')
    return True


def install():
    # Original core source stays immutable; this adapter and its hash are archived separately.
    from rag.formal import runner,analysis,reporting
    from .statistics import paired_statistics
    from .reporting import reports,figures
    runner.matrix=matrix
    analysis.matrix=matrix
    analysis.paired_statistics=paired_statistics
    reporting.reports=reports
    reporting.figures=figures
