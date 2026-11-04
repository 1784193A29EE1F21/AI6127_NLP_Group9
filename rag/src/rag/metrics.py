"""Retrieval metrics (P/R/Hit/MRR/nDCG) and QA metrics (EM/F1/ROUGE-L)."""
from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Iterable, Sequence


# retrieval
def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return sum(1 for d in ranked[:k] if d in relevant) / k


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return sum(1 for d in ranked[:k] if d in relevant) / len(relevant) if relevant else 0.0


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return float(any(d in relevant for d in ranked[:k]))


def mrr_at_k(ranked: Sequence[str], relevant: set[str], k: int = 10) -> float:
    for i, d in enumerate(ranked[:k], start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def retrieval_metrics(ranked: Sequence[str], relevant: set[str],
                      ks: Iterable[int] = (1, 3, 5, 10)) -> dict[str, float]:
    out = {}
    for k in ks:
        out[f"P@{k}"] = precision_at_k(ranked, relevant, k)
        out[f"R@{k}"] = recall_at_k(ranked, relevant, k)
        out[f"Hit@{k}"] = hit_at_k(ranked, relevant, k)
    out["MRR@10"] = mrr_at_k(ranked, relevant, 10)
    out["nDCG@10"] = ndcg_at_k(ranked, relevant, 10)
    return out


# QA
def normalize_answer(s: str) -> str:
    """SQuAD normalisation: lower, drop punctuation, articles and extra spaces."""
    s = s.lower()
    s = re.sub(r"\[\d+\]", " ", s)  # citation markers
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def token_recall(pred: str, gold: str) -> float:
    """Fraction of gold tokens present in the prediction (lenient for long answers)."""
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not g:
        return 0.0
    common = Counter(p) & Counter(g)
    return sum(common.values()) / len(g)


def _lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    lcs = _lcs(p, g)
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(g)
    return 2 * prec * rec / (prec + rec)


def bertscore_f1(preds: list[str], golds: list[str], model_type: str = "roberta-large") -> list[float] | None:
    """Optional: requires `pip install bert-score` (downloads roberta-large, ~1.4 GB)."""
    try:
        from bert_score import score
    except ImportError:
        return None
    _, _, f1 = score(preds, golds, lang="en", model_type=model_type, rescale_with_baseline=True,
                     verbose=False)
    return [float(x) for x in f1]
