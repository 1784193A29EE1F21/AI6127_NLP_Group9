"""Query enhancement: pseudo-relevance feedback, LLM rewriting, HyDE."""
from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np

from .bm25 import BM25Index
from .llm import LLM
from .text import tokenize

_CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")


def is_non_english(text: str) -> bool:
    return bool(_CJK.search(text)) or sum(ord(c) > 127 for c in text) > 0.3 * max(1, len(text))


def prf_expand(query: str, bm25: BM25Index, fb_docs: int = 8, fb_terms: int = 10,
               orig_weight: float = 0.7, max_df_ratio: float = 0.1) -> dict[str, float]:
    """RM3-style expansion: add the best terms from the top BM25 passages to the query."""
    q = Counter(tokenize(query))
    if not q:
        return {}
    idx, scores = bm25.search(dict(q), top_k=fb_docs)
    if len(idx) == 0:
        return dict(q)
    scores = scores / (scores.sum() + 1e-9)
    rows = bm25.rows()
    inv_vocab = bm25.inv_vocab()
    term_w: dict[int, float] = defaultdict(float)
    for i, s in zip(idx, scores):
        r = rows[int(i)]
        for j, w in zip(r.indices, r.data):
            term_w[int(j)] += float(s) * float(w)
    max_df = max_df_ratio * bm25.n_docs
    cand = [(inv_vocab[j], w) for j, w in term_w.items()
            if bm25.doc_freq[j] <= max_df and inv_vocab[j] not in q and not inv_vocab[j].isdigit()]
    expansion = sorted(cand, key=lambda x: -x[1])[:fb_terms]
    qn = sum(q.values())
    out = {t: orig_weight * c / qn for t, c in q.items()}
    en = sum(w for _, w in expansion) or 1.0
    for t, w in expansion:
        out[t] = out.get(t, 0.0) + (1 - orig_weight) * w / en
    return out


REWRITE_PROMPT = """You rewrite user questions into search queries for an ENGLISH Wikipedia search engine.
Question: {query}

Return JSON with keys:
  "english_query": the question translated to clear English (keep names, dates, numbers),
  "keywords": 3-8 important keywords or synonyms (English) that a relevant passage would contain.
Return JSON only."""

HYDE_PROMPT = """Write a short, factual Wikipedia-style paragraph (60-100 words, English) that answers the question.
If you are unsure, still write a plausible paragraph - it is only used for search.
Question: {query}
Paragraph:"""


def rewrite_query(llm: LLM, query: str) -> dict:
    out = llm.generate_json(REWRITE_PROMPT.format(query=query), default={}) or {}
    eq = (out.get("english_query") or query).strip()
    kws = [k for k in (out.get("keywords") or []) if isinstance(k, str)]
    return {"english_query": eq, "keywords": kws, "bm25_query": eq + " " + " ".join(kws)}


def hyde_passage(llm: LLM, query: str) -> str:
    return llm.generate(HYDE_PROMPT.format(query=query), temperature=0.0).strip()


def average_vectors(vectors: list[np.ndarray]) -> np.ndarray:
    v = np.mean(np.stack(vectors), axis=0)
    return v / (np.linalg.norm(v) + 1e-9)
