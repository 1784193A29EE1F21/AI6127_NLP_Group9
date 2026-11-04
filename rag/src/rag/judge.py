"""LLM-as-judge RAG metrics: faithfulness, answer relevance, context precision/recall."""
from __future__ import annotations

import numpy as np

from .generator import format_passages
from .llm import LLM
from .metrics import normalize_answer

JUDGE_PROMPT = """You are evaluating a retrieval-augmented question answering system.

QUESTION: {question}
GOLD ANSWER (written by a human annotator): {gold}

RETRIEVED PASSAGES:
{passages}

SYSTEM ANSWER: {answer}

Tasks:
1. "claims": split the SYSTEM ANSWER into atomic factual claims; for each, "supported" = true only if it is directly supported by the RETRIEVED PASSAGES (not by your own knowledge).
2. "passage_useful": for each retrieved passage in order, true if it contains information needed for the GOLD ANSWER.
3. "correctness": 1 if the SYSTEM ANSWER conveys the GOLD ANSWER (wording may differ), 0.5 if partially, 0 if wrong or missing.
4. "noncommittal": true if the SYSTEM ANSWER refuses or says the information is not available.
5. "generated_questions": 3 different questions that the SYSTEM ANSWER would be a good answer to.

Return JSON only:
{{"claims": [{{"claim": "...", "supported": true}}], "passage_useful": [true, false, ...], "correctness": 1, "noncommittal": false, "generated_questions": ["...", "...", "..."]}}"""


def context_precision(useful: list[bool]) -> float:
    num, hits = 0.0, 0
    for k, u in enumerate(useful, start=1):
        if u:
            hits += 1
            num += hits / k
    return num / hits if hits else 0.0


def context_recall(evidence: str, passages: list[str]) -> float:
    ev = set(normalize_answer(evidence).split())
    if not ev:
        return 0.0
    ctx = set(normalize_answer(" ".join(passages)).split())
    return len(ev & ctx) / len(ev)


class RAGJudge:
    def __init__(self, llm: LLM, encoder=None):
        self.llm = llm
        self.encoder = encoder  # DenseEncoder for answer relevance (optional)

    def evaluate(self, question: str, gold: str, evidence: str, answer: str, hits: list[dict]) -> dict:
        out = self.llm.generate_json(
            JUDGE_PROMPT.format(question=question, gold=gold, passages=format_passages(hits),
                                answer=answer), default={}) or {}
        claims = [c for c in out.get("claims", []) if isinstance(c, dict)]
        faith = (sum(bool(c.get("supported")) for c in claims) / len(claims)) if claims else (
            1.0 if out.get("noncommittal") else 0.0)
        useful = [bool(x) for x in (out.get("passage_useful") or [])][: len(hits)]
        try:
            corr = float(out.get("correctness", 0))
        except (TypeError, ValueError):
            corr = 0.0
        rel = None
        gq = [q for q in (out.get("generated_questions") or []) if isinstance(q, str) and q.strip()]
        if out.get("noncommittal"):
            rel = 0.0
        elif self.encoder is not None and gq:
            v = self.encoder.encode_queries([question] + gq)
            rel = float(np.mean(v[1:] @ v[0]))
        return {
            "faithfulness": faith,
            "answer_relevance": rel,
            "context_precision": context_precision(useful),
            "context_recall": context_recall(evidence, [h["text"] for h in hits]),
            "llm_correctness": corr,
            "noncommittal": bool(out.get("noncommittal", False)),
            "n_claims": len(claims),
            "unsupported_claims": [c.get("claim") for c in claims if not c.get("supported")],
        }
