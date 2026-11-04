"""Answer generation with citations, faithfulness self-check, evidence check."""
from __future__ import annotations

import re

from .llm import LLM

ANSWER_PROMPT = """You are a careful question-answering assistant. Answer the QUESTION using ONLY the numbered PASSAGES from Wikipedia.

Rules:
- Every sentence of your answer must be supported by the passages; cite them with square brackets like [1] or [2][3] right after the claim (plain [n] only, no other citation format).
- If the passages do not contain the answer, say so clearly (e.g. "The retrieved passages do not say ...") instead of guessing.
- {style}
- Answer in the same language as the question.

PASSAGES:
{passages}

QUESTION: {question}
ANSWER:"""

STYLES = {
    "concise": "Be concise: 1-2 sentences that directly answer the question.",
    "detailed": "Give a complete but focused answer (one short paragraph, or bullet points for comparisons).",
}

AGENT_PROMPT = """You answer questions using ONLY the numbered PASSAGES from Wikipedia.

PASSAGES:
{passages}

QUESTION: {question}

Decide whether the passages contain enough evidence to answer the question completely.
Return JSON with keys:
  "sufficient": true or false,
  "missing": what information is still missing (empty string if sufficient),
  "follow_up_query": an English search query that would find the missing information (empty if sufficient)
Return JSON only."""

SELF_CHECK_PROMPT = """You are a strict fact-checker. Below are numbered PASSAGES and an ANSWER that should be fully grounded in them.

PASSAGES:
{passages}

ANSWER:
{answer}

Split the answer into individual factual claims. For each claim decide if it is directly supported by the passages.
Return JSON: {{"claims": [{{"claim": "...", "supported": true/false, "passage": <number or null>}}]}}
Return JSON only."""

REVISE_PROMPT = """Rewrite the ANSWER so that it keeps only the claims supported by the PASSAGES (keep the [n] citations).
Remove or explicitly mark as uncertain these UNSUPPORTED claims: {unsupported}
Answer in the same language as the original answer.

PASSAGES:
{passages}

ANSWER:
{answer}

REVISED ANSWER:"""


def format_passages(hits: list[dict], max_words: int = 350) -> str:
    out = []
    for i, h in enumerate(hits, start=1):
        words = h["text"].split()
        text = " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")
        header = h["title"] + (f" > {h['section']}" if h.get("section") else "")
        out.append(f"[{i}] ({header})\n{text}")
    return "\n\n".join(out)


_ODD_CITE = re.compile(r"[【\[]\s*(\d+)\s*(?:†[^】\]]*)?[】\]]")


def normalize_citations(text: str) -> str:
    """Turn model-specific citation styles like 【1】 or 【2†L9】 into [1], [2]."""
    text = _ODD_CITE.sub(lambda m: f"[{m.group(1)}]", text)
    return text.replace("[?]", "").strip()


def cited_numbers(answer: str) -> list[int]:
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})


def generate_answer(llm: LLM, question: str, hits: list[dict], style: str = "detailed") -> str:
    prompt = ANSWER_PROMPT.format(style=STYLES.get(style, style), passages=format_passages(hits),
                                  question=question)
    return normalize_citations(llm.generate(prompt, temperature=0.0))


def closed_book_answer(llm: LLM, question: str, style: str = "concise") -> str:
    """Baseline without retrieval (to show what RAG adds)."""
    prompt = f"Answer the question. {STYLES.get(style, style)}\nQUESTION: {question}\nANSWER:"
    return llm.generate(prompt, temperature=0.0).strip()


def assess_evidence(llm: LLM, question: str, hits: list[dict]) -> dict:
    out = llm.generate_json(AGENT_PROMPT.format(passages=format_passages(hits), question=question),
                            default={}) or {}
    return {"sufficient": bool(out.get("sufficient", True)), "missing": out.get("missing", ""),
            "follow_up_query": (out.get("follow_up_query") or "").strip()}


def self_check(llm: LLM, answer: str, hits: list[dict], revise: bool = True) -> dict:
    """Faithfulness self-check: flag unsupported claims and optionally revise."""
    passages = format_passages(hits)
    res = llm.generate_json(SELF_CHECK_PROMPT.format(passages=passages, answer=answer),
                            default={"claims": []}) or {"claims": []}
    if (not isinstance(res, dict) or not isinstance(res.get("claims"), list)
            or not res["claims"] or any(not isinstance(c, dict)
                or not isinstance(c.get("supported"), bool) or not c.get("claim")
                for c in res["claims"])):
        raise ValueError("Self-check returned invalid or empty claims; support could not be verified")
    claims = [c for c in res.get("claims", []) if isinstance(c, dict)]
    unsupported = [c.get("claim", "") for c in claims if not c.get("supported", True)]
    revised = answer
    if revise and unsupported:
        revised = normalize_citations(llm.generate(REVISE_PROMPT.format(
            unsupported="; ".join(unsupported), passages=passages, answer=answer)))
    n = len(claims)
    return {"claims": claims, "unsupported": unsupported, "revised_answer": revised,
            "support_ratio": (n - len(unsupported)) / n if n else 1.0}
