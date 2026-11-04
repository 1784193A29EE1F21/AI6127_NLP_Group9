"""Gradio web UI.

Usage: uv run python app.py
"""
from __future__ import annotations

import html
import os

import gradio as gr

from rag.pipeline import GenerationConfig, RAGSystem, RetrievalConfig

print("Loading indexes and models ...")
RAG = RAGSystem()
RAG.warmup()
HAS_LLM = bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
               or os.environ.get("LLM_PROVIDER", "gemini") == "openai")
print(f"Ready: {len(RAG.chunks):,} passages. LLM available: {HAS_LLM}")

EXAMPLES = [
    ["When and where was An American in Paris first performed, and who conducted the premiere?", ""],
    ["Why did Ayn Rand reject libertarianism?", ""],
    ["酸雨是如何形成的？它对环境有什么影响？", ""],
    ["What is the difference between Ajax the Great and Ajax the Lesser?", ""],
    ["What tools does it use to control inflation?", "central bank"],
]


def render_sources(hits: list[dict], cited: list[int]) -> str:
    cards = []
    for h in hits:
        star = " ⭐ cited" if h["rank"] in cited else ""
        meta = []
        if h.get("bm25_rank"):
            meta.append(f"BM25 #{h['bm25_rank']}")
        if h.get("dense_rank"):
            meta.append(f"Dense #{h['dense_rank']}")
        if h.get("rerank") is not None:
            meta.append(f"rerank {h['rerank']:.2f}")
        text = html.escape(h["text"][:700]) + ("…" if len(h["text"]) > 700 else "")
        cards.append(
            f"<div style='border:1px solid #ddd;border-radius:8px;padding:10px;margin:8px 0'>"
            f"<b>[{h['rank']}] <a href='{h['url']}' target='_blank'>{html.escape(h['title'])}</a></b>"
            f" › {html.escape(h['section'] or 'Introduction')}<span style='color:#c80'>{star}</span><br>"
            f"<small style='color:#777'>{' · '.join(meta)}</small>"
            f"<p style='margin:6px 0 0 0'>{text}</p></div>")
    return "".join(cards) or "<i>No passages found.</i>"


def search(query, title_filter, mode, rerank, prf, rewrite, hyde, diversity, top_k,
           generate, style, self_check, iterative):
    if not query.strip():
        return "Please type a question.", "", ""
    rcfg = RetrievalConfig(mode=mode, rerank=rerank, prf=prf, rewrite=rewrite and HAS_LLM,
                           hyde=hyde and HAS_LLM, top_k=int(top_k), title_filter=title_filter or "",
                           max_per_doc=2 if diversity else 0)
    if generate and HAS_LLM:
        try:
            res = RAG.answer(query, rcfg, GenerationConfig(style=style, self_check=self_check,
                                                           iterative=iterative))
        except Exception as e:  # show API problems instead of crashing the UI
            hits, info = RAG.retrieve(query, rcfg)
            return f"⚠️ LLM error: {e}", render_sources(hits, []), ""
        hits, T = res["hits"], res["timings"]
        answer = res["answer"]
        sc = res.get("self_check")
        if sc and sc["unsupported"]:
            answer += "\n\n> ⚠️ **Self-check removed / flagged unsupported claims:**\n" + "\n".join(
                f"> - {c}" for c in sc["unsupported"])
        elif sc:
            answer += f"\n\n✅ Self-check: all {len(sc['claims'])} claims are supported by the sources."
        cited = res["citations"]
        extra = res
    else:
        hits, info = RAG.retrieve(query, rcfg)
        T, cited, extra = info["timings"], [], info
        answer = ("*(retrieval only — tick **Generate answer** and set an API key for an LLM answer)*"
                  if not HAS_LLM or not generate else "")
    notes = []
    if extra.get("rewrite"):
        notes.append(f"🔁 Rewritten query: *{extra['rewrite']['english_query']}*  "
                     f"(keywords: {', '.join(extra['rewrite']['keywords'])})")
    if extra.get("note"):
        notes.append("ℹ️ " + extra["note"])
    for r in extra.get("iterative_rounds", []) or []:
        if not r["sufficient"]:
            notes.append(f"🔎 Follow-up retrieval: *{r['follow_up_query']}*")
    timing = " · ".join(f"{k} {1000 * v:.0f} ms" for k, v in T.items())
    meta = "\n\n".join(notes + [f"⏱️ {timing}"])
    return answer, render_sources(hits, cited), meta


with gr.Blocks(title="WikiRAG – AI6127") as demo:
    gr.Markdown("## 📚 WikiRAG — ask questions about Wikipedia articles\n"
                "Hybrid retrieval (BM25 + dense) → cross-encoder re-ranking → grounded answer with citations. "
                "You can ask in English or Chinese.")
    with gr.Row():
        q = gr.Textbox(label="Question", placeholder="e.g. Why did the Apollo 11 landing almost abort?", scale=4)
        tf = gr.Textbox(label="Only articles whose title contains (optional, comma-separated)", scale=2)
    btn = gr.Button("Search", variant="primary")
    with gr.Accordion("Options", open=False):
        with gr.Row():
            mode = gr.Radio(["hybrid", "bm25", "dense"], value="hybrid", label="Retriever")
            top_k = gr.Slider(1, 10, value=5, step=1, label="Passages")
        with gr.Row():
            rerank = gr.Checkbox(True, label="Cross-encoder re-ranking")
            prf = gr.Checkbox(False, label="Pseudo-relevance feedback (BM25)")
            rewrite = gr.Checkbox(False, label="LLM query rewrite / translate")
            hyde = gr.Checkbox(False, label="HyDE")
            diversity = gr.Checkbox(False, label="Max 2 passages per article")
        with gr.Row():
            generate = gr.Checkbox(HAS_LLM, label="Generate answer (LLM)")
            style = gr.Radio(["detailed", "concise"], value="detailed", label="Answer style")
            self_check = gr.Checkbox(True, label="Faithfulness self-check")
            iterative = gr.Checkbox(False, label="Iterative retrieval")
    answer = gr.Markdown(label="Answer")
    meta = gr.Markdown()
    gr.Markdown("### Sources")
    sources = gr.HTML()
    inputs = [q, tf, mode, rerank, prf, rewrite, hyde, diversity, top_k, generate, style, self_check, iterative]
    btn.click(search, inputs, [answer, sources, meta])
    q.submit(search, inputs, [answer, sources, meta])
    gr.Examples(EXAMPLES, [q, tf])

if __name__ == "__main__":
    demo.launch(inbrowser=os.environ.get("RAG_OPEN_BROWSER", "1") == "1")
