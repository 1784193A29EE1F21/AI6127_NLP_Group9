"""End-to-end RAG pipeline: retrieve -> fuse -> re-rank -> generate."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from . import config
from .bm25 import BM25Index
from .corpus import Chunk, load_chunks
from .dense import DenseEncoder, DenseIndex
from .fusion_rerank import CrossEncoderReranker, reciprocal_rank_fusion
from .generator import assess_evidence, cited_numbers, generate_answer, self_check
from .query import average_vectors, hyde_passage, is_non_english, prf_expand, rewrite_query


@dataclass
class RetrievalConfig:
    mode: str = "hybrid"          # "bm25" | "dense" | "hybrid"
    top_k: int = 5                # passages returned / given to the LLM
    k_sparse: int = 100           # candidates from BM25
    k_dense: int = 100            # candidates from the dense index
    rrf_k: int = 60
    rerank: bool = True
    rerank_candidates: int = 20   # how many fused candidates the cross-encoder re-scores
    prf: bool = False             # pseudo-relevance feedback for BM25
    rewrite: bool = False         # LLM query rewriting / translation
    hyde: bool = False            # LLM hypothetical document for dense search
    title_filter: str = ""        # metadata filter: only articles whose title contains this
    max_per_doc: int = 0          # diversity: max passages per article (0 = unlimited)


@dataclass
class GenerationConfig:
    style: str = "detailed"       # "concise" | "detailed"
    self_check: bool = False      # faithfulness self-check + revision
    iterative: bool = False       # agentic follow-up retrieval
    max_rounds: int = 2


class RAGSystem:
    def __init__(self, index_dir: Path | None = None, chunks_file: Path | None = None,
                 dense_name: str = "dense", rerank_model: str | None = None,
                 device: str | None = None, llm=None, load_dense: bool = True):
        index_dir = Path(index_dir or config.PATHS.index_dir)
        self.chunks: list[Chunk] = load_chunks(Path(chunks_file or config.PATHS.chunks_file))
        self.bm25 = BM25Index.load(index_dir)
        assert self.bm25.n_docs == len(self.chunks), "BM25 index does not match chunks.jsonl - rebuild it"
        self.dense: DenseIndex | None = None
        self.encoder: DenseEncoder | None = None
        if load_dense:
            self.dense = DenseIndex.load(index_dir, dense_name)
            assert self.dense.emb.shape[0] == len(self.chunks), "dense index does not match chunks.jsonl"
        self.device = device
        self.rerank_model = rerank_model or config.RERANK_MODEL
        self._reranker: CrossEncoderReranker | None = None
        self._llm = llm
        self._titles = np.array([c.title.lower() for c in self.chunks])

    # lazy parts
    def get_encoder(self) -> DenseEncoder:
        if self.encoder is None:
            self.encoder = DenseEncoder(self.dense.model_name, device=self.device)
        return self.encoder

    def get_reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(self.rerank_model, device=self.device)
        return self._reranker

    @property
    def llm(self):
        if self._llm is None:
            from .llm import LLM

            self._llm = LLM()
        return self._llm

    def warmup(self) -> None:
        if self.dense is not None:
            self.get_encoder().encode_query("warm up")
        self.get_reranker().score("warm up", ["warm up"])

    # retrieval
    def _mask(self, title_filter: str) -> np.ndarray | None:
        if not title_filter.strip():
            return None
        terms = [t.strip().lower() for t in title_filter.split(",") if t.strip()]
        mask = np.zeros(len(self.chunks), dtype=bool)
        for t in terms:
            mask |= np.char.find(self._titles, t) >= 0
        return mask

    def _hit(self, i: int) -> dict:
        c = self.chunks[i]
        return {"idx": int(i), "chunk_id": c.chunk_id, "doc_id": c.doc_id, "title": c.title,
                "section": c.section, "url": c.url, "text": c.text}

    def retrieve(self, query: str, cfg: RetrievalConfig | None = None) -> tuple[list[dict], dict]:
        cfg = cfg or RetrievalConfig()
        info: dict = {"timings": {}, "config": asdict(cfg)}
        T = info["timings"]
        mask = self._mask(cfg.title_filter)

        bm25_query, dense_queries, hyde_text = query, [query], None
        if cfg.rewrite:
            t = time.perf_counter()
            rw = rewrite_query(self.llm, query)
            T["rewrite"] = time.perf_counter() - t
            info["rewrite"] = rw
            bm25_query = rw["bm25_query"]
            if rw["english_query"].strip().lower() != query.strip().lower():
                dense_queries.append(rw["english_query"])
        if cfg.hyde and cfg.mode != "bm25":
            t = time.perf_counter()
            hyde_text = hyde_passage(self.llm, query)
            T["hyde"] = time.perf_counter() - t
            info["hyde"] = hyde_text

        lists, ranks = [], {}
        if cfg.mode in ("bm25", "hybrid"):
            t = time.perf_counter()
            q = prf_expand(bm25_query, self.bm25) if cfg.prf else bm25_query
            if cfg.prf:
                info["prf_terms"] = sorted(q, key=lambda k: -q[k])[:20]
            idx, _ = self.bm25.search(q, cfg.k_sparse, mask)
            T["bm25"] = time.perf_counter() - t
            lists.append(list(idx))
            ranks["bm25_rank"] = {int(d): r for r, d in enumerate(idx, 1)}
            if is_non_english(query) and not cfg.rewrite:
                info["note"] = ("Non-English query: BM25 cannot match an English corpus; "
                                "dense multilingual retrieval does the work (enable rewrite to translate).")
        if cfg.mode in ("dense", "hybrid"):
            t = time.perf_counter()
            enc = self.get_encoder()
            vecs = list(enc.encode_queries(dense_queries))
            if hyde_text:
                vecs.append(enc.encode_passages([hyde_text], show_progress_bar=False)[0])
            qv = average_vectors(vecs) if len(vecs) > 1 else vecs[0]
            idx, _ = self.dense.search(qv, cfg.k_dense, mask)
            T["dense"] = time.perf_counter() - t
            lists.append(list(idx))
            ranks["dense_rank"] = {int(d): r for r, d in enumerate(idx, 1)}

        t = time.perf_counter()
        if len(lists) == 2:
            fused = reciprocal_rank_fusion(lists, k=cfg.rrf_k)
        else:
            fused = [(int(d), 1.0 / (cfg.rrf_k + r)) for r, d in enumerate(lists[0], 1)] if lists else []
        T["fusion"] = time.perf_counter() - t

        n_cand = cfg.rerank_candidates if cfg.rerank else max(cfg.top_k * 4, 20)
        cands = fused[:n_cand]
        hits = []
        for d, s in cands:
            h = self._hit(d)
            h["rrf"] = round(float(s), 5)
            for name, rk in ranks.items():
                h[name] = rk.get(d)
            hits.append(h)

        if cfg.rerank and hits:
            t = time.perf_counter()
            rq = info.get("rewrite", {}).get("english_query", query)
            scores = self.get_reranker().score(rq, [self.chunks[h["idx"]].index_text() for h in hits])
            for h, s in zip(hits, scores):
                h["rerank"] = round(float(s), 4)
            hits.sort(key=lambda h: -h["rerank"])
            T["rerank"] = time.perf_counter() - t

        if cfg.max_per_doc > 0:
            per_doc: dict[str, int] = {}
            kept = []
            for h in hits:
                if per_doc.get(h["doc_id"], 0) < cfg.max_per_doc:
                    kept.append(h)
                    per_doc[h["doc_id"]] = per_doc.get(h["doc_id"], 0) + 1
            hits = kept
        hits = hits[: cfg.top_k]
        for r, h in enumerate(hits, 1):
            h["rank"] = r
        T["retrieval_total"] = sum(v for k, v in T.items() if k != "retrieval_total")
        return hits, info

    # generation
    def answer(self, query: str, rcfg: RetrievalConfig | None = None,
               gcfg: GenerationConfig | None = None) -> dict:
        rcfg, gcfg = rcfg or RetrievalConfig(), gcfg or GenerationConfig()
        t_all = time.perf_counter()
        calls_before = self.llm.stats.calls
        hits, info = self.retrieve(query, rcfg)
        T = info["timings"]
        rounds = []
        if gcfg.iterative:
            for _ in range(gcfg.max_rounds - 1):
                t = time.perf_counter()
                a = assess_evidence(self.llm, query, hits)
                T["assess"] = T.get("assess", 0) + time.perf_counter() - t
                rounds.append(a)
                if a["sufficient"] or not a["follow_up_query"]:
                    break
                more, follow_info = self.retrieve(a["follow_up_query"], replace(rcfg, rewrite=False, hyde=False))
                T["followup_retrieval"] = (T.get("followup_retrieval", 0)
                                           + follow_info["timings"]["retrieval_total"])
                seen = {h["chunk_id"] for h in hits}
                new = [h for h in more if h["chunk_id"] not in seen][: max(2, rcfg.top_k // 2)]
                hits = hits + new
                for r, h in enumerate(hits, 1):
                    h["rank"] = r
        t = time.perf_counter()
        draft = generate_answer(self.llm, query, hits, gcfg.style)
        T["generate"] = time.perf_counter() - t
        result = {"query": query, "answer": draft, "draft_answer": draft, "hits": hits,
                  "rewrite": info.get("rewrite"), "hyde": info.get("hyde"), "note": info.get("note"),
                  "iterative_rounds": rounds}
        if gcfg.self_check:
            t = time.perf_counter()
            sc = self_check(self.llm, draft, hits, revise=True)
            T["self_check"] = time.perf_counter() - t
            result["self_check"] = sc
            result["answer"] = sc["revised_answer"]
        result["citations"] = cited_numbers(result["answer"])
        T["total"] = time.perf_counter() - t_all
        result["timings"] = T
        result["llm_calls"] = self.llm.stats.calls - calls_before
        return result
