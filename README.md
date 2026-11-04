# WikiRAG — AI6127 Group 9

WikiRAG is an open-domain question-answering system over English Wikipedia. It builds a passage-level knowledge base offline, combines lexical and semantic search, and uses retrieved evidence to generate answers with source citations. A Gradio interface supports interactive queries and retrieval settings.

The project covers **knowledge-base construction, retrieval, interactive querying, answer generation, evaluation and ablation studies**.

## System overview

```text
Wikipedia articles → cleaning and chunking → passage corpus
                                           ├─ BM25 index ────┐
                                           └─ dense index ───┤
User question ──────────────────────────────────────────────┤
    → reciprocal-rank fusion → optional cross-encoder reranking
    → top passages → LLM answer with numbered source citations
    → optional self-check or iterative retrieval
```

Retrieval options include pseudo-relevance feedback, query rewriting, HyDE, title filtering and passage diversity. The default dense encoder is `intfloat/multilingual-e5-small`; the default reranker is `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`. Generation supports Gemini and OpenAI-compatible endpoints. The interface displays retrieved passages and links so users can inspect the evidence.

## Knowledge base and data

The original corpus was built from WikiExtractor output `wiki_output/AA/wiki_00` through `wiki_99`. The build cleans articles, removes duplicate material and short pages, and splits documents into passages with a target of 180 words, a maximum of 300 words and one overlapping sentence.

| Corpus statistic | Original build |
|---|---:|
| Documents | 4,608 |
| Passages | 115,325 |
| Clean document words | 15,919,108 |
| Lowercased word types | 242,586 |

The question-answer evaluation set contains **1,359 unique questions from 331 articles**, each with two human answers and evidence. The first answer is the primary reference; both are retained for multi-reference analysis. The records use JSONL and include fields such as `id`, `question`, `answer`, `evidence`, `question_type`, `source_url`, `gold_doc_id` and `gold_chunk_ids`. Question-type agreement and answer-level agreement are different measurements; independent answer-level agreement review is still pending.

**Data files, model weights, indexes and experiment results are distributed separately and are not in this code repository.** To reproduce the original corpus and passage IDs, use the same source WikiExtractor files; another Wikipedia snapshot may produce different IDs. The processed passage file exceeds GitHub's 100 MB per-file limit.

## Installation

Install Python 3.14 and [uv](https://docs.astral.sh/uv/), then:

```bash
git clone https://github.com/1784193A29EE1F21/AI6127_NLP_Group9.git
cd AI6127_NLP_Group9/rag
uv sync
```

Copy `.env.example` to `.env` and set the provider, model, endpoint and API key. Keep `.env` private. Model weights download on first use; generation and LLM-based evaluation may incur API charges.

## Build the knowledge base and run the UI

Place the source files in `wiki_output/AA/` at the repository root, or use `--raw_dir` to specify their location. From `rag/`:

```bash
uv run python scripts/01_build_corpus.py
uv run python scripts/03_build_index.py
uv run python app.py
```

The first command writes cleaned documents, passages and statistics to `data/corpus/`; the second builds sparse and dense indexes under `data/index/`. The app loads the indexes and prints a local Gradio URL. Indexing can take substantial time and memory. Restore the same corpus and indexes from an archive if they are already available.

Example queries include “Why did Ayn Rand reject libertarianism?” and “What is the difference between Ajax the Great and Ajax the Lesser?” The interface also supports title filtering and configurable retrieval and generation options.

## Evaluation workflow

Restore the evaluation JSONL under `rag/data/eval/` and build or restore the matching corpus and indexes before running these commands from `rag/`:

| Task | Command | Output |
|---|---|---|
| Retrieval evaluation | `uv run python scripts/04_eval_retrieval.py --eval data/eval/human_pair/eval_qa.jsonl` | `results/retrieval/` |
| Five queries and timing | `uv run python scripts/05_benchmark_queries.py` | `results/five_queries.json` and `.md` |
| Five queries with generation | `uv run python scripts/05_benchmark_queries.py --generate` | Same outputs, including answers |
| Small QA evaluation | `uv run python scripts/06_eval_generation.py --eval data/eval/human_pair/eval_qa.jsonl --n 10 --only A B` | `results/generation/` |
| Unit and integration tests | `uv run --with psutil --with pytest pytest tests` | Test results in the terminal |

Retrieval evaluation reports rank-aware measures including Precision@K, Recall@K, MRR and nDCG. Generation evaluation includes exact match, token F1 and ROUGE-L, together with model-judged answer correctness, faithfulness, relevance and context measures where available. Model judgments are not a substitute for independent human validation. Passage relevance labels derived from evidence matching also require human verification.

The five-query script measures retrieval stages after warm-up; generation is optional. The ten-question command is a small example, not the full formal evaluation. The full test suite expects the separately archived evaluation data; without it, two audit tests cannot run.

`scripts/02_build_eval.py` can build an evaluation set from annotation files and the passage corpus. Its original annotation directories must be restored separately. `scripts/iaa_make_sheets.py` and `scripts/iaa_compute.py` support annotation review; report the actual review protocol and completed labels when reporting agreement.

## Experiments and ablations

Retrieval experiments compare sparse, dense and hybrid search, with options such as reranking and query expansion. The selected full-data generation study uses six configurations:

| Group | Configuration |
|---|---|
| 0 | Closed-book generation |
| A | Hybrid retrieval |
| B | A + reranking |
| G | B + self-check |
| S | B + iterative retrieval |
| T | B + self-check + iterative retrieval |

Generation and scoring for 0, A and G were completed on 1,359 questions each; B, S and T were incomplete when execution stopped. The automated Q4 audit and independent human answer-agreement review were not complete at that point. Results, timings and experimental metadata are stored separately.

`scripts/04_eval_retrieval.py`, `06_eval_generation.py` and `07_run_ablations.py` cover retrieval and generation comparisons. The full-study orchestration is in `scripts/16_formal_full.py`, `19_resume_fast.py` and `22_six_group_study.py`. These repository copies omit the original automatic writing-bundle step, so their source hashes differ from the frozen original run. Resume that run with its archived original source, manifests and state, rather than assuming this checkout is a resumable checkpoint.

## Code layout

| Path | Purpose |
|---|---|
| `rag/app.py` | Gradio interface |
| `rag/src/rag/corpus.py`, `text.py` | Corpus cleaning, chunking and tokenization |
| `rag/src/rag/bm25.py`, `dense.py` | Sparse and dense retrieval |
| `rag/src/rag/fusion_rerank.py`, `query.py` | Rank fusion, reranking and query enhancements |
| `rag/src/rag/pipeline.py`, `generator.py`, `llm.py` | End-to-end retrieval and grounded generation |
| `rag/src/rag/metrics.py`, `judge.py` | Evaluation metrics and model judging |
| `rag/src/rag/formal/`, `six/` | Formal experiment and six-group study logic |
| `rag/scripts/`, `rag/tests/` | Build/evaluation scripts and tests |

The assignment submission also requires links in the final PDF to separately packaged data and source code. Respect the source Wikipedia license and attribution requirements when distributing the corpus.
