# WikiRAG — AI6127 Group 9

English Wikipedia question answering with a locally constructed knowledge base, sparse and dense retrieval, reciprocal-rank fusion, optional cross-encoder reranking, and evidence-grounded answer generation with citations and self-checking. The Gradio interface is in `rag/app.py`.

## Setup

Install Python 3.14 and [uv](https://docs.astral.sh/uv/), then run from `rag/`:

```bash
uv sync
```

Copy `.env.example` to `.env` and configure a model provider, endpoint, model and API key. Never commit `.env`. Model weights are downloaded on first use. Indexing and generation require substantial compute, time, and, for paid endpoints, money.

## Corpus, index and interface

The original WikiExtractor output is distributed separately: place `wiki_00` through `wiki_99` in `wiki_output/AA/` at repository root, or supply `--raw_dir` to the corpus command. These original files are needed to reproduce the reported corpus and passage identifiers; a different Wikipedia dump or the optional Hugging Face source is not necessarily equivalent.

```bash
cd rag
uv run python scripts/01_build_corpus.py
uv run python scripts/03_build_index.py
uv run python app.py
```

Open the URL printed by Gradio. The original corpus and prebuilt indexes are not in this repository. The processed `chunks.jsonl` exceeds GitHub's 100 MB file limit; supply the separately archived data or rebuild it. The assignment also requires a separate compressed data package and a source-code archive linked from the submitted PDF. This repository alone does not satisfy that submission requirement.

## Evaluation

The evaluation data is distributed separately. The two-human evaluation dataset `rag/data/eval/human_pair/eval_qa.jsonl` contains 1,359 questions with two human answers and evidence; the first human answer is the primary reference and the two answers can also be used as multiple references. Each line contains question ID, question, reference answers, evidence, source, question type and relevant passage IDs. Answer-level agreement has not yet been independently verified.

The retrieval command below defaults to the earlier integrated dataset, which is not included here. Pass `--help` to check its evaluation-file option and restore the necessary dataset if needed. The formal validation code also requires the separately archived integrated dataset and `consensus_type/`; neither is the Q4 evaluation set.

```bash
uv run python scripts/04_eval_retrieval.py
uv run python scripts/05_benchmark_queries.py --generate
uv run python scripts/06_eval_generation.py --eval data/eval/human_pair/eval_qa.jsonl --n 10 --only A B
uv run --with psutil --with pytest pytest tests
```

The sample generation command makes paid model calls if the configured endpoint is paid. It is a small example, not the full formal study.

## Experiments and current status

The approved full-data comparison uses six configurations: `0` closed-book; `A` hybrid; `B` hybrid + reranker; `G` B + self-check; `S` B + iterative retrieval; `T` B + self-check + iterative retrieval. Generation and scoring for 0/A/G were completed on 1,359 questions each. B/S/T were not complete when the processes were stopped. Results and experimental metadata are retained separately. The Q4 automated audit and independent human answer-agreement review were not complete at the time of this export. Model-judge metrics are not independent human judgments.

Experiment orchestration is in `rag/scripts/16_formal_full.py`, `19_resume_fast.py`, `22_six_group_study.py`, `rag/src/rag/formal/` and `rag/src/rag/six/`. These repository copies omit the original automatic writing-bundle step. The original run's frozen source hashes therefore do not match these modified scripts: use the original archived source and experiment state to resume that run. A full run also requires its execution contract, corpus, indexes, model configuration and caches. Verify inputs and API costs before starting.

Sparse/dense retrieval, fusion, generation, metrics and evaluation implementations are in `rag/src/rag/`. Source annotations and large experiment archives are distributed separately. Wikipedia content remains subject to the source license and attribution requirements.
