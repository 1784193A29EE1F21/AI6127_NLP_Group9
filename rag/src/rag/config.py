"""Paths, model names and API settings (can be overridden in .env)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:  # python-dotenv is optional
    pass

# silence harmless Windows / Hugging Face warnings
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("USE_TF", "0")  # This project uses PyTorch, not TensorFlow.

PROJECT_ROOT = Path(os.environ.get("RAG_ROOT", Path(__file__).resolve().parents[2]))


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Paths:
    root: Path = PROJECT_ROOT
    raw_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw")
    corpus_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "corpus")
    eval_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "eval")
    index_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "index")
    results_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "results")
    cache_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "llm_cache")

    @property
    def chunks_file(self) -> Path:
        return self.corpus_dir / "chunks.jsonl"

    @property
    def docs_file(self) -> Path:
        return self.corpus_dir / "docs.jsonl"

    @property
    def eval_file(self) -> Path:
        return self.eval_dir / "eval_qa.jsonl"


PATHS = Paths()

# models
# multilingual encoder (512 tokens) so Chinese queries also work
EMBED_MODEL = _env("EMBED_MODEL", "intfloat/multilingual-e5-small")
RERANK_MODEL = _env("RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

# LLM
# "gemini" or "openai" (any OpenAI-compatible API: Groq, DeepSeek, Ollama ...)
LLM_PROVIDER = _env("LLM_PROVIDER", "gemini")
LLM_MODEL = _env("LLM_MODEL", "gemini-3.6-flash")
OPENAI_BASE_URL = _env("OPENAI_BASE_URL", "http://localhost:11434/v1")
LLM_RPM = int(_env("LLM_RPM", "0"))  # max requests/minute (0 = unlimited)

# optional separate judge model for evaluation (empty = same as generator)
JUDGE_PROVIDER = _env("JUDGE_PROVIDER", "")
JUDGE_MODEL = _env("JUDGE_MODEL", "")
JUDGE_BASE_URL = _env("JUDGE_BASE_URL", "")
JUDGE_API_KEY = _env("JUDGE_API_KEY", "")
