"""Dense retriever: sentence-transformers encoder + exact inner-product search."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config  # Apply the PyTorch-only runtime settings before importing transformers.


def _prefixes(model_name: str) -> tuple[str, str]:
    """Some encoders are trained with instruction prefixes."""
    name = model_name.lower()
    if "e5" in name:
        return "query: ", "passage: "
    if "bge" in name and "m3" not in name and ("-en" in name or "-zh" in name):
        return "Represent this sentence for searching relevant passages: ", ""
    return "", ""


class DenseEncoder:
    def __init__(self, model_name: str, device: str | None = None, max_seq_length: int = 512):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        # paraphrase-MiniLM models only support 128 tokens; others go to 512
        self.model.max_seq_length = min(max_seq_length, self.model.max_seq_length or max_seq_length)
        self.q_prefix, self.p_prefix = _prefixes(model_name)

    @property
    def dim(self) -> int:
        m = self.model
        f = getattr(m, "get_embedding_dimension", None) or m.get_sentence_embedding_dimension
        return f()

    def encode_queries(self, queries: list[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode([self.q_prefix + q for q in queries], batch_size=batch_size,
                                 normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False).astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode_queries([query])[0]

    def encode_passages(self, passages: list[str], batch_size: int = 32,
                        show_progress_bar: bool = True) -> np.ndarray:
        return self.model.encode([self.p_prefix + p for p in passages], batch_size=batch_size,
                                 normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=show_progress_bar).astype(np.float32)


class DenseIndex:
    def __init__(self, embeddings: np.ndarray, model_name: str):
        self.emb = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.model_name = model_name

    def scores(self, qvec: np.ndarray) -> np.ndarray:
        return self.emb @ qvec

    def search(self, qvec: np.ndarray, top_k: int = 100,
               mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        s = self.scores(qvec)
        if mask is not None:
            s = np.where(mask, s, -np.inf)
        k = min(top_k, int(np.isfinite(s).sum()))
        if k <= 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        idx = np.argpartition(-s, k - 1)[:k]
        idx = idx[np.argsort(-s[idx])]
        return idx, s[idx]

    def save(self, directory: Path, name: str = "dense") -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / f"{name}_emb.npy", self.emb)
        with open(directory / f"{name}_meta.json", "w", encoding="utf-8") as f:
            json.dump({"model_name": self.model_name, "shape": list(self.emb.shape)}, f)

    @classmethod
    def load(cls, directory: Path, name: str = "dense") -> "DenseIndex":
        directory = Path(directory)
        with open(directory / f"{name}_meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        return cls(np.load(directory / f"{name}_emb.npy"), meta["model_name"])
