"""BM25 sparse retriever implemented from scratch on a SciPy inverted index."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

from .text import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.W: sparse.csc_matrix | None = None  # (n_docs, n_terms) BM25 weights
        self.doc_freq: np.ndarray | None = None
        self.n_docs = 0

    # build
    def fit(self, token_lists: list[list[str]]) -> "BM25Index":
        self.vocab = {}
        self._csr = None
        self._inv = None
        rows, cols, vals = [], [], []
        doc_len = np.zeros(len(token_lists), dtype=np.float32)
        for i, toks in enumerate(token_lists):
            doc_len[i] = len(toks)
            for t, c in Counter(toks).items():
                j = self.vocab.setdefault(t, len(self.vocab))
                rows.append(i)
                cols.append(j)
                vals.append(c)
        self.n_docs = len(token_lists)
        tf = sparse.csr_matrix((np.asarray(vals, np.float32), (rows, cols)),
                               shape=(self.n_docs, len(self.vocab)))
        df = np.bincount(cols, minlength=len(self.vocab)).astype(np.float32)
        self.doc_freq = df
        self.idf = np.log1p((self.n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        avgdl = max(float(doc_len.mean()), 1.0) if self.n_docs else 1.0
        norm = self.k1 * (1 - self.b + self.b * doc_len / avgdl)  # per document
        tf = tf.tocoo()
        w = tf.data * (self.k1 + 1) / (tf.data + norm[tf.row]) * self.idf[tf.col]
        self.W = sparse.csc_matrix((w.astype(np.float32), (tf.row, tf.col)), shape=tf.shape)
        return self

    # query
    def query_vector(self, query: str | dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        """Map a query (text, or {token: weight} for expanded queries) to term ids/weights."""
        if isinstance(query, str):
            weights = Counter(tokenize(query))
        else:
            weights = query
        ids, ws = [], []
        for t, w in weights.items():
            j = self.vocab.get(t)
            if j is not None:
                ids.append(j)
                ws.append(w)
        return np.asarray(ids, dtype=np.int64), np.asarray(ws, dtype=np.float32)

    def scores(self, query: str | dict[str, float]) -> np.ndarray:
        ids, ws = self.query_vector(query)
        if len(ids) == 0:
            return np.zeros(self.n_docs, dtype=np.float32)
        return np.asarray(self.W[:, ids] @ ws).ravel()

    def search(self, query: str | dict[str, float], top_k: int = 100,
               mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        s = self.scores(query)
        if mask is not None:
            s = np.where(mask, s, -np.inf)
        k = min(top_k, int((s > 0).sum()))
        if k <= 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        idx = np.argpartition(-s, k - 1)[:k]
        idx = idx[np.argsort(-s[idx])]
        return idx, s[idx]

    # helpers
    def rows(self) -> sparse.csr_matrix:
        """Row-major copy (doc -> terms), used by pseudo-relevance feedback."""
        if getattr(self, "_csr", None) is None:
            self._csr = self.W.tocsr()
        return self._csr

    def inv_vocab(self) -> list[str]:
        if getattr(self, "_inv", None) is None:
            inv = [""] * len(self.vocab)
            for t, j in self.vocab.items():
                inv[j] = t
            self._inv = inv
        return self._inv

    # io
    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(directory / "bm25_weights.npz", self.W)
        np.save(directory / "bm25_idf.npy", self.idf)
        np.save(directory / "bm25_df.npy", self.doc_freq)
        with open(directory / "bm25_meta.json", "w", encoding="utf-8") as f:
            json.dump({"k1": self.k1, "b": self.b, "n_docs": self.n_docs, "vocab": self.vocab}, f)

    @classmethod
    def load(cls, directory: Path) -> "BM25Index":
        directory = Path(directory)
        with open(directory / "bm25_meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        obj = cls(meta["k1"], meta["b"])
        obj.vocab, obj.n_docs = meta["vocab"], meta["n_docs"]
        obj.W = sparse.load_npz(directory / "bm25_weights.npz").tocsc()
        obj.idf = np.load(directory / "bm25_idf.npy")
        obj.doc_freq = np.load(directory / "bm25_df.npy")
        return obj
