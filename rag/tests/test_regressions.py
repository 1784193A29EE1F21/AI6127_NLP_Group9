import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from rag.corpus import Document, chunk_document
from rag.bm25 import BM25Index
from rag.text import tokenize


def test_chunk_maximum_with_long_overlap():
    sentences = [" ".join([word] * n) + "." for word, n in
                 [("Alpha", 175), ("Beta", 120), ("Gamma", 190), ("Delta", 20)]]
    doc = Document("1", "Test", "https://example.org", [("", [" ".join(sentences)])])
    chunks = chunk_document(doc)
    assert all(c.n_words <= 300 for c in chunks)
    assert all(word in " ".join(c.text for c in chunks) for word in ["Alpha", "Beta", "Gamma", "Delta"])


def test_overlap_after_target_does_not_exceed_limit():
    doc = Document("1", "Test", "", [("", [" ".join(["Alpha"] * 250) + ". "
                                           + " ".join(["Beta"] * 200) + "."])])
    assert all(c.n_words <= 300 for c in chunk_document(doc))


def test_refit_bm25_invalidates_vocabulary_and_cached_rows():
    index = BM25Index().fit([["old"]])
    index.rows()
    index.inv_vocab()
    index.fit([["new"], ["new", "term"]])
    assert "old" not in index.vocab
    assert index.rows().shape == (2, 2)
    assert set(index.inv_vocab()) == {"new", "term"}


def test_empty_annotations_do_not_overwrite(tmp_path):
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    corpus = tmp_path / "chunks.jsonl"
    corpus.write_text("", encoding="utf-8")
    output = tmp_path / "eval.jsonl"
    output.write_text("keep existing data", encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts/02_build_eval.py"
    result = subprocess.run([sys.executable, str(script), "--ann_dirs", str(annotations),
                             "--chunks", str(corpus), "--out", str(output)], capture_output=True)
    assert result.returncode != 0
    assert output.read_text(encoding="utf-8") == "keep existing data"


def test_bm25_roundtrip_and_filter(tmp_path):
    index = BM25Index().fit([tokenize("Acid rain damages forests"), tokenize("Paris is in France")])
    index.save(tmp_path)
    loaded = BM25Index.load(tmp_path)
    ids, scores = loaded.search("acid rain")
    assert ids.tolist() == [0]
    assert scores[0] > 0
    assert loaded.search("unmatchableword")[0].size == 0


def test_gemini_explicit_key(monkeypatch, tmp_path):
    import types
    import google
    fake = types.ModuleType("google.genai")
    fake.Client = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "google.genai", fake)
    monkeypatch.setattr(google, "genai", fake, raising=False)
    from rag.llm import LLM
    client = LLM(provider="gemini", api_key="test-judge-key", cache_dir=tmp_path)
    assert client._client["api_key"] == "test-judge-key"


def test_invalid_self_check_does_not_report_full_support():
    import pytest
    from rag.generator import self_check
    class InvalidJudge:
        def generate_json(self, *args, **kwargs):
            return {"claims": []}
    with pytest.raises(ValueError, match="could not be verified"):
        self_check(InvalidJudge(), "An unchecked answer [1].", [])
