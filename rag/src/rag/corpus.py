"""Parse WikiExtractor files, clean articles and split them into passages."""
from __future__ import annotations

import html
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

from .text import split_sentences

DOC_RE = re.compile(r'<doc id="(\d+)" url="([^"]*)" title="([^"]*)">\n?(.*?)</doc>', re.S)

SKIP_SECTIONS = {
    "see also", "references", "external links", "further reading", "notes", "bibliography",
    "sources", "citations", "footnotes", "works cited", "notes and references", "explanatory notes",
    "general and cited sources", "cited sources", "general sources", "selected bibliography",
}

_PLACEHOLDER = re.compile(r"\b(?:formula|codice)_\d+\b")
_EMPTY_BRACKETS = re.compile(r"\(\s*[;,]*\s*\)|\[\s*\]")
_SPACES = re.compile(r"[ \t ]+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")


@dataclass
class Document:
    doc_id: str
    title: str
    url: str
    sections: list[tuple[str, list[str]]]  # (section name, paragraphs)

    @property
    def n_words(self) -> int:
        return sum(len(p.split()) for _, ps in self.sections for p in ps)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    url: str
    text: str
    n_words: int

    @property
    def header(self) -> str:
        return f"{self.title} > {self.section}" if self.section else self.title

    def index_text(self) -> str:
        """Text that is indexed (header + body)."""
        return f"{self.header}\n{self.text}"


# input
def iter_wiki_files(raw_dir: Path) -> list[Path]:
    """All WikiExtractor output files (wiki_00, wiki_00.md, AA/wiki_00 ...) sorted."""
    files = [p for p in Path(raw_dir).rglob("wiki_*") if p.is_file() and not p.name.endswith("_qa.json")]
    return sorted(files, key=lambda p: (str(p.parent), p.name))


def iter_raw_docs_from_files(files: Iterable[Path]) -> Iterator[tuple[str, str, str, str]]:
    for f in files:
        content = f.read_text(encoding="utf-8", errors="ignore")
        for m in DOC_RE.finditer(content):
            doc_id, url, title, body = m.groups()
            yield doc_id, html.unescape(url), html.unescape(title), body


def iter_raw_docs_from_hf(max_doc_id: int | None = None, limit: int | None = None,
                          config: str = "20231101.en") -> Iterator[tuple[str, str, str, str]]:
    """Fallback: stream English Wikipedia from the Hugging Face Hub (wikimedia/wikipedia)."""
    from datasets import load_dataset  # optional dependency

    ds = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
    n = 0
    for row in ds:
        if max_doc_id is not None and int(row["id"]) > max_doc_id:
            continue  # HF rows are not guaranteed to be sorted by numeric page id.
        # HF text has headings as bare lines; mimic WikiExtractor ("Heading.")
        lines = []
        for line in row["text"].split("\n"):
            s = line.strip()
            if s and len(s.split()) <= 8 and not s.endswith((".", "!", "?", ":", '"')):
                s += "."
            lines.append(s)
        yield row["id"], row["url"], row["title"], row["title"] + "\n" + "\n".join(lines)
        n += 1
        if limit and n >= limit:
            break


# clean
def _is_heading(line: str) -> bool:
    """WikiExtractor writes section titles as a short line ending in '.'."""
    if not line.endswith(".") or len(line) > 100:
        return False
    w = line.split()
    return 0 < len(w) <= 10 and ". " not in line[:-1]


def clean_line(line: str) -> str:
    line = html.unescape(line)
    line = _PLACEHOLDER.sub("", line)
    line = _EMPTY_BRACKETS.sub("", line)
    line = line.replace("()", "").replace('""', "")
    line = _SPACES.sub(" ", line)
    line = _SPACE_BEFORE_PUNCT.sub(r"\1", line)
    return line.strip()


def clean_document(doc_id: str, url: str, title: str, body: str,
                   min_par_words: int = 8) -> Document:
    lines = body.split("\n")
    if lines and lines[0].strip() == title:
        lines = lines[1:]
    sections: list[tuple[str, list[str]]] = [("", [])]
    skipping = False
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if _is_heading(line):
            name = line[:-1].strip()
            skipping = name.lower() in SKIP_SECTIONS
            if not skipping:
                sections.append((name, []))
            continue
        if skipping:
            continue
        line = clean_line(line)
        if len(line.split()) < min_par_words:  # list items, captions, leftovers
            continue
        sections[-1][1].append(line)
    sections = [(s, ps) for s, ps in sections if ps]
    return Document(doc_id=doc_id, title=title, url=url, sections=sections)


# chunk
def chunk_document(doc: Document, target_words: int = 180, max_words: int = 300,
                   min_words: int = 40, overlap_sentences: int = 1) -> list[Chunk]:
    chunks: list[Chunk] = []
    if not 0 < min_words <= target_words <= max_words or overlap_sentences < 0:
        raise ValueError("Require 0 < min_words <= target_words <= max_words and nonnegative overlap")

    def emit(section: str, sents: list[str]) -> None:
        text = " ".join(sents).strip()
        if text:
            chunks.append(Chunk(chunk_id="", doc_id=doc.doc_id, title=doc.title, section=section,
                                url=doc.url, text=text, n_words=len(text.split())))

    for section, paragraphs in doc.sections:
        sents: list[str] = []
        for p in paragraphs:
            for s in split_sentences(p):
                # extremely long "sentences" (tables, lists) are hard-wrapped
                ws = s.split()
                while len(ws) > max_words:
                    sents.append(" ".join(ws[:max_words]))
                    ws = ws[max_words:]
                if ws:
                    sents.append(" ".join(ws))
        start_len = len(chunks)
        window: list[str] = []
        n = 0
        fresh = 0  # sentences in the window that are not yet in any chunk

        def restart() -> None:
            nonlocal window, n, fresh
            window = window[-overlap_sentences:] if overlap_sentences else []
            n = sum(len(x.split()) for x in window)
            fresh = 0

        for s in sents:
            sw = len(s.split())
            if fresh and n + sw > max_words:
                emit(section, window)
                restart()
            while window and not fresh and n + sw > max_words:
                n -= len(window.pop(0).split())
            window.append(s)
            n += sw
            fresh += 1
            if n >= target_words:
                emit(section, window)
                restart()
        if fresh:
            tail = " ".join(window[-fresh:])
            if (len(chunks) > start_len and len(tail.split()) < min_words
                    and chunks[-1].n_words + len(tail.split()) <= max_words):
                # merge a tiny remainder into the previous chunk of this section
                last = chunks[-1]
                last.text = f"{last.text} {tail}"
                last.n_words = len(last.text.split())
            else:
                emit(section, window)
    for i, c in enumerate(chunks):
        c.chunk_id = f"{doc.doc_id}-{i}"
    return chunks


# io
def save_jsonl(rows: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_chunks(path: Path) -> list[Chunk]:
    return [Chunk(**r) for r in load_jsonl(path)]


def chunk_to_dict(c: Chunk) -> dict:
    return asdict(c)
