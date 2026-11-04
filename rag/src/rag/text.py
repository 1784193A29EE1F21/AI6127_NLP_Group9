"""Tokenizer (lowercase, stop-words, stemming) and sentence splitter."""
from __future__ import annotations

import re
import threading
import unicodedata

try:
    import snowballstemmer
except ImportError:  # stemming is optional
    snowballstemmer = None

_local = threading.local()  # Snowball stemmer is not thread-safe: one per thread


def _stemmer():
    st = getattr(_local, "stemmer", None)
    if st is None and snowballstemmer is not None:
        st = _local.stemmer = snowballstemmer.stemmer("english")
    return st

# Standard English stop-word list (Lucene/NLTK style, ~130 words)
STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been before
    being below between both but by can could did do does doing down during each few for from
    further had has have having he her here hers herself him himself his how i if in into is it
    its itself just me more most my myself no nor not now of off on once only or other our ours
    ourselves out over own same she should so some such than that the their theirs them
    themselves then there these they this those through to too under until up very was we were
    what when where which while who whom why will with would you your yours yourself yourselves
    also may might must shall us within without upon via""".split()
)

_WORD_RE = re.compile(r"[a-z0-9]+(?:['’][a-z]+)?")
_ANY_WORD_RE = re.compile(r"\w+", re.UNICODE)


def normalize(text: str) -> str:
    """NFKC-normalise, strip accents (café -> cafe) and lower-case."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower()


_CACHE: dict[str, str] = {}


def _stem(word: str) -> str:
    s = _CACHE.get(word)
    if s is None:
        st = _stemmer()
        s = st.stemWord(word) if st else word
        if len(_CACHE) < 1_000_000:
            _CACHE[word] = s
    return s


def tokenize(text: str, stem: bool = True, remove_stopwords: bool = True) -> list[str]:
    """Tokeniser for BM25: lowercase, accent-folded, stop-words removed, Snowball stemmed."""
    toks = _WORD_RE.findall(normalize(text))
    toks = [t.replace("’", "'").split("'")[0] for t in toks]  # drop possessive 's
    if remove_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    if stem:
        toks = [_stem(t) for t in toks]
    return [t for t in toks if t]


def words(text: str) -> list[str]:
    """Plain word tokens (for word / type counts in corpus statistics)."""
    return _ANY_WORD_RE.findall(text)


_SENT_RE = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"')\]]))\s+(?=[A-Z0-9\"'(\[])")
# abbreviations that should not end a sentence
_ABBREV = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof|St|Jr|Sr|vs|etc|e\.g|i\.e|c|ca|No|Vol|U\.S|Gen|Lt|Col|Inc|Ltd|Co)\.$")


def split_sentences(text: str) -> list[str]:
    parts = _SENT_RE.split(text)
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if out and (_ABBREV.search(out[-1]) or len(out[-1]) < 4):
            out[-1] = out[-1] + " " + p
        else:
            out.append(p)
    return out
