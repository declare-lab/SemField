"""Loading documents and splitting them into sentences.

The experiments need documents of human-written text, split into sentences,
from more than one domain. Domain matters: the whole point of the measurement
is that the design effect depends on how tightly a corpus concentrates in
embedding space, so a news corpus and a book corpus should give different
answers.

Three sources are supported, tried in this order:

  1. A local directory of .txt or .jsonl files, passed with --corpus-dir.
     This always works and needs no network.
  2. A Hugging Face dataset name, passed with --hf-dataset.
  3. A small built-in sample, so that the pipeline can be exercised end to end
     before committing to a download. The built-in sample is far too small for
     a real measurement and the code says so.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

__all__ = ["split_sentences", "load_documents", "regroup_sentences",
           "regroup_spans", "BUILTIN_SAMPLE"]


_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g",
    "i.e", "fig", "no", "vol", "pp", "al", "inc", "ltd", "co", "u.s", "u.k",
}

_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+")


def split_sentences(text: str, min_words: int = 4) -> list[str]:
    """Split text into sentences.

    Uses a regular expression with an abbreviation guard. If nltk is installed
    its Punkt tokeniser is used instead, which is more accurate. Sentence
    splitting only needs to be consistent between generation and detection;
    the method tolerates disagreement on individual boundaries, because a
    merge or split changes at most the cells of the sentences involved and
    never cascades to the rest of the document.
    """
    text = " ".join(text.split())
    if not text:
        return []

    try:                                      # prefer a real tokeniser
        import nltk
        from nltk.tokenize import sent_tokenize

        try:
            pieces = sent_tokenize(text)
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
            pieces = sent_tokenize(text)
    except Exception:
        pieces, buffer = [], ""
        for chunk in _BOUNDARY.split(text):
            buffer = (buffer + " " + chunk).strip()
            tail = buffer.rstrip(".!?\"')]").split()
            if tail and tail[-1].lower().strip(".") in _ABBREVIATIONS:
                continue
            pieces.append(buffer)
            buffer = ""
        if buffer:
            pieces.append(buffer)

    return [s for s in (p.strip() for p in pieces)
            if len(s.split()) >= min_words]


def _read_local(directory: Path) -> list[str]:
    docs: list[str] = []
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() == ".txt":
            docs.append(path.read_text(encoding="utf-8", errors="ignore"))
        elif path.suffix.lower() in {".jsonl", ".json"}:
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for field in ("text", "content", "document", "article", "body"):
                    if isinstance(obj.get(field), str):
                        docs.append(obj[field])
                        break
    return docs


def load_documents(corpus_dir: str | None = None,
                   hf_dataset: str | None = None,
                   hf_split: str = "train",
                   hf_field: str = "text",
                   max_documents: int = 200) -> tuple[list[str], str]:
    """Return (documents, description_of_source)."""
    if corpus_dir:
        directory = Path(corpus_dir).expanduser()
        if not directory.is_dir():
            raise FileNotFoundError(f"no such directory: {directory}")
        docs = _read_local(directory)
        if not docs:
            raise ValueError(f"no .txt or .jsonl files with text in {directory}")
        return docs[:max_documents], f"local directory {directory}"

    if hf_dataset:
        from datasets import load_dataset

        name, _, config = hf_dataset.partition(":")
        data = load_dataset(name, config or None, split=hf_split,
                            streaming=True)
        docs = []
        for row in data:
            value = row.get(hf_field)
            if isinstance(value, str) and len(value.split()) > 50:
                docs.append(value)
            if len(docs) >= max_documents:
                break
        if not docs:
            raise ValueError(f"no usable text in field {hf_field!r}")
        return docs, f"hugging face dataset {hf_dataset} [{hf_split}]"

    return list(BUILTIN_SAMPLE), "BUILT-IN SAMPLE (too small for a real measurement)"


BUILTIN_SAMPLE = [
    """The committee met on Tuesday to review the quarterly figures. Revenue
    had risen by four percent over the previous quarter. The chair noted that
    most of the increase came from the northern region. Costs were broadly
    flat. Two members asked whether the trend would continue into the winter.
    The finance officer said it was too early to tell. She promised a fuller
    breakdown at the next meeting. The committee agreed to reconvene in March.
    A short discussion followed about the hiring freeze. No decision was
    reached. The meeting closed at half past four.""",
    """The river runs shallow here for most of the year. In spring the melt
    comes down from the hills and the water rises over the stones. Children
    used to swim in the pool below the bridge. The bridge itself is older than
    the village records. Its stones were cut somewhere upstream and floated
    down. Nobody now remembers who built it. A plaque was fixed to the parapet
    in the last century. The lettering has since worn away. In summer the
    banks are thick with nettles. The path is difficult to follow.""",
]


def regroup_sentences(per_document, target_sentences: int,
                      min_sentences: int = 16):
    """Cut long documents into pieces of a realistic length.

    A detector is pointed at an article or a chapter, not at a whole book.
    booksum ships entire books, whose median length here is over three thousand
    sentences, and the design effect grows with length, so reporting a result
    at that length invites the objection that the document was chosen to
    flatter it. Regrouping to about fifty sentences puts booksum on the same
    footing as the news and scientific corpora.

    This takes the already-split sentence lists and regroups them, rather than
    re-splitting the text. The flat sentence sequence is therefore unchanged,
    which matters because the embedding cache is keyed on exactly that
    sequence: regrouping costs nothing, where re-splitting would force a
    corpus of hundreds of thousands of sentences to be encoded again.

    Pieces shorter than min_sentences are dropped from the analysis, but their
    sentences stay in the flat list, so the cache still holds.

    Because pieces are dropped while their sentences remain, the piece lengths
    no longer tile the flat sequence and a caller must NOT rebuild offsets from
    them. Use regroup_spans for that.
    """
    flat = [s for doc in per_document for s in doc]
    return [flat[a:b] for a, b in
            regroup_spans(per_document, target_sentences, min_sentences)]


def regroup_spans(per_document, target_sentences: int,
                  min_sentences: int = 16) -> list[tuple[int, int]]:
    """Where each regrouped piece sits in the flat sentence sequence.

    Returns (start, end) index pairs into the concatenation of per_document,
    which is the array the embeddings are computed over.

    This exists because the pieces do not tile that array. A tail piece shorter
    than min_sentences is dropped from the analysis but its sentences stay in
    the flat sequence, so cumulative piece lengths drift from true positions
    after the first drop. Slicing embeddings by those cumulative lengths hands
    a document another document's sentences: on twenty booksum books, thirty
    four dropped sentences were enough to misalign 1384 of 1532 pieces, and the
    error is silent because every shape still matches.
    """
    spans: list[tuple[int, int]] = []
    offset = 0
    for sentences in per_document:
        if target_sentences:
            for start in range(0, len(sentences), target_sentences):
                piece = sentences[start: start + target_sentences]
                if len(piece) >= min_sentences:
                    spans.append((offset + start,
                                  offset + start + len(piece)))
        else:
            spans.append((offset, offset + len(sentences)))
        offset += len(sentences)
    return spans
