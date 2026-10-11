"""Strictly validate a SemField generation JSON and its feature sidecar.

This validator is deliberately independent of the generator.  It catches the
failure mode where all document IDs and score entries exist but one document
has fewer sentences, leaving the flat NPZ sidecar with fewer rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def fingerprint(documents: list[dict]) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for position, document in enumerate(documents):
        document_id = int(document.get("document_id", position))
        digest.update(document_id.to_bytes(8, "big", signed=True))
        sentences = document.get("sentences", ())
        digest.update(len(sentences).to_bytes(8, "big"))
        for sentence in sentences:
            digest.update(str(sentence).encode("utf-8", "ignore"))
            digest.update(b"\x00")
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-json", required=True)
    parser.add_argument("--features-npz", default=None)
    parser.add_argument("--sentences", type=int, default=None)
    parser.add_argument("--documents", type=int, default=None)
    parser.add_argument("--allow-short-ids", default="",
                        help="comma-separated document IDs allowed to have fewer sentences")
    args = parser.parse_args()

    generation = Path(args.generation_json).expanduser().resolve()
    payload = json.loads(generation.read_text(encoding="utf-8"))
    documents = payload.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("generation JSON has no documents list")
    target_documents = args.documents or len(documents)
    target_sentences = args.sentences or int(payload["config"]["sentences"])
    allow_short = {int(x) for x in args.allow_short_ids.split(",") if x.strip()}
    if len(documents) != target_documents:
        raise ValueError(f"documents={len(documents)}, expected {target_documents}")
    ids = [int(row["document_id"]) for row in documents]
    if ids != list(range(target_documents)):
        raise ValueError("document IDs are not exactly 0..documents-1")
    short = [ids[i] for i, row in enumerate(documents)
             if len(row.get("sentences", [])) != target_sentences
             or int(row.get("sentences_emitted", -1)) != target_sentences]
    short = [doc_id for doc_id in short if doc_id not in allow_short]
    if short:
        raise ValueError(f"documents with sentence count != {target_sentences}: {short}")
    if any(not str(row.get("text", "")).strip() for row in documents):
        raise ValueError("one or more generation documents have empty text")

    sidecar_name = payload["feature_sidecar"]["path"]
    sidecar = (Path(args.features_npz).expanduser().resolve()
               if args.features_npz else generation.parent / sidecar_name)
    with np.load(sidecar) as values:
        features = values["features"]
        offsets = values["offsets"]
        document_ids = values["document_ids"]
    expected_rows = sum(len(row.get("sentences", [])) for row in documents)
    if features.ndim != 2 or features.shape[0] != expected_rows:
        raise ValueError(f"feature shape {features.shape} does not match "
                         f"{expected_rows} sentence rows")
    if len(offsets) != target_documents + 1:
        raise ValueError("sidecar offsets do not cover all documents")
    if not np.array_equal(document_ids, np.arange(target_documents, dtype=np.int64)):
        raise ValueError("sidecar document IDs are not exactly 0..documents-1")
    widths = np.diff(offsets)
    expected_widths = np.asarray([len(row.get("sentences", [])) for row in documents])
    if not np.array_equal(widths, expected_widths):
        raise ValueError("sidecar sentence spans do not match generation documents")
    actual_fingerprint = fingerprint(documents)
    expected_fingerprint = payload["feature_sidecar"].get(
        "sentence_fingerprint_blake2b128")
    if expected_fingerprint and expected_fingerprint != actual_fingerprint:
        raise ValueError("generation sentence fingerprint does not match its documents")
    if payload["feature_sidecar"].get("shape") != [int(v) for v in features.shape]:
        raise ValueError("feature_sidecar.shape metadata does not match NPZ")
    print(f"VALID generation: {target_documents} docs x {target_sentences} sentences; "
          f"features={features.shape}, sidecar={sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
