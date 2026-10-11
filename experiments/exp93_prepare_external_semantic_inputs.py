"""Prepare matched Hugging Face inputs for the official semantic baselines.

The released SAMark and SwordStamp programs consume ``datasets.Dataset``
objects saved with :func:`datasets.Dataset.save_to_disk`, while this benchmark
keeps the matched cohort in ``comparison-baselines/inputs/shared.json``.  This
adapter performs only format conversion; it never calls a model and never
changes the text.  The output contains one generation cohort and one
independent human-null cohort, both with a single ``text`` column so either
upstream implementation can consume them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


def _digest(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _read_documents(path: Path, limit: int) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("documents"), list):
        raise ValueError(f"{path} must contain an object with a documents list")
    rows = payload["documents"][:limit]
    if len(rows) < limit:
        raise ValueError(f"{path} contains {len(rows)} documents; need {limit}")

    result: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"documents[{index}] is not an object")
        prompt = row.get("prompt")
        source = row.get("source_text")
        reference = row.get("reference")
        if not all(isinstance(value, str) and value.strip()
                   for value in (prompt, source, reference)):
            raise ValueError(
                f"documents[{index}] needs non-empty prompt, source_text, and reference"
            )
        result.append({
            "document_id": int(row.get("document_id", index)),
            "prompt": prompt.strip(),
            "source_text": source.strip(),
            "reference": reference.strip(),
        })
    return result


def _save_dataset(path: Path, texts: list[str], ids: list[int]) -> None:
    from datasets import Dataset

    Dataset.from_dict({"text": texts, "document_id": ids}).save_to_disk(str(path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shared",
        default="runs/llama32-1b-500/comparison-baselines/inputs/shared.json",
        help="Matched-cohort manifest produced by exp26_prepare_baseline_inputs.py.",
    )
    parser.add_argument("--documents", type=int, default=500)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace only the explicitly named output directory.",
    )
    args = parser.parse_args()
    if args.documents <= 0:
        parser.error("--documents must be positive")

    shared = Path(args.shared).expanduser().resolve()
    out = Path(args.out_dir).expanduser().resolve()
    if not shared.is_file():
        raise FileNotFoundError(f"matched cohort not found: {shared}")
    if out.exists():
        if not args.overwrite and any(out.iterdir()):
            raise FileExistsError(
                f"non-empty output exists: {out}; use --overwrite explicitly"
            )
        if args.overwrite:
            shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    rows = _read_documents(shared, args.documents)
    ids = [int(row["document_id"]) for row in rows]
    generation = [str(row["source_text"]) for row in rows]
    human = [str(row["reference"]) for row in rows]
    _save_dataset(out / "generation", generation, ids)
    _save_dataset(out / "human", human, ids)

    manifest = {
        "schema_version": 1,
        "experiment": "exp93_prepare_external_semantic_inputs",
        "source_shared": str(shared),
        "documents": len(rows),
        "generation_dataset": str(out / "generation"),
        "human_null_dataset": str(out / "human"),
        "generation_text_sha256": _digest(generation),
        "human_text_sha256": _digest(human),
        "document_ids": ids,
        "semantics": {
            "generation_text": (
                "source_text (the upstream generator extracts its prompt from "
                "this full C4 row)"
            ),
            "human_null_text": "reference (independent matched human C4 text)",
            "prompt_rule": (
                "SAMark uses the first sentence; SwordStamp uses its official "
                "first-len_prompt-token sentence helper"
            ),
            "model_calls": 0,
        },
    }
    _atomic_json(out / "manifest.json", manifest)
    print(f"prepared {len(rows)} documents")
    print(f"generation: {out / 'generation'}")
    print(f"human null: {out / 'human'}")
    print(f"manifest:   {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
