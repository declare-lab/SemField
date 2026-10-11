"""Prepare one matched C4 subset for all external watermark baselines.

The PMark and AliMark releases ship the same 500 C4 examples in different
formats.  This script verifies that correspondence and emits: a neutral JSON
manifest, SimMark prompt/human Hugging Face datasets, and an AliMark JSONL
subset.  It never downloads data or models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: expected object")
                rows.append(value)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pmark-data", default="third_party/PMark/data/c4-val-500")
    parser.add_argument("--alimark-data", default="third_party/AliMark/dataset/c4.json")
    parser.add_argument("--documents", type=int, default=100)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from datasets import Dataset, load_from_disk

    pmark_path = Path(args.pmark_data).resolve()
    alimark_path = Path(args.alimark_data).resolve()
    out = Path(args.out_dir).resolve()
    if args.documents <= 0:
        parser.error("--documents must be positive")
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            parser.error(f"non-empty output exists: {out}; use --overwrite")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    pmark = load_from_disk(str(pmark_path))
    alimark = read_jsonl(alimark_path)
    count = min(args.documents, len(pmark), len(alimark))
    documents = []
    for index in range(count):
        prow = pmark[index]
        arow = alimark[index]
        prompt = str(prow.get("prompt", "")).strip()
        text = str(prow.get("text", "")).strip()
        aprompt = str(arow.get("prompt", "")).strip()
        reference = str(arow.get("natural_text", "")).strip()
        if not prompt and aprompt:
            # PMark's booksum-train-500 ships raw BookSum columns and carries
            # no "prompt" field, unlike their c4-val-500. The prompt is simply
            # the prefix of that row's text, and AliMark's matching split does
            # carry it, so take it from there. The text-prefix check below is
            # what actually proves the two rows are the same document, and it
            # still runs unchanged.
            prompt = aprompt
        if not prompt or not reference:
            raise ValueError(f"empty prompt/reference at row {index}")
        if prompt != aprompt:
            raise ValueError(f"PMark/AliMark prompt mismatch at row {index}")
        if not text.startswith(prompt):
            raise ValueError(f"PMark text does not start with prompt at row {index}")
        documents.append({
            "document_id": index,
            "prompt": prompt,
            "reference": reference,
            "source_text": text,
        })

    shared = {
        "schema_version": 1,
        "experiment": "exp26_prepare_baseline_inputs",
        "documents": documents,
        "source": {
            "pmark_dataset": str(pmark_path),
            "alimark_jsonl": str(alimark_path),
            "alimark_sha256": sha256(alimark_path),
        },
    }
    atomic_json(out / "shared.json", shared)
    Dataset.from_dict({"text": [d["prompt"] for d in documents]}).save_to_disk(
        str(out / "simmark-prompts")
    )
    Dataset.from_dict({"text": [d["reference"] for d in documents]}).save_to_disk(
        str(out / "simmark-human")
    )
    with (out / "alimark-c4.json").open("w", encoding="utf-8") as handle:
        for document in documents:
            json.dump({"prompt": document["prompt"],
                       "natural_text": document["reference"]},
                      handle, ensure_ascii=False)
            handle.write("\n")
    print(f"prepared {count} matched documents in {out}")


if __name__ == "__main__":
    main()
