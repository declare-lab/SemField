"""Build an official-detector HF dataset from normalized attack artifacts.

SAMark and SwordStamp expect a ``datasets.Dataset`` with clean watermarked
text in ``text`` and attacked text in ``para_text``.  The shared benchmark
stores those texts in separate normalized JSON files; this adapter only joins
them by document_id and performs no generation or text modification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _documents(path: Path, limit: int | None = None) -> dict[int, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("documents")
    if not isinstance(rows, list):
        raise ValueError(f"{path} has no documents list")
    if payload.get("status") != "complete":
        raise ValueError(f"{path} is not complete")
    if limit is not None:
        rows = rows[:limit]
    if limit is not None and len(rows) != limit:
        raise ValueError(f"{path} contains {len(rows)} documents; need {limit}")
    result: dict[int, dict] = {}
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: documents[{position}] is not an object")
        document_id = int(row.get("document_id", position))
        text = str(row.get("text", "")).strip()
        if not text:
            raise ValueError(f"{path}: document {document_id} has empty text")
        if document_id in result:
            raise ValueError(f"{path}: duplicate document_id {document_id}")
        result[document_id] = row
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--attack", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--documents", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    clean_path = args.clean.expanduser().resolve()
    attack_path = args.attack.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.documents <= 0:
        parser.error("--documents must be positive")
    for path in (clean_path, attack_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    # Load both sides before truncating.  This matters for a documented
    # partial attack cohort: the repaired Qwen Parrot/k-SwordStamp artifact
    # contains 499 IDs (0..499 except 358), while the clean file still has
    # all 500.  Selecting by the attack IDs preserves the pairing instead of
    # accidentally pairing clean ID 358 with attack ID 359.
    clean = _documents(clean_path)
    attack = _documents(attack_path)
    if len(attack) == args.documents and set(attack).issubset(clean):
        ids = sorted(attack)
    else:
        common = sorted(set(clean).intersection(attack))
        if len(common) < args.documents:
            raise ValueError(
                f"fewer than {args.documents} paired documents: "
                f"clean={len(clean)} attack={len(attack)} common={len(common)}"
            )
        ids = common[:args.documents]
    clean = {document_id: clean[document_id] for document_id in ids}
    attack = {document_id: attack[document_id] for document_id in ids}

    contract = {
        "schema_version": 1,
        "experiment": "exp95_prepare_external_attack_dataset",
        "clean": str(clean_path),
        "clean_sha256": _sha256(clean_path),
        "attack": str(attack_path),
        "attack_sha256": _sha256(attack_path),
        "documents": len(ids),
        "requested_documents": args.documents,
        "document_ids": ids,
    }
    manifest_path = output_dir / "attack_manifest.json"
    if args.resume and manifest_path.is_file():
        try:
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if old == contract and (output_dir / "dataset_info.json").is_file():
                print(f"complete attack dataset found: {output_dir}")
                return 0
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    if set(clean) != set(attack):
        missing_attack = sorted(set(clean) - set(attack))
        missing_clean = sorted(set(attack) - set(clean))
        raise ValueError(
            f"document_id mismatch: missing attack={missing_attack[:8]}, "
            f"missing clean={missing_clean[:8]}"
        )

    from datasets import Dataset

    ids = sorted(clean)
    dataset = Dataset.from_dict({
        "document_id": ids,
        "text": [str(clean[i]["text"]).strip() for i in ids],
        "para_text": [str(attack[i]["text"]).strip() for i in ids],
    })
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        dataset.save_to_disk(str(temporary))
        (temporary / "attack_manifest.json").write_text(
            json.dumps(contract, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            print(f"rebuilding derived attack dataset: {output_dir}")
            shutil.rmtree(output_dir)
        temporary.replace(output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"prepared {len(ids)} attack pairs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
