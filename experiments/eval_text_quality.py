"""Evaluate text distributions with MAUVE.

Generation documents are matched exactly by ``document_id`` against a
reference JSON. The metrics use each document's ``text`` field; prompts are
only checked for pairing and are excluded from metric text. MAUVE measures
distributional similarity to
the supplied reference. Neither metric is a direct human quality judgment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_documents(path: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    documents = payload.get("documents") if isinstance(payload, dict) else None
    if not isinstance(documents, list) or not documents:
        raise ValueError(f"{path}: expected a non-empty 'documents' list")
    rows: dict[int, dict[str, Any]] = {}
    for position, row in enumerate(documents):
        if not isinstance(row, dict) or "document_id" not in row:
            raise ValueError(f"{path}: document {position} lacks document_id")
        document_id = row["document_id"]
        if isinstance(document_id, bool) or not isinstance(document_id, int):
            raise ValueError(
                f"{path}: document {position} document_id must be an integer"
            )
        if document_id in rows:
            raise ValueError(f"{path}: duplicate document_id {document_id}")
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}: empty or missing text for ID {document_id}")
        prompt = row.get("prompt")
        if prompt is not None and not isinstance(prompt, str):
            raise ValueError(f"{path}: prompt for ID {document_id} must be a string")
        rows[document_id] = {"text": text, "prompt": prompt}
    return rows


def validate_pair(
    name: str,
    path: Path,
    generated: dict[int, dict[str, Any]],
    reference: dict[int, dict[str, Any]],
) -> None:
    gen_ids, ref_ids = set(generated), set(reference)
    if gen_ids != ref_ids:
        missing = sorted(ref_ids - gen_ids)
        extra = sorted(gen_ids - ref_ids)
        raise ValueError(
            f"{name} ({path}) document IDs do not exactly match reference; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    for document_id in sorted(ref_ids):
        generated_prompt = generated[document_id]["prompt"]
        reference_prompt = reference[document_id]["prompt"]
        if (
            generated_prompt is not None
            and reference_prompt is not None
            and generated_prompt != reference_prompt
        ):
            raise ValueError(
                f"{name} ({path}) prompt mismatch for document_id {document_id}"
            )




def mauve(
    generated: list[str],
    reference: list[str],
    *,
    model: str,
    device_id: int,
    batch_size: int,
    max_text_length: int,
    num_buckets: str | int,
    seeds: list[int],
) -> dict[str, Any]:
    try:
        import mauve as mauve_package
    except ImportError as exc:
        raise RuntimeError(
            "MAUVE requires the optional quality dependencies; install with "
            "`uv pip install -e '.[quality]'`"
        ) from exc

    values = []
    for seed in seeds:
        result = mauve_package.compute_mauve(
            p_text=generated,
            q_text=reference,
            featurize_model_name=model,
            device_id=device_id,
            max_text_length=max_text_length,
            batch_size=batch_size,
            num_buckets=num_buckets,
            seed=seed,
            verbose=True,
        )
        values.append({"seed": seed, "mauve": float(result.mauve)})

    scores = [row["mauve"] for row in values]
    mean = sum(scores) / len(scores)
    sample_std = (
        math.sqrt(sum((score - mean) ** 2 for score in scores) / (len(scores) - 1))
        if len(scores) > 1 else None
    )
    return {
        "p_distribution": "generated text",
        "q_distribution": "matched reference text",
        "interpretation": "higher indicates greater distributional similarity to this reference",
        "feature_model": model,
        "device_id": device_id,
        "batch_size": batch_size,
        "max_text_length_tokens": max_text_length,
        "num_buckets": num_buckets,
        "scores_by_seed": values,
        "mean_over_seeds": mean,
        "std_over_seeds": sample_std,
    }


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("use NAME=path/to/generation.json")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("input name and path must be non-empty")
    return name.strip(), Path(raw_path).expanduser().resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", action="append", required=True, type=parse_input,
        metavar="NAME=JSON", help="generation JSON; repeat once per condition",
    )
    parser.add_argument("--reference", required=True, type=Path,
                        help="reference JSON with matching document IDs")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[25, 26, 27])
    parser.add_argument("--device-id", type=int, default=-1,
                        help="MAUVE feature device; -1 selects CPU")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-text-length", type=int, default=1024)
    parser.add_argument("--mauve-model", default="gpt2-large")
    parser.add_argument("--num-buckets", default="auto",
                        help="MAUVE buckets: 'auto' or integer > 1")
    args = parser.parse_args(argv)

    if args.batch_size <= 0 or args.max_text_length <= 0:
        parser.error("--batch-size and --max-text-length must be positive")
    if not args.seeds:
        parser.error("provide at least one --seeds value")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds values must be unique")
    try:
        num_buckets: str | int = (
            "auto" if args.num_buckets == "auto" else int(args.num_buckets)
        )
    except ValueError:
        parser.error("--num-buckets must be 'auto' or an integer")
    if isinstance(num_buckets, int) and num_buckets <= 1:
        parser.error("--num-buckets must be greater than 1")

    names = [name for name, _ in args.input]
    if len(set(names)) != len(names):
        parser.error("input names must be unique")
    reference_path = args.reference.expanduser().resolve()
    reference = load_documents(reference_path)
    reference_ids = sorted(reference)
    reference_texts = [reference[index]["text"] for index in reference_ids]
    inputs = []
    results: dict[str, Any] = {}

    for name, path in args.input:
        generated = load_documents(path)
        validate_pair(name, path, generated, reference)
        generated_texts = [generated[index]["text"] for index in reference_ids]
        row: dict[str, Any] = {"documents": len(generated_texts)}
        row["mauve"] = mauve(
            generated_texts,
            reference_texts,
            model=args.mauve_model,
            device_id=args.device_id,
            batch_size=args.batch_size,
            max_text_length=args.max_text_length,
            num_buckets=num_buckets,
            seeds=args.seeds,
        )
        results[name] = row
        inputs.append({
            "name": name,
            "path": str(path),
            "sha256": sha256(path),
            "document_ids": reference_ids,
        })

    output = {
        "schema_version": 1,
        "protocol": "paired_text_quality_mauve_v1",
        "reference": {
            "path": str(reference_path),
            "sha256": sha256(reference_path),
            "document_ids": reference_ids,
        },
        "settings": {
            "metrics": "mauve",
            "seeds": args.seeds,
            "device_id": args.device_id,
            "batch_size": args.batch_size,
            "max_text_length": args.max_text_length,
            "mauve_model": args.mauve_model,
            "num_buckets": num_buckets,
        },
        "notes": {
            "scope": "text fields only; prompts checked for pairing when present and excluded from metrics",
            "mauve_interpretation": "distributional similarity to the supplied reference; not a direct quality judgment",
        },
        "inputs": inputs,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".tmp")
    temporary.write_text(
        json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
