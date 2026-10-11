"""Perplexity for every method in the 500-document benchmark, scored the same way.

Why this exists: exp23_semfield_quality.py only accepts SemField's own
generation files (it hard-filters on experiment == "exp19_semfield_generate"),
so it silently produces nothing for a baseline method's file. There is no
existing script that scores all 12 methods (9 baselines + 3 SemField
variants) under one judge model and one reference set. Building one instead
of forcing two incompatible pipelines together avoids exactly the kind of
mismatch -- different judge, different reference text, different document
subset -- that would make the numbers not directly comparable.

Every method is scored the same way here: perplexity of that method's own
watermarked/rewritten text, under the same judge model for everyone, against
the SAME human-reference document set (the exact file that fed the real
detection scores -- verify with its sha256 before trusting a copy, since at
least one method directory in this repo turned out to hold a stale,
differently-truncated duplicate).

No document is compared against its own method's "null" variant here. Some
methods (SemField) have one and some baselines don't, so pairing each method
against its own null would silently switch what's being measured method to
method. Comparing everyone against the one shared human reference is the
only choice that stays the same question for every row.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semfield.vllm_ppl import (  # noqa: E402
    load_vllm_judge, score_documents_vllm_alimark,
)


def label_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("use LABEL=path/to/generation.json")
    label, path = value.split("=", 1)
    return label, Path(path)


def load_texts(path: Path, max_documents: int | None) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    documents = list(payload.get("documents", ()))
    if max_documents is not None:
        documents = documents[:max_documents]
    return [row for row in documents if str(row.get("text", "")).strip()]


def relative_quality(method_ppl: dict, reference_ppl: dict) -> dict[str, object]:
    """Compact relative quality indicator against the human corpus."""
    return {
        "corpus_ppl_ratio": float(
            method_ppl["perplexity"] /
            max(1e-12, reference_ppl["perplexity"])
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", action="append", required=True, type=label_path,
                        help="LABEL=path, repeatable, one per method's watermarked text")
    parser.add_argument("--reference", required=True,
                        help="the one shared human-reference.json every method is compared against")
    parser.add_argument("--reference-sha256", default=None,
                        help="if given, refuse to run unless --reference matches this hash")
    parser.add_argument("--judge", required=True)
    parser.add_argument("--precision", choices=["auto", "fp16", "bf16", "fp32"],
                        default="bf16")
    parser.add_argument("--vllm-batch-size", type=int, default=64)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-length", type=int, default=2048,
                        help="AliMark per-document token cap")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    reference_path = Path(args.reference)
    if args.reference_sha256:
        import hashlib
        actual = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        if actual != args.reference_sha256:
            parser.error(
                f"--reference {reference_path} has sha256 {actual}, "
                f"expected {args.reference_sha256} -- refusing to run against "
                f"the wrong reference file"
            )
    reference_documents = load_texts(reference_path, args.max_documents)
    reference_texts = [str(row["text"]) for row in reference_documents]
    print(f"reference: {reference_path} ({len(reference_texts)} documents)")

    max_length = int(args.max_length)
    if max_length < 2:
        parser.error("--max-length must be at least 2")
    model, tokenizer, dtype = load_vllm_judge(
        args.judge, args.precision, max_length,
        args.vllm_gpu_memory_utilization,
    )

    print("scoring shared human reference")
    reference_ppl = score_documents_vllm_alimark(
        reference_documents, model, tokenizer, max_length, args.bootstrap,
        args.seed, args.vllm_batch_size,
    )

    results: dict[str, dict] = {}
    for index, (label, path) in enumerate(args.input):
        if label in results:
            parser.error(f"duplicate label {label!r}")
        documents = load_texts(path, args.max_documents)
        print(f"{label}: {path} ({len(documents)} documents)")
        ppl = score_documents_vllm_alimark(
            documents, model, tokenizer, max_length, args.bootstrap,
            args.seed + index, args.vllm_batch_size,
        )
        row: dict[str, object] = {
            "file": str(path),
            "documents": len(documents),
            "perplexity": ppl,
            "relative_to_reference": relative_quality(ppl, reference_ppl),
        }
        results[label] = row
        print(f"  PPL {row['perplexity']['perplexity']:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "experiment": "exp58_unified_quality",
        "judge": args.judge,
        "backend": "vllm_prompt_logprobs",
        "protocol": "alimark_single_pass_truncate_2048",
        "judge_dtype": dtype,
        "reference": str(reference_path),
        "reference_documents": len(reference_texts),
        "reference_perplexity": reference_ppl,
        "max_documents": args.max_documents,
        "bootstrap": args.bootstrap,
        "results": results,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
