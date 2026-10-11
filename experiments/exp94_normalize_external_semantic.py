"""Normalize an official SAMark/SwordStamp dataset to the SemField JSON schema.

The upstream generators save a Hugging Face dataset whose ``text`` column is
the prompt plus the generated continuation.  SemField's shared baseline files
store only the continuation in ``text`` and keep the prompt separately.  This
adapter performs that lossless boundary conversion and copies the locked
human reference; it never scores, paraphrases, or changes generated text.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _sentences(text: str) -> list[str]:
    from nltk.tokenize import sent_tokenize

    return [part.strip() for part in sent_tokenize(text) if part.strip()]


def _load_shared(path: Path, documents: int) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("documents") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) < documents:
        raise ValueError(f"{path} must contain at least {documents} documents")
    selected = rows[:documents]
    result: list[dict[str, object]] = []
    for index, row in enumerate(selected):
        if not isinstance(row, dict):
            raise ValueError(f"documents[{index}] is not an object")
        for field in ("document_id", "prompt", "reference", "source_text"):
            if field not in row or not str(row[field]).strip():
                raise ValueError(f"documents[{index}] has no usable {field}")
        result.append(row)
    return result


def _actual_prompt(
    method: str,
    row: dict[str, object],
    source: dict[str, object],
    len_prompt: int,
) -> str:
    if method == "samark":
        # The fair SAMark bridge records the exact sentence prompt in each log.
        prompt = row.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            return prompt.strip()
        return str(source["prompt"]).strip()

    # SwordStamp's official generator calls this same helper before it enters
    # the configured semspan sampler (the helper's defaults are sentence/NLTK).
    # Replaying it lets us strip the exact prompt even when a long source
    # sentence is truncated at the configured 32-token boundary.
    from watermarking.primitives import extract_prompt_from_text

    return extract_prompt_from_text(str(source["source_text"]), len_prompt).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("samark", "swordstamp", "k-swordstamp"), required=True)
    parser.add_argument("--raw", required=True, help="upstream HF dataset saved to disk")
    parser.add_argument("--shared", required=True, help="locked SemField shared.json cohort")
    parser.add_argument("--out-dir", required=True, help="method output directory")
    parser.add_argument("--documents", type=int, required=True)
    parser.add_argument("--len-prompt", type=int, default=32)
    args = parser.parse_args()
    if args.documents <= 0 or args.len_prompt <= 0:
        parser.error("--documents and --len-prompt must be positive")

    from datasets import load_from_disk

    shared_path = Path(args.shared).expanduser().resolve()
    raw_path = Path(args.raw).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    sources = _load_shared(shared_path, args.documents)
    dataset = load_from_disk(str(raw_path))
    if len(dataset) != len(sources):
        raise ValueError(
            f"{raw_path} has {len(dataset)} rows; shared cohort has {len(sources)}"
        )
    if "text" not in dataset.column_names:
        raise ValueError(f"{raw_path} has no text column")

    generated: list[dict[str, object]] = []
    human: list[dict[str, object]] = []
    actual_prompts: list[str] = []
    for index, source in enumerate(sources):
        full = str(dataset[index]["text"]).strip()
        if not full:
            raise ValueError(f"empty generated text at row {index}")
        prompt = _actual_prompt(args.method, dataset[index], source, args.len_prompt)
        if not prompt or not full.startswith(prompt):
            raise ValueError(
                f"generated row {index} does not start with the upstream prompt; "
                f"method={args.method!r}, prompt={prompt!r}, text={full[:120]!r}"
            )
        continuation = full[len(prompt):].strip()
        if not continuation:
            raise ValueError(f"empty generated continuation at row {index}")
        document_id = int(source["document_id"])
        shared_prompt = str(source["prompt"]).strip()
        generated.append({
            "document_id": document_id,
            "prompt": shared_prompt,
            "generation_prompt": prompt,
            "text": continuation,
            "sentences": _sentences(continuation),
        })
        reference = str(source["reference"]).strip()
        human.append({
            "document_id": document_id,
            "prompt": shared_prompt,
            "text": reference,
            "sentences": _sentences(reference),
        })
        actual_prompts.append(prompt)

    common = {
        "schema_version": 1,
        "experiment": "exp94_normalize_external_semantic",
        "method": args.method,
        "source": {
            "raw_dataset": str(raw_path),
            "shared_cohort": str(shared_path),
            "documents": len(generated),
            "len_prompt": args.len_prompt,
        },
    }
    _atomic_json(out_dir / "watermarked.json", {
        **common, "status": "complete", "variant": "watermarked",
        "documents": generated,
    })
    _atomic_json(out_dir / "unwatermarked.json", {
        **common, "status": "complete", "variant": "human-reference",
        "documents": human,
    })
    _atomic_json(out_dir / "normalization.json", {
        **common,
        "actual_generation_prompts": actual_prompts,
        "prompt_truncation_count": sum(
            a != str(s["prompt"]).strip()
            for a, s in zip(actual_prompts, sources)
        ),
    })
    print(f"normalized {len(generated)} {args.method} documents in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
