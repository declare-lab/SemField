"""Normalize official SimMark, PMark, or AliMark outputs for shared attacks."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False,
                      allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def split_sentences(text: str) -> list[str]:
    from nltk.tokenize import sent_tokenize
    return [value.strip() for value in sent_tokenize(text) if value.strip()]


def document(document_id: int, prompt: str, text: str) -> dict:
    text = text.strip()
    return {"document_id": document_id, "prompt": prompt, "text": text,
            "sentences": split_sentences(text)}


def simmark(raw: Path, shared: dict) -> tuple[list[dict], list[dict]]:
    from datasets import load_from_disk
    wm_values = load_from_disk(str(raw))["text"]
    docs = shared["documents"][:len(wm_values)]
    wm = []
    for row, text in zip(docs, wm_values):
        prompt = str(row["prompt"])
        continuation = str(text).strip()
        if continuation.startswith(prompt.strip()):
            continuation = continuation[len(prompt.strip()):].lstrip()
        wm.append(document(int(row["document_id"]), prompt, continuation))
    null = [document(int(row["document_id"]), row["prompt"], row["reference"])
            for row in docs]
    return wm, null


def pmark(raw: Path, shared: dict) -> tuple[list[dict], list[dict]]:
    wm, null = [], []
    by_id = {int(row["document_id"]): row for row in shared["documents"]}
    paths = [path for path in raw.glob("*.json") if path.stem.isdigit()]
    for path in sorted(paths, key=lambda p: int(p.stem)):
        idx = int(path.stem)
        row = json.loads(path.read_text(encoding="utf-8"))
        source = by_id[idx]
        wm.append(document(idx, source["prompt"], row["generated_text"]))
        null.append(document(idx, source["prompt"], source["reference"]))
    return wm, null


def alimark(raw: Path, shared: dict) -> tuple[list[dict], list[dict]]:
    rows = json.loads(raw.read_text(encoding="utf-8"))
    wm, null = [], []
    shared_documents = shared["documents"]
    by_id = {int(item["document_id"]): item for item in shared_documents}
    for key in sorted(rows, key=lambda value: int(value)):
        idx = int(key)
        row = rows[key]
        if idx not in by_id:
            raise ValueError(f"AliMark row {idx} is not in the shared cohort")
        # The benchmark null is the shared human reference for every method.
        # AliMark's raw release also tries to generate an unwatermarked answer,
        # but that auxiliary generation can be empty (document 32 in the
        # Llama-3.2-1B run).  Using it here made the AliMark null different
        # from SimMark, PMark, KGW, SIR, and the report's human-reference
        # file.  The shared reference is the matched, nonempty null used by
        # the evaluation protocol, so use it as the single source of truth.
        prompt = str(by_id[idx]["prompt"])
        wm_text = str(row["watermarked_result"].get("text", "")).strip()
        null_text = str(by_id[idx]["reference"]).strip()
        if not wm_text:
            raise ValueError(f"AliMark watermarked row {idx} is empty")
        if not null_text:
            raise ValueError(f"shared human reference row {idx} is empty")
        wm.append(document(idx, prompt, wm_text))
        null.append(document(idx, prompt, null_text))
    return wm, null


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["simmark", "pmark", "alimark"],
                        required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--shared", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    shared = json.loads(Path(args.shared).read_text(encoding="utf-8"))
    wm, null = globals()[args.method](Path(args.raw), shared)
    if not wm or len(wm) != len(null):
        raise ValueError("normalization produced empty or unmatched splits")
    expected = len(shared.get("documents", ()))
    if len(wm) != expected:
        raise ValueError(
            f"normalization produced {len(wm)} documents; shared cohort has "
            f"{expected}"
        )
    if any(not str(row.get("text", "")).strip() for row in wm + null):
        raise ValueError("normalization produced an empty document")
    common = {"schema_version": 1, "experiment": "exp28_normalize_baseline",
              "method": args.method, "prompts": [row["prompt"] for row in wm]}
    out = Path(args.out_dir)
    atomic_json(out / "watermarked.json",
                {**common, "variant": "watermarked", "documents": wm})
    atomic_json(out / "unwatermarked.json",
                {**common, "variant": "none", "documents": null})
    print(f"normalized {len(wm)} {args.method} documents in {out}")


if __name__ == "__main__":
    main()
