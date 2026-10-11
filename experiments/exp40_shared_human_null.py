"""Convert the shared manifest to AliMark's length-matched human null."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nltk.tokenize import sent_tokenize

from experiments.exp27_markllm_baseline import atomic_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--documents", type=int, default=500)
    parser.add_argument("--sentences", type=int, default=12)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.documents < 1:
        parser.error("--documents must be positive")
    if args.sentences < 1:
        parser.error("--sentences must be positive")
    source = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = []
    for item in source["documents"][:args.documents]:
        text = str(item["reference"]).strip()
        parts = [part.strip() for part in sent_tokenize(text) if part.strip()]
        if not parts:
            raise ValueError(f"empty reference for document {item['document_id']}")
        original_parts = list(parts)
        while len(parts) < args.sentences:
            parts.extend(original_parts)
        parts = parts[:args.sentences]
        text = " ".join(parts)
        rows.append({
            "document_id": int(item["document_id"]),
            "prompt": str(item["prompt"]),
            "text": text,
            "sentences": parts,
        })
    atomic_json(Path(args.out).resolve(), {
        "schema_version": 1,
        "experiment": "exp40_shared_human_null",
        "variant": "human-reference-exact-sentences",
        "sentences": args.sentences,
        "documents": rows,
    })
    print(f"wrote {len(rows)} untouched human references to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
