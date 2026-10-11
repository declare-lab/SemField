#!/usr/bin/env python3
"""Calibrate an official SAMark/SwordStamp detector on an independent null corpus.

This is detector-only.  It never generates text and never consumes an existing
watermarked score.  The detector configuration is the same one used by the
official external-baseline runs; the output is a resumable empirical upper-tail
threshold at the requested false-positive rate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SWORD_ROOT = ROOT / "third_party" / "SwordStamp"
SAMARK_ROOT = ROOT / "third_party" / "SAMark"
SAMARK_RUNNER_ROOT = SWORD_ROOT / "external" / "SAMark"
# The external attack runner uses this official SAMark detector checkout.  It
# is a batched, score-only wrapper around the same detector math as the
# standalone checkout; using it here keeps the null threshold on the exact
# score scale already used by the attack artifacts.  Both checkouts expose a
# module named ``samark_detect`` with different signatures.
sys.path.insert(0, str(SWORD_ROOT))
sys.path.insert(0, str(SAMARK_ROOT))
sys.path.insert(0, str(SAMARK_RUNNER_ROOT))


def _read_rows(path: Path, limit: int | None) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict) or not str(row.get("text", "")).strip():
                    raise ValueError(f"invalid null row in {path}")
                rows.append(row)
                if limit is not None and len(rows) >= limit:
                    break
    if not rows:
        raise ValueError(f"empty null corpus: {path}")
    return rows


def _corpus_hash(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for row in rows:
        h.update(str(row.get("sha256", "")).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _threshold(values: list[float], alpha: float) -> tuple[float, int]:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    allowed = max(0, int(math.floor(alpha * len(ordered))))
    index = max(0, len(ordered) - allowed - 1)
    return float(np.nextafter(ordered[index], np.inf)), allowed


def _build_samark(args):
    from nltk.tokenize import sent_tokenize
    from sentence_transformers import SentenceTransformer
    from samark_detect import detect_sentence_level

    embedder = SentenceTransformer(args.samark_embedder, device=args.device)

    def score(text: str) -> float:
        sentences = sent_tokenize(text)[: args.max_sentences]
        result = detect_sentence_level(
            sentences,
            embedder,
            args.samark_msig,
            transform="tanh",
            transform_k=args.samark_transform_k,
        )
        return float(result["z_score"])

    return score, {
        "detector": "third_party/SwordStamp/external/SAMark/samark_detect.py::detect_sentence_level",
        "embedder": args.samark_embedder,
        "msig": args.samark_msig,
        "pivot_seed": 42,
        "transform": "tanh",
        "transform_k": args.samark_transform_k,
        "max_sentences": args.max_sentences,
    }


def _build_swordstamp(args):
    import os as _os
    _os.chdir(SWORD_ROOT)
    from config.loader import load_config
    from segmentation import Segmenter
    from watermarking.detect import setup_detector, truncate_human_null

    config_path = (
        SWORD_ROOT / "config" / "presets" /
        ("semspan_lsh_fixed_diverse_best-of-n.yaml"
         if args.method == "swordstamp"
         else "semspan_kmeans_fixed_diverse_best-of-n.yaml")
    )
    overrides = [
        f"generation.model={args.model}",
        f"generation.max_new_tokens={args.max_new_tokens}",
        f"generation.len_prompt={args.len_prompt}",
        "runtime.semcut_batch_size=512",
        f"detection.text_span={args.text_span}",
    ]
    cfg = load_config([str(config_path)], overrides=overrides, env_prefix=None)
    detect_fn, mode_desc, encoder = setup_detector(cfg)
    segmenter = Segmenter.from_config(
        cfg.segmentation,
        encoder=encoder,
        encoder_id=cfg.watermark.embedder,
        batch_size=cfg.runtime.semcut_batch_size,
    )

    # Match the official detector's human/null length normalization.
    def score_many(texts: list[str]) -> list[float]:
        normalized = truncate_human_null(texts, cfg, segmenter)
        if args.text_span == 'continuation':
            from segmentation import DEFAULT_TYPE, DEFAULT_BACKEND
            sentence_segmenter = Segmenter(DEFAULT_TYPE, DEFAULT_BACKEND)
            tails = []
            for text in normalized:
                units = sentence_segmenter.segment(text)
                if not units:
                    raise ValueError('Empty human-null document')
                prompt = units[0].display.strip()
                if not text.startswith(prompt):
                    raise ValueError('Cannot establish human-null prompt boundary')
                tails.append(text[len(prompt):].strip())
            normalized = tails
        return [float(detect_fn(segmenter.segment(text))) for text in normalized]

    return score_many, {
        "detector": "third_party/SwordStamp/watermarking/detect.py::setup_detector",
        "mode": mode_desc,
        "config": str(config_path.relative_to(ROOT)),
        "watermark": {
            "mode": cfg.watermark.mode,
            "sp_dim": cfg.watermark.sp_dim,
            "lambda": cfg.watermark.lmbd,
            "embedder": cfg.watermark.embedder,
            "cc_path": cfg.watermark.cc_path,
            "hash_key": cfg.watermark.hash_key,
        },
        "segmentation": cfg.segmentation.__dict__,
        "max_new_tokens": cfg.generation.max_new_tokens,
        "len_prompt": cfg.generation.len_prompt,
        "text_span": args.text_span,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("samark", "swordstamp", "k-swordstamp"), required=True)
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--max-sentences", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=205)
    parser.add_argument("--len-prompt", type=int, default=32)
    parser.add_argument('--text-span', choices=['full_text', 'continuation'], default='full_text')
    parser.add_argument("--samark-embedder", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--samark-msig", type=int, default=2)
    parser.add_argument("--samark-transform-k", type=float, default=30.0)
    args = parser.parse_args()
    rows = _read_rows(args.corpus_jsonl, args.limit)
    expected = {
        "method": args.method,
        "corpus": str(args.corpus_jsonl.resolve()),
        "documents": len(rows),
        "corpus_sha256": _corpus_hash(rows),
        "alpha": args.alpha,
    }
    if args.method != 'samark' and args.text_span != 'full_text':
        expected['text_span'] = args.text_span
        expected['model'] = args.model
        expected['max_new_tokens'] = args.max_new_tokens
        expected['len_prompt'] = args.len_prompt
    out = args.out.resolve()
    partial = out.with_suffix(out.suffix + ".partial.jsonl")
    done: dict[int, float] = {}
    if partial.exists():
        with partial.open(encoding="utf-8") as handle:
            header = json.loads(handle.readline())
            if header.get("contract") != expected:
                raise RuntimeError(f"partial contract mismatch: {partial}")
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    done[int(item["document_id"])] = float(item["score"])
    pending = [i for i, row in enumerate(rows) if int(row.get("document_id", i)) not in done]
    partial.parent.mkdir(parents=True, exist_ok=True)
    if not partial.exists():
        with partial.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"contract": expected}) + "\n")

    if args.method == "samark":
        score_one, detector_meta = _build_samark(args)
        score_batch = lambda texts: [score_one(text) for text in texts]
    else:
        score_batch, detector_meta = _build_swordstamp(args)

    from tqdm.auto import tqdm
    with partial.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), max(1, args.chunk_size)):
            indices = pending[start:start + max(1, args.chunk_size)]
            scores = score_batch([str(rows[i]["text"]) for i in indices])
            if len(scores) != len(indices):
                raise RuntimeError("detector returned the wrong number of scores")
            for i, score in zip(indices, scores):
                if not math.isfinite(float(score)):
                    raise RuntimeError(f"non-finite score at document {rows[i].get('document_id', i)}")
                doc_id = int(rows[i].get("document_id", i))
                done[doc_id] = float(score)
                handle.write(json.dumps({"document_id": doc_id, "score": float(score)}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            tqdm.write(f"{args.method}: {min(start + len(indices), len(pending))}/{len(pending)}")

    scores = [done[int(row.get("document_id", i))] for i, row in enumerate(rows)]
    calibration = [s for s, row in zip(scores, rows) if row.get("split") == "calibration"]
    holdout = [s for s, row in zip(scores, rows) if row.get("split") == "holdout"]
    if not calibration:
        raise RuntimeError("null corpus must contain split=calibration rows")
    cutoff, allowed = _threshold(calibration, args.alpha)
    result = {
        "format": "external-native-large-null-calibration/v1",
        "method": args.method,
        "model_for_length_matching": args.model,
        "corpus": str(args.corpus_jsonl.resolve()),
        "contract": expected,
        "detector": detector_meta,
        "alpha": args.alpha,
        "threshold": cutoff,
        "calibration_documents": len(calibration),
        "holdout_documents": len(holdout),
        "calibration_fpr": float(np.mean(np.asarray(calibration) >= cutoff)),
        "holdout_fpr": float(np.mean(np.asarray(holdout) >= cutoff)) if holdout else None,
        "scores": {
            "document_ids": [int(row.get("document_id", i)) for i, row in enumerate(rows)],
            "calibration": calibration,
            "holdout": holdout,
        },
    }
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    partial.unlink(missing_ok=True)
    print(json.dumps({k: result[k] for k in ("method", "threshold", "calibration_fpr", "holdout_fpr", "calibration_documents", "holdout_documents")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
