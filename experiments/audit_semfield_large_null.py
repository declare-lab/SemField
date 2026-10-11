#!/usr/bin/env python3
"""Audit the shared-field SemField threshold with a large independent null.

This is deliberately a detector-only audit.  It never regenerates text and it
does not alter any existing score files.  C4 documents are streamed, exact
overlaps with the evaluation manifests are excluded, and a calibration/holdout
split is kept so that the reported false-positive rate is not evaluated on the
same null examples used to choose the threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semfield.corpora import split_sentences  # noqa: E402
from semfield.embedders import embed, load_embedder  # noqa: E402
from semfield.semfield import (  # noqa: E402
    benchmark_ensemble_key,
    field_statistic,
    keyed_gaussian,
    two_sided_field_statistic,
)
from semfield.semfield_kernels import build_feature_map  # noqa: E402


def _normalise(text: str) -> str:
    return " ".join(str(text).lower().split())


def _digest(text: str) -> str:
    return hashlib.sha256(_normalise(text).encode("utf-8", "ignore")).hexdigest()


def _manifest_hashes(paths: list[Path]) -> set[str]:
    hashes: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        for row in payload.get("documents", []):
            for key in ("source_text", "reference", "prompt", "text"):
                value = row.get(key)
                if isinstance(value, str) and value.strip():
                    hashes.add(_digest(value))
    return hashes


def _threshold(scores: np.ndarray, alpha: float) -> tuple[float, int]:
    """Return a strict upper-tail threshold and its allowed count.

    The next representable float above the order statistic makes the rule
    ``score >= threshold`` conservative even if scores tie at the cutoff.
    """
    values = np.sort(np.asarray(scores, dtype=np.float64))
    n = int(values.size)
    allowed = max(0, int(math.floor(float(alpha) * n)))
    index = max(0, n - allowed - 1)
    threshold = float(np.nextafter(values[index], np.inf))
    return threshold, allowed


def _bootstrap_quantile(scores: np.ndarray, q: float, reps: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(scores, dtype=np.float64)
    out = np.empty(int(reps), dtype=np.float64)
    # 5000 x 1000 is small, but batching avoids a large temporary array.
    for start in range(0, int(reps), 100):
        count = min(100, int(reps) - start)
        indices = rng.integers(0, values.size, size=(count, values.size))
        out[start:start + count] = np.quantile(values[indices], q, axis=1)
    return [float(x) for x in np.quantile(out, [0.025, 0.5, 0.975])]


def _positive_scores(score_dir: Path, score_prefix: str) -> dict[str, np.ndarray]:
    names = {
        "clean": "semfield-pl-shared-field__clean.json",
        "pegasus": "semfield-pl-shared-field__pegasus.json",
        "parrot": "semfield-pl-shared-field__parrot.json",
        "dipper": "semfield-pl-shared-field__dipper.json",
        "llm-rewrite": "semfield-pl-shared-field__llm-rewrite.json",
    }
    output: dict[str, np.ndarray] = {}
    for label, filename in names.items():
        path = score_dir / filename.replace("semfield-pl-shared-field", score_prefix)
        payload = json.loads(path.read_text())
        values = np.asarray(payload["scores"]["watermarked"], dtype=np.float64)
        if values.size != 500:
            raise ValueError(f"expected 500 positive scores in {path}, got {values.size}")
        output[label] = values
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-docs", type=int, default=5000)
    parser.add_argument("--holdout-docs", type=int, default=1000)
    parser.add_argument("--sentences", type=int, default=12)
    parser.add_argument("--min-words", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--dataset", default="allenai/c4")
    parser.add_argument("--config", default="en")
    parser.add_argument("--split", default="train")
    parser.add_argument("--embedder", default="mpnet")
    parser.add_argument(
        "--feature-map", default="linear",
        help="feature map used by the detector (e.g. linear or cone-equalized)",
    )
    parser.add_argument("--cone-equalizer-path", default=None)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--documents-jsonl", type=Path, default=None,
                        help="also persist the exact null corpus for other detectors")
    parser.add_argument("--scores-dir", type=Path, default=Path(
        "runs/llama32-1b-500/semfield-pl64-shared-field-v2/scores"))
    parser.add_argument("--null-npz", type=Path, default=None)
    parser.add_argument(
        "--tail", choices=("two-sided", "one-sided"), default="two-sided",
        help="null/detection tail; PL uses two-sided, no-PL uses one-sided",
    )
    parser.add_argument(
        "--score-prefix", default="semfield-pl-shared-field",
        help="prefix of saved positive score bundles",
    )
    parser.add_argument(
        "--reuse-documents-jsonl", type=Path, default=None,
        help="reuse an existing exact null-document JSONL instead of streaming C4",
    )
    args = parser.parse_args()
    if not (0.0 < args.alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    manifests = [
        ROOT / "runs/llama32-1b-500/comparison-baselines/inputs/shared.json",
        ROOT / "runs/qwen3-1.7b-500/comparison-baselines/inputs/shared.json",
        ROOT / "runs/opt-1.3b-500/comparison-baselines-hf/inputs/shared.json",
    ]
    excluded = _manifest_hashes(manifests)
    wanted = int(args.calibration_docs) + int(args.holdout_docs)
    docs: list[list[str]] = []
    hashes: list[str] = []
    urls: list[str] = []
    if args.reuse_documents_jsonl is not None:
        source = Path(args.reuse_documents_jsonl)
        if not source.is_file():
            raise FileNotFoundError(source)
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                if len(docs) >= wanted:
                    break
                row = json.loads(line)
                sentences = row.get("sentences")
                if not isinstance(sentences, list) or len(sentences) < int(args.sentences):
                    raise ValueError(f"invalid sentence list in {source}")
                docs.append([str(s).strip() for s in sentences[:int(args.sentences)]])
                hashes.append(str(row.get("sha256", _digest(" ".join(sentences)))))
                urls.append(str(row.get("url", "")))
        if len(docs) != wanted:
            raise RuntimeError(f"{source} contains {len(docs)} usable documents; needed {wanted}")
        print(f"reusing {len(docs)} null documents from {source}", flush=True)
    else:
        seen: set[str] = set(excluded)
        from datasets import load_dataset

        print(f"streaming {wanted} independent null documents from {args.dataset}/{args.config}", flush=True)
        stream = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
        for row in stream:
            raw = str(row.get("text", "")).strip()
            if not raw:
                continue
            digest = _digest(raw)
            if digest in seen:
                continue
            sentences = split_sentences(raw, min_words=int(args.min_words))
            if len(sentences) < int(args.sentences):
                continue
            seen.add(digest)
            docs.append([str(s).strip() for s in sentences[:int(args.sentences)]])
            hashes.append(digest)
            urls.append(str(row.get("url", "")))
            if len(docs) % 250 == 0:
                print(f"  collected {len(docs)}/{wanted}", flush=True)
            if len(docs) >= wanted:
                break
        if len(docs) < wanted:
            raise RuntimeError(f"only collected {len(docs)} usable documents; needed {wanted}")

    if args.documents_jsonl is None:
        args.documents_jsonl = args.output.with_suffix(".documents.jsonl")
    args.documents_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.documents_jsonl.open("w", encoding="utf-8") as handle:
        for index, (sentences, digest, url) in enumerate(zip(docs, hashes, urls)):
            handle.write(json.dumps({
                "document_id": index,
                "text": " ".join(sentences),
                "sentences": sentences,
                "sha256": digest,
                "url": url,
                "split": "calibration" if index < int(args.calibration_docs) else "holdout",
            }, ensure_ascii=False) + "\n")

    flat = [sentence for document in docs for sentence in document]
    print(f"encoding {len(flat)} sentences on {args.device}", flush=True)
    model = load_embedder(args.embedder, device=args.device, precision="fp16")
    embeddings = embed(
        model, flat, batch_size=int(args.batch_size), normalize=True,
        devices=[args.device], model_name=args.embedder,
        precision="fp16", show_progress_bar=True,
    )
    feature_map = build_feature_map(
        "linear",
        input_dim=int(embeddings.shape[1]),
        feature_map=args.feature_map,
        cone_equalizer_path=args.cone_equalizer_path,
    )
    key = benchmark_ensemble_key(0, 0)
    weights = keyed_gaussian(key, feature_map.n_features, channel=0)

    scores = np.empty(len(docs), dtype=np.float64)
    two_sided = args.tail == "two-sided"
    offset = 0
    for index in range(len(docs)):
        rows = embeddings[offset:offset + int(args.sentences)]
        offset += int(args.sentences)
        scorer = two_sided_field_statistic if two_sided else field_statistic
        scores[index] = scorer(feature_map.transform(rows), weights).z_score
    calibration = scores[:int(args.calibration_docs)]
    holdout = scores[int(args.calibration_docs):]
    threshold, allowed = _threshold(calibration, args.alpha)
    calibration_fpr = float(np.mean(calibration >= threshold))
    holdout_fpr = float(np.mean(holdout >= threshold))

    positives = _positive_scores(args.scores_dir, args.score_prefix)
    positive_summary: dict[str, dict[str, float]] = {}
    for label, values in positives.items():
        positive_summary[label] = {
            "n": int(values.size),
            "mean_score": float(np.mean(values)),
            "median_score": float(np.median(values)),
            "tpr_at_alpha": float(np.mean(values >= threshold)),
            "tpr_at_alpha_percent": float(100.0 * np.mean(values >= threshold)),
        }
    attack_labels = ["pegasus", "parrot", "dipper", "llm-rewrite"]
    mean_tpr = float(np.mean([positive_summary[x]["tpr_at_alpha"] for x in attack_labels]))
    worst_tpr = float(min(positive_summary[x]["tpr_at_alpha"] for x in attack_labels))

    if args.null_npz is None:
        args.null_npz = args.output.with_suffix(".null_scores.npz")
    args.null_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.null_npz,
        calibration=calibration,
        holdout=holdout,
        all_scores=scores,
    )
    result = {
        "format": "semfield-large-null-audit/v1",
        "field": "SemField shared key-derived field",
        "tail": args.tail,
        "score_prefix": args.score_prefix,
        "key_derivation": "benchmark_ensemble_key(seed=0,index=0)",
        "feature_map": feature_map.metadata(),
        "encoder": "sentence-transformers/all-mpnet-base-v2",
        "dataset": {"name": args.dataset, "config": args.config, "split": args.split},
        "collection": {
            "calibration_documents": int(args.calibration_docs),
            "holdout_documents": int(args.holdout_docs),
            "sentences_per_document": int(args.sentences),
            "excluded_manifest_hashes": len(excluded),
            "excluded_manifests": [str(x) for x in manifests if x.exists()],
            "document_hashes": hashes,
            "urls": urls,
        },
        "threshold": {
            "alpha": float(args.alpha),
            "value": threshold,
            "calibration_allowed_count": int(allowed),
            "calibration_fpr": calibration_fpr,
            "holdout_fpr": holdout_fpr,
            "calibration_q99": float(np.quantile(calibration, 0.99)),
            "holdout_q99": float(np.quantile(holdout, 0.99)),
            "q99_bootstrap_95pct": _bootstrap_quantile(calibration, 0.99, 1000, args.seed),
        },
        "null_summary": {
            "calibration_mean": float(np.mean(calibration)),
            "calibration_median": float(np.median(calibration)),
            "calibration_max": float(np.max(calibration)),
            "holdout_mean": float(np.mean(holdout)),
            "holdout_median": float(np.median(holdout)),
            "holdout_max": float(np.max(holdout)),
            "scores_npz": str(args.null_npz),
            "documents_jsonl": str(args.documents_jsonl),
        },
        "existing_positive_scores": positive_summary,
        "existing_attack_aggregate": {
            "mean_tpr_at_alpha_percent": 100.0 * mean_tpr,
            "worst_tpr_at_alpha_percent": 100.0 * worst_tpr,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "threshold": threshold,
        "calibration_fpr_percent": 100.0 * calibration_fpr,
        "holdout_fpr_percent": 100.0 * holdout_fpr,
        "mean_attack_tpr_percent": 100.0 * mean_tpr,
        "worst_attack_tpr_percent": 100.0 * worst_tpr,
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
