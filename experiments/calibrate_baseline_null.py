#!/usr/bin/env python3
"""Score one baseline on the shared independent null corpus.

This is detector-only: it never generates a watermarked continuation and it
does not touch attack artifacts.  Scores are checkpointed by document so a
long detector pass can be resumed safely.
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
sys.path.insert(0, str(ROOT))
from experiments.exp34_native_baseline_robustness import Detector  # noqa: E402


def contract(args: argparse.Namespace, rows: list[dict]) -> dict:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["sha256"]).encode("ascii"))
        digest.update(b"\0")
    return {
        "method": args.method,
        "model": args.model,
        "embedder": args.embedder,
        "corpus_sha256": digest.hexdigest(),
        "documents": len(rows),
        "precision": args.precision,
    }


def threshold(scores: np.ndarray, alpha: float) -> tuple[float, int]:
    values = np.sort(np.asarray(scores, dtype=np.float64))
    allowed = max(0, int(math.floor(float(alpha) * len(values))))
    index = max(0, len(values) - allowed - 1)
    return float(np.nextafter(values[index], np.inf)), allowed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True,
                        choices=["kgw", "exp", "synthid", "sir", "semstamp",
                                 "k-semstamp", "simmark", "pmark-online",
                                 "pmark-offline", "alimark"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedder", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument(
        "--checkpoint-every", type=int, default=8,
        help=(
            "flush the resumable partial file after this many batches; "
            "larger values reduce filesystem sync overhead without changing "
            "scores (use 1 for maximum crash durability)"
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--vllm-python", default=sys.executable)
    parser.add_argument("--vllm-max-model-len", type=int, default=2048)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=32)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.70)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be positive")

    rows: list[dict] = []
    with args.corpus_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        raise RuntimeError("null corpus is empty")
    texts = [str(row["text"]) for row in rows]
    ids = [int(row["document_id"]) for row in rows]
    expected = contract(args, rows)
    out = args.out.resolve()
    partial = out.with_suffix(out.suffix + ".partial.jsonl")
    completed: dict[int, float] = {}
    if partial.exists():
        with partial.open(encoding="utf-8") as handle:
            header = json.loads(handle.readline())
            if header.get("contract") != expected:
                raise RuntimeError(f"partial contract mismatch: {partial}")
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    completed[int(row["document_id"])] = float(row["score"])
    pending = [i for i in range(len(rows)) if ids[i] not in completed]
    if not partial.exists():
        partial.parent.mkdir(parents=True, exist_ok=True)
        with partial.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"contract": expected}) + "\n")

    from types import SimpleNamespace
    detector_args = SimpleNamespace(
        method=args.method, model=args.model, embedder=args.embedder,
        precision=args.precision, device=args.device, seed=args.seed,
        vllm_python=args.vllm_python,
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_max_num_seqs=args.vllm_max_num_seqs,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        alimark_embedding_dim=768, out=str(out),
        pmark_num_samples=64, pmark_channels=4, pmark_soft_k=150.0,
        pmark_delta=0.001,
    )
    print(f"{args.method}: loading detector for {args.model} on {args.device}", flush=True)
    detector = Detector(detector_args, texts)
    # exp34 predates the detector-only calibration pass and only initializes
    # MarkLLM objects for the methods used by its attack CLI.  EXP and SynthID
    # use the same MarkLLM path, so initialize that path explicitly here.
    if args.method in {"exp", "synthid"} and not hasattr(detector, "mark"):
        detector._markllm()
    try:
        from tqdm.auto import tqdm
        progress = tqdm(total=len(rows), initial=len(completed),
                        desc=f"{args.method} null", unit="doc", dynamic_ncols=True)
        with partial.open("a", encoding="utf-8") as handle:
            batch_count = 0
            for start in range(0, len(pending), max(1, args.chunk_size)):
                batch_indices = pending[start:start + max(1, args.chunk_size)]
                batch = [texts[i] for i in batch_indices]
                values, _decisions = detector.score_many(batch)
                if len(values) != len(batch):
                    raise RuntimeError("detector returned the wrong number of scores")
                for index, value in zip(batch_indices, values):
                    score = float(value)
                    # MarkLLM's EXP detector reports the complementary
                    # statistic; the repository's unified scorer converts it
                    # to the common higher-is-more-watermarked direction.
                    if args.method == "exp":
                        score = 1.0 - score
                    if not math.isfinite(score):
                        raise RuntimeError(f"non-finite score at document {ids[index]}")
                    completed[ids[index]] = score
                    handle.write(json.dumps({"document_id": ids[index], "score": score}) + "\n")
                batch_count += 1
                if batch_count % args.checkpoint_every == 0 or (
                    start + len(batch_indices) >= len(pending)
                ):
                    handle.flush()
                    os.fsync(handle.fileno())
                progress.update(len(batch_indices))
        progress.close()
    finally:
        if getattr(detector, "llm", None) is not None:
            detector.llm.close()

    scores = np.asarray([completed[i] for i in ids], dtype=np.float64)
    calibration = scores[:sum(str(row.get("split")) == "calibration" for row in rows)]
    holdout = scores[len(calibration):]
    cutoff, allowed = threshold(calibration, args.alpha)
    output = {
        "format": "baseline-large-null-calibration/v1",
        "method": args.method,
        "model": args.model,
        "corpus": str(args.corpus_jsonl.resolve()),
        "contract": expected,
        "alpha": float(args.alpha),
        "threshold": cutoff,
        "calibration_documents": int(len(calibration)),
        "holdout_documents": int(len(holdout)),
        "calibration_fpr": float(np.mean(calibration >= cutoff)),
        "holdout_fpr": float(np.mean(holdout >= cutoff)) if len(holdout) else None,
        "scores": {"document_ids": ids, "calibration": calibration.tolist(),
                   "holdout": holdout.tolist()},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n")
    partial.unlink(missing_ok=True)
    print(json.dumps({"method": args.method, "threshold": cutoff,
                      "calibration_fpr": output["calibration_fpr"],
                      "holdout_fpr": output["holdout_fpr"], "output": str(out)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
