"""Score one clean/full-rewrite condition with a native baseline detector.

This is deliberately separate from ``exp34``: it implements AliMark's primary
protocol (attacked watermarked positives versus untouched human negatives),
caches the clean-negative scores, and restores PMark's prompt so secret rows
are not shifted during detection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.exp34_native_baseline_robustness import Detector, rows


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


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def texts(payload: dict[int, dict], ids: list[int]) -> list[str]:
    return [" ".join(str(value) for value in payload[i]["sentences"])
            for i in ids]


def prompts(payload: dict[int, dict], ids: list[int]) -> list[str]:
    return [str(payload[i].get("prompt", "")).strip() for i in ids]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def pmark_scores(detector: Detector, values: list[str], prefix: list[str],
                 ids: list[int], args: argparse.Namespace, split: str) -> list[float]:
    from nltk.tokenize import sent_tokenize
    journal = Path(args.out).resolve().with_suffix(
        Path(args.out).suffix + f".{split}.partial.jsonl")
    contract = {
        "method": args.method, "split": split, "ids": ids,
        "model": args.model, "embedder": args.embedder,
        "seed": args.seed, "num_samples": args.pmark_num_samples,
        "channels": args.pmark_channels, "soft_k": args.pmark_soft_k,
        "delta": args.pmark_delta, "watermarked": sha256(args.watermarked),
        "null": sha256(args.null), "rewrite": sha256(args.rewrite),
    }
    completed: dict[int, float] = {}
    if journal.exists():
        with journal.open("r", encoding="utf-8") as handle:
            header = json.loads(handle.readline())
            if header.get("contract") != contract:
                raise ValueError(f"PMark score journal contract differs: {journal}")
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    completed[int(row["document_id"])] = float(row["score"])
    else:
        journal.parent.mkdir(parents=True, exist_ok=True)
        with journal.open("w", encoding="utf-8") as handle:
            json.dump({"contract": contract}, handle, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    from tqdm.auto import tqdm
    progress = tqdm(total=len(values), initial=len(completed),
                    desc=f"{args.method} {split}", unit="doc",
                    dynamic_ncols=True)
    for text, prompt, document_id in zip(values, prefix, ids):
        if document_id in completed:
            continue
        seed_everything(args.seed + 1_000_003 * document_id +
                        (0 if split == "null" else 97))
        sequence = ([prompt] if prompt else []) + sent_tokenize(text)
        result = detector.detect_paragraph(
            sequence, num_samples=args.pmark_num_samples,
            embedder=detector.encoder, pivot="rand",
            median_method=detector.median,
            dedup=(detector.median != "hd"), msig=args.pmark_channels,
            llm=detector.llm)
        score = float(result["z_score"])
        if not math.isfinite(score):
            raise ValueError(f"non-finite PMark score for document {document_id}")
        completed[document_id] = score
        with journal.open("a", encoding="utf-8") as handle:
            json.dump({"document_id": document_id, "score": score}, handle,
                      ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        progress.update(1)
    progress.close()
    return [completed[document_id] for document_id in ids]


@torch.inference_mode()
def score(detector: Detector, values: list[str], prefix: list[str],
          ids: list[int], args: argparse.Namespace, split: str) -> list[float]:
    if args.method.startswith("pmark-"):
        return pmark_scores(detector, values, prefix, ids, args, split)
    scores, _ = detector.score_many(values)
    output = [float(value) for value in scores]
    if any(not math.isfinite(value) for value in output):
        raise ValueError(f"{args.method} produced a non-finite score")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True,
                        choices=["simmark", "pmark-online", "pmark-offline",
                                 "alimark", "kgw", "sir"])
    parser.add_argument("--watermarked", required=True,
                        help="clean watermarked generation (for IDs/prompts)")
    parser.add_argument("--null", required=True, help="untouched human/null JSON")
    parser.add_argument("--rewrite", required=True,
                        help="clean watermarked JSON or complete-rewrite JSON")
    parser.add_argument("--condition", required=True,
                        help="clean, pegasus, parrot, dipper, or gpt35")
    parser.add_argument("--null-cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="facebook/opt-1.3b")
    parser.add_argument("--embedder",
                        default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--alimark-embedding-dim", type=int, default=768,
                        help="AliMark sentence-vector width; must match its key")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--pmark-num-samples", type=int, default=64)
    parser.add_argument("--pmark-channels", type=int, default=4)
    parser.add_argument("--pmark-soft-k", type=float, default=150.0)
    parser.add_argument("--pmark-delta", type=float, default=0.001)
    parser.add_argument("--vllm-python",
                        default=sys.executable)
    parser.add_argument("--vllm-max-model-len", type=int, default=2048)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=192)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float,
                        default=0.80)
    args = parser.parse_args()

    clean_wm, null, rewrite = rows(args.watermarked), rows(args.null), rows(args.rewrite)
    ids = sorted(set(clean_wm) & set(null) & set(rewrite))
    if args.max_documents is not None:
        if args.max_documents < 1: parser.error("--max-documents must be positive")
        ids = ids[:args.max_documents]
    expected = sorted(set(clean_wm) & set(null))[:args.max_documents]
    if not ids or ids != expected:
        missing = sorted(set(expected).difference(ids))
        parser.error(f"rewrite does not cover the paired cohort; missing={missing[:10]}")

    null_values, null_prompts = texts(null, ids), prompts(null, ids)
    detector = Detector(args, null_values)
    cache_path = Path(args.null_cache).resolve()
    cache_contract = {
        "method": args.method, "null_sha256": sha256(args.null),
        "document_ids": ids, "model": args.model, "embedder": args.embedder,
        "pmark": {"num_samples": args.pmark_num_samples,
                  "channels": args.pmark_channels,
                  "soft_k": args.pmark_soft_k, "delta": args.pmark_delta,
                  "prompt_restored": True},
    }
    cached = None
    if cache_path.exists():
        candidate = json.loads(cache_path.read_text(encoding="utf-8"))
        candidate_scores = candidate.get("scores", [])
        if (candidate.get("contract") == cache_contract
                and len(candidate_scores) == len(ids)):
            cached = candidate
        else:
            print(
                f"stale null cache; recomputing and replacing {cache_path}",
                flush=True,
            )
    if cached is not None:
        null_scores = [float(value) for value in cached["scores"]]
        print(f"reused {len(null_scores)} clean-null scores from {cache_path}")
    else:
        null_scores = score(detector, null_values, null_prompts, ids, args, "null")
        atomic_json(cache_path, {"schema_version": 1,
                    "experiment": "exp36_native_full_rewrite_null_cache",
                    "contract": cache_contract, "scores": null_scores})
        Path(args.out).resolve().with_suffix(
            Path(args.out).suffix + ".null.partial.jsonl").unlink(missing_ok=True)

    positive_scores = score(detector, texts(rewrite, ids),
                            prompts(clean_wm, ids), ids, args, "watermarked")
    output = {
        "schema_version": 1, "experiment": "exp36_native_full_rewrite",
        "method": args.method, "condition": args.condition,
        "protocol": {
            "name": "AliMark-positive-only-full-rewrite",
            "positive": "complete rewrite of watermarked generation",
            "negative": "untouched human text",
            "partial_rewrite": False,
            "pmark_prompt_restored": args.method.startswith("pmark-"),
        },
        "config": vars(args),
        "sources": {"watermarked_sha256": sha256(args.watermarked),
                    "null_sha256": sha256(args.null),
                    "rewrite_sha256": sha256(args.rewrite)},
        "scores": {"document_ids": ids, "watermarked": positive_scores,
                   "null": null_scores},
    }
    atomic_json(Path(args.out).resolve(), output)
    Path(args.out).resolve().with_suffix(
        Path(args.out).suffix + ".watermarked.partial.jsonl").unlink(
            missing_ok=True)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
