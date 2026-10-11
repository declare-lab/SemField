"""Score MarkLLM SemStamp, k-SemStamp, or SynthID under full rewrites."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import semfield._hf_env  # noqa: F401,E402
from semfield.generation import default_precision_for_model  # noqa: E402
from experiments.exp27_markllm_baseline import atomic_json, preflight
from experiments.exp34_native_baseline_robustness import rows
from experiments.exp36_native_full_rewrite import sha256, texts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=["SEMSTAMP", "KSEMSTAMP", "SynthID", "EXP"],
                        required=True)
    parser.add_argument("--watermarked", required=True)
    parser.add_argument("--null", required=True)
    parser.add_argument("--rewrite", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--null-cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--markllm-dir", default="third_party/MarkLLM")
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL", "facebook/opt-1.3b"),
        help="causal LM checkpoint (defaults to $MODEL or OPT-1.3B)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"],
                        default=None)
    parser.add_argument("--max-documents", type=int, default=None)
    args = parser.parse_args()
    if args.precision is None:
        args.precision = default_precision_for_model(args.model)

    watermarked_payload = json.loads(
        Path(args.watermarked).read_text(encoding="utf-8")
    )
    clean, null, rewrite = rows(args.watermarked), rows(args.null), rows(args.rewrite)
    paired = sorted(set(clean) & set(null))
    if args.max_documents is not None:
        if args.max_documents < 1:
            parser.error("--max-documents must be positive")
        paired = paired[:args.max_documents]
    ids = [value for value in paired if value in rewrite]
    if not ids or ids != paired:
        parser.error(f"rewrite is incomplete; missing={sorted(set(paired)-set(ids))[:10]}")

    markllm = Path(args.markllm_dir).resolve()
    preflight(markllm, args.algorithm)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
             "fp32": torch.float32}[args.precision]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=dtype).to(args.device).eval()
    from semfield.liger_inference import apply_liger_to_model
    apply_liger_to_model(model)

    old_cwd = Path.cwd()
    sys.path.insert(0, str(markllm))
    os.chdir(markllm)
    try:
        from watermark.auto_watermark import AutoWatermark
        from utils.transformers_config import TransformersConfig
        config = TransformersConfig(
            model=model, tokenizer=tokenizer, device=args.device,
            max_new_tokens=205, min_new_tokens=195, do_sample=True,
            temperature=0.7, top_p=0.95, repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id)
        detector = AutoWatermark.load(
            args.algorithm,
            algorithm_config=str(markllm / "config" / f"{args.algorithm}.json"),
            transformers_config=config)

        @torch.inference_mode()
        def score_many(values: list[str], label: str) -> tuple[list[float], list[int]]:
            scores, invalid = [], []
            from tqdm.auto import tqdm
            progress = tqdm(total=len(values), desc=f"{args.algorithm} {label}",
                            unit="doc", dynamic_ncols=True)
            for document_id, text in zip(ids, values):
                try:
                    value = float(detector.detect_watermark(text)["score"])
                    if args.algorithm == "EXP": value = 1.0 - value
                except (IndexError, ValueError, ZeroDivisionError):
                    value = -1e9
                if not math.isfinite(value):
                    value = -1e9
                    invalid.append(document_id)
                scores.append(value)
                progress.update(1)
            progress.close()
            return scores, invalid

        cache_path = Path(args.null_cache).resolve()
        contract = {
            "algorithm": args.algorithm,
            "null_sha256": sha256(args.null),
            "document_ids": ids,
            "model": args.model,
        }
        cached = None
        if cache_path.exists():
            candidate = json.loads(cache_path.read_text(encoding="utf-8"))
            candidate_scores = candidate.get("scores", [])
            if (candidate.get("contract") == contract
                    and len(candidate_scores) == len(ids)):
                cached = candidate
            else:
                print(
                    f"stale null cache; recomputing and replacing {cache_path}",
                    flush=True,
                )
        if cached is not None:
            null_scores = [float(value) for value in cached["scores"]]
            null_invalid = [int(value) for value in cached.get("invalid_ids", [])]
        else:
            null_scores, null_invalid = score_many(texts(null, ids), "null")
            atomic_json(cache_path, {
                "schema_version": 1, "contract": contract,
                "scores": null_scores, "invalid_ids": null_invalid,
            })
        positive_scores, positive_invalid = score_many(texts(rewrite, ids), "wm")
    finally:
        os.chdir(old_cwd)

    method = {"SEMSTAMP": "semstamp", "KSEMSTAMP": "k-semstamp",
              "SynthID": "synthid", "EXP": "exp"}[args.algorithm]
    output = {
        "schema_version": 1,
        "experiment": "exp39_markllm_full_rewrite",
        "method": method,
        "condition": args.condition,
        "protocol": {
            "name": "AliMark-positive-only-full-rewrite",
            "positive": "complete rewrite of watermarked generation",
            "negative": "untouched human reference",
            "partial_rewrite": False,
            "unscorable_policy": "retain document with score -1e9",
        },
        "config": vars(args),
        "sources": {
            "watermarked_sha256": sha256(args.watermarked),
            "null_sha256": sha256(args.null),
            "rewrite_sha256": sha256(args.rewrite),
        },
        "generation_diagnostics": watermarked_payload.get(
            "generation_diagnostics"
        ),
        "scores": {
            "document_ids": ids,
            "watermarked": positive_scores,
            "null": null_scores,
            "invalid_watermarked_ids": positive_invalid,
            "invalid_null_ids": null_invalid,
        },
    }
    atomic_json(Path(args.out).resolve(), output)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
