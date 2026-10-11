"""Generate official MarkLLM semantic baselines on the shared AliMark cohort.

The output contains only the generated continuation.  The negative file is the
same untouched human reference for every method, matching AliMark's released
evaluation.  Runs are atomic and resumable per document.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import semfield._hf_env  # noqa: F401,E402
from semfield.generation import default_precision_for_model  # noqa: E402
from experiments.exp27_markllm_baseline import (
    atomic_json,
    generated_only,
    jsonable,
    preflight,
    seed_all,
    sentences,
)


def journal_rows(path: Path) -> dict[int, dict]:
    """Read checkpointed document rows from an append-only journal."""
    if not path.exists():
        return {}
    rows: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[int(row["document_id"])] = row
    return rows


def journal_append(path: Path, row: dict) -> None:
    """Append one completed document. Rewriting the whole list per document
    made checkpoint I/O quadratic in the cohort size."""
    with path.open("a", encoding="utf-8") as handle:
        json.dump(row, handle, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def diagnostics_summary(rows: list[dict]) -> dict:
    """Summarize SemStamp generation diagnostics without changing scores."""
    values = [row.get("generation_diagnostics") for row in rows
              if isinstance(row.get("generation_diagnostics"), dict)]
    if not values:
        return {"documents": 0}

    def total(name: str) -> int:
        return int(sum(int(value.get(name, 0) or 0) for value in values))

    def mean(name: str):
        numbers = [float(value[name]) for value in values
                   if value.get(name) is not None]
        return (sum(numbers) / len(numbers)) if numbers else None

    return {
        "documents": len(values),
        "total_trials": total("total_trials"),
        "total_emitted_sentences": total("emitted_sentences"),
        "total_accepted_trials": total("accepted_trials"),
        "total_retry_count": total("retry_count"),
        "total_exhaustion_count": total("exhaustion_count"),
        "mean_acceptance_rate": mean("acceptance_rate"),
        "mean_retry_count": mean("retry_count"),
        "mean_exhaustion_count": mean("exhaustion_count"),
        "mean_trials_per_sentence": mean("mean_trials_per_sentence"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=["SEMSTAMP", "KSEMSTAMP", "SynthID", "EXP"],
                        required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--markllm-dir", default="third_party/MarkLLM")
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL", "facebook/opt-1.3b"),
        help="causal LM checkpoint (defaults to $MODEL or OPT-1.3B)",
    )
    parser.add_argument("--documents", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"],
                        default=None)
    parser.add_argument("--max-new-tokens", type=int, default=205)
    parser.add_argument("--min-new-tokens", type=int, default=195)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--matched-controls", action="store_true",
                        help="also generate method-matched unwatermarked outputs")
    args = parser.parse_args()
    if args.precision is None:
        args.precision = default_precision_for_model(args.model)
    if args.documents < 1:
        parser.error("--documents must be positive")

    markllm = Path(args.markllm_dir).resolve()
    preflight(markllm, args.algorithm)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    source = json.loads(Path(args.input).read_text(encoding="utf-8"))
    inputs = source["documents"][:args.documents]
    out_dir = Path(args.out_dir).resolve()
    wm_path = out_dir / "watermarked.json"
    null_path = out_dir / "human-reference.json"
    control_path = out_dir / "unwatermarked.json"
    wm_journal = wm_path.with_suffix(wm_path.suffix + ".partial.jsonl")
    control_journal = control_path.with_suffix(control_path.suffix + ".partial.jsonl")
    complete: dict[int, dict] = {}
    controls: dict[int, dict] = {}
    if args.resume and wm_path.exists():
        old = json.loads(wm_path.read_text(encoding="utf-8"))
        complete = {int(row["document_id"]): row
                    for row in old.get("documents", [])}
    elif wm_path.exists():
        parser.error(f"{wm_path} exists; pass --resume")
    complete.update(journal_rows(wm_journal))
    if args.matched_controls and args.resume and control_path.exists():
        old = json.loads(control_path.read_text(encoding="utf-8"))
        controls = {int(row["document_id"]): row
                    for row in old.get("documents", [])}
    if args.matched_controls:
        controls.update(journal_rows(control_journal))

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
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens, do_sample=True,
            temperature=args.temperature, top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id)
        watermark = AutoWatermark.load(
            args.algorithm,
            algorithm_config=str(markllm / "config" / f"{args.algorithm}.json"),
            transformers_config=config)
        common = {
            "schema_version": 1,
            "experiment": "exp38_markllm_semantic_shared",
            "method": args.algorithm,
            "config": jsonable(vars(args)),
            "protocol": {
                "cohort": "AliMark shared C4",
                "positive": "watermarked model continuation",
                "negative": "untouched human reference",
                "partial_rewrite": False,
            },
            "source": {
                "markllm_commit": "c45ddc40f7b761beabe55a1b8dc4690e531d1c6d",
            },
        }
        human = []
        skipped: list[int] = []
        from tqdm.auto import tqdm
        progress = tqdm(total=len(inputs), initial=len(complete),
                        desc=f"{args.algorithm} wm", unit="wm",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                        dynamic_ncols=True)
        for position, item in enumerate(inputs):
            document_id = int(item["document_id"])
            prompt = str(item["prompt"])
            human.append({
                "document_id": document_id, "prompt": prompt,
                "text": str(item["reference"]),
                "sentences": sentences(str(item["reference"])),
            })
            if document_id not in complete:
                parts = []
                diagnostics = None
                for attempt in range(5):
                    seed_all(args.seed + document_id + attempt * 1_000_003)
                    with torch.inference_mode():
                        full = watermark.generate_watermarked_text(prompt)
                    text = generated_only(prompt, full, tokenizer)
                    parts = sentences(text)
                    if parts:
                        diagnostics = getattr(
                            watermark, "last_generation_stats", None)
                        if attempt > 0:
                            print(f"{args.algorithm} document {document_id}: empty continuation on "
                                  f"attempt(s) 0-{attempt - 1}, succeeded on attempt {attempt} "
                                  f"(seed={args.seed + document_id + attempt * 1_000_003})", flush=True)
                        break
                if not parts:
                    print(f"WARNING: {args.algorithm} document {document_id} generated no continuation "
                          f"after 5 attempts (deterministic watermark-key/context degeneracy) — "
                          f"excluding this document from {args.algorithm}'s generation set.", flush=True)
                    skipped.append(document_id)
                    progress.update(1)
                    continue
                complete[document_id] = {
                    "document_id": document_id, "prompt": prompt,
                    "text": text, "sentences": parts,
                }
                if isinstance(diagnostics, dict):
                    complete[document_id]["generation_diagnostics"] = diagnostics
                    print(
                        f"{args.algorithm} document {document_id}: "
                        f"acceptance_rate={diagnostics['acceptance_rate']} "
                        f"retries={diagnostics['retry_count']} "
                        f"exhaustions={diagnostics['exhaustion_count']} "
                        f"trials={diagnostics['total_trials']}",
                        flush=True,
                    )
                journal_append(wm_journal, complete[document_id])
                progress.update(1)
            if args.matched_controls and document_id not in controls:
                parts = []
                for attempt in range(5):
                    seed_all(args.seed + document_id + attempt * 1_000_003)
                    with torch.inference_mode():
                        full = watermark.generate_unwatermarked_text(prompt)
                    text = generated_only(prompt, full, tokenizer)
                    parts = sentences(text)
                    if parts:
                        if attempt > 0:
                            print(f"{args.algorithm} control document {document_id}: empty continuation on "
                                  f"attempt(s) 0-{attempt - 1}, succeeded on attempt {attempt} "
                                  f"(seed={args.seed + document_id + attempt * 1_000_003})", flush=True)
                        break
                if not parts:
                    print(f"WARNING: {args.algorithm} control document {document_id} generated no continuation "
                          f"after 5 attempts (deterministic watermark-key/context degeneracy) — "
                          f"excluding this document from {args.algorithm}'s control set.", flush=True)
                    if document_id not in skipped:
                        skipped.append(document_id)
                    continue
                controls[document_id] = {
                    "document_id": document_id, "prompt": prompt,
                    "text": text, "sentences": parts,
                }
                journal_append(control_journal, controls[document_id])
        progress.close()
        atomic_json(wm_path, {
            **common, "status": "complete", "variant": "watermarked",
            "generation_diagnostics": diagnostics_summary(
                [complete[k] for k in sorted(complete)]
            ),
            "documents": [complete[k] for k in sorted(complete)],
        })
        wm_journal.unlink(missing_ok=True)
        atomic_json(null_path, {
            **common, "status": "complete", "variant": "human-reference",
            "documents": human,
        })
        if args.matched_controls:
            atomic_json(control_path, {
                **common, "status": "complete",
                "variant": "unwatermarked-control",
                "documents": [controls[k] for k in sorted(controls)],
            })
            control_journal.unlink(missing_ok=True)
    finally:
        os.chdir(old_cwd)
    print(f"wrote {wm_path} and {null_path}")
    if skipped:
        print(f"{args.algorithm}: excluded {len(skipped)} document(s) with no usable continuation: "
              f"{sorted(set(skipped))} ({len(complete)}/{len(inputs)} generated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
