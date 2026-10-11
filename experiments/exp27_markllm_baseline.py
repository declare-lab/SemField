"""Run faithful MarkLLM KGW or SIR on the shared baseline prompts.

This adapter uses the pinned official MarkLLM tree.  It is resumable at the
method level, records both watermarked and matched unwatermarked generations,
and never downloads models.  SIR deliberately fails preflight when its trained
transform and embedding model are absent; substituting another encoder would
not be a faithful SIR baseline.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import semfield._hf_env  # noqa: F401,E402
from semfield.generation import default_precision_for_model  # noqa: E402


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
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def jsonable(value):
    """Convert detector tensors/NumPy scalars to strict JSON values."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def sentences(text: str) -> list[str]:
    from nltk.tokenize import sent_tokenize
    return [part.strip() for part in sent_tokenize(text) if part.strip()]


def generated_only(prompt: str, text: str, tokenizer=None) -> str:
    """Remove a prompt from a decoded generation.

    Chat templates contain special role tokens.  MarkLLM decodes generated
    sequences with ``skip_special_tokens=True``, so the decoded prefix is not
    byte-for-byte equal to the rendered prompt.  Compare against both forms;
    otherwise an instruction benchmark accidentally scores the user prompt as
    part of the generated answer.
    """
    value = text.strip()
    prefixes = [prompt.strip()]
    if tokenizer is not None:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        visible = tokenizer.decode(prompt_ids, skip_special_tokens=True).strip()
        if visible:
            prefixes.append(visible)
    for prefix in sorted(set(prefixes), key=len, reverse=True):
        if value.startswith(prefix):
            value = value[len(prefix):].lstrip()
            break
    return value


def required_paths(markllm: Path, algorithm: str) -> list[Path]:
    paths = [markllm / "config" / f"{algorithm}.json"]
    if algorithm == "SIR":
        config = json.loads(paths[0].read_text(encoding="utf-8"))
        paths.extend([markllm / config["transform_model_name"],
                      markllm / config["embedding_model_path"]])
    return paths


def preflight(markllm: Path, algorithm: str) -> None:
    missing = [str(path) for path in required_paths(markllm, algorithm)
               if not path.exists()]
    if missing:
        raise RuntimeError(
            f"{algorithm} cannot run faithfully; missing official artifacts:\n  "
            + "\n  ".join(missing)
        )
    import importlib.util
    packages = ["torch", "transformers", "jieba"]
    absent = [name for name in packages if importlib.util.find_spec(name) is None]
    if absent:
        raise RuntimeError("missing installed packages: " + ", ".join(absent))


def seed_all(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def detect_inference(watermark, text: str):
    import torch
    with torch.inference_mode():
        return watermark.detect_watermark(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=["KGW", "SIR"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--markllm-dir", default="third_party/MarkLLM")
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL", "facebook/opt-1.3b"),
        help="causal LM checkpoint (defaults to $MODEL or OPT-1.3B)",
    )
    parser.add_argument("--documents", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"],
                        default=None)
    parser.add_argument("--max-new-tokens", type=int, default=205)
    parser.add_argument("--min-new-tokens", type=int, default=195)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.precision is None:
        args.precision = default_precision_for_model(args.model)

    markllm = Path(args.markllm_dir).resolve()
    preflight(markllm, args.algorithm)
    if args.preflight:
        print(f"{args.algorithm} preflight passed")
        return

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    inputs = payload["documents"][:args.documents]
    out_dir = Path(args.out_dir).resolve()
    wm_path = out_dir / "watermarked.json"
    null_path = out_dir / "unwatermarked.json"
    existing_wm = {}
    existing_null = {}
    if args.resume:
        for path, target in ((wm_path, existing_wm), (null_path, existing_null)):
            if path.exists():
                value = json.loads(path.read_text(encoding="utf-8"))
                target.update({int(row["document_id"]): row
                               for row in value.get("documents", [])})

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
             "fp32": torch.float32}[args.precision]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=dtype
    ).to(args.device).eval()
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
            min_new_tokens=args.min_new_tokens,
            do_sample=True, temperature=args.temperature, top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
        )
        algorithm_config = markllm / "config" / f"{args.algorithm}.json"
        if args.algorithm == "SIR":
            # SIR.json hardcodes a mapping file sized for OPT's 50272-token
            # vocabulary (watermark/sir/mapping/300_mapping_50272.json). Every
            # other model's tokenizer produces ids outside that range, and
            # MarkLLM's SIR loads the file whenever it exists rather than
            # checking it matches the current vocab, so a generator this
            # mapping was not built for indexes past the end of the array.
            # _get_mapping() in watermark/sir/sir.py already handles a new
            # vocabulary correctly -- it generates and saves a fresh mapping
            # whenever the target file is absent -- so pointing at a
            # vocab-sized filename that does not yet exist is enough; SIR.json
            # itself, and every mapping file it names, is left untouched.
            import json as _json
            vocab_size = len(tokenizer)
            sir_config = _json.loads(algorithm_config.read_text())
            sir_config["mapping_name"] = (
                f"watermark/sir/mapping/300_mapping_{vocab_size}.json"
            )
            model_config = markllm / "config" / f"SIR-{vocab_size}.json"
            model_config.write_text(_json.dumps(sir_config, indent=2))
            algorithm_config = model_config
        watermark = AutoWatermark.load(
            args.algorithm,
            algorithm_config=str(algorithm_config),
            transformers_config=config,
        )
        started = time.time()
        from tqdm.auto import tqdm
        progress = tqdm(total=len(inputs), initial=len(existing_wm),
                        desc=f"{args.algorithm} wm", unit="wm",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                        dynamic_ncols=True)
        for position, item in enumerate(inputs):
            document_id = int(item["document_id"])
            prompt = item["prompt"]
            if document_id not in existing_null:
                seed_all(args.seed + 2 * document_id)
                with torch.inference_mode():
                    generated = watermark.generate_unwatermarked_text(prompt)
                text = generated_only(prompt, generated, tokenizer)
                existing_null[document_id] = {
                    "document_id": document_id, "prompt": prompt,
                    "text": text, "sentences": sentences(text),
                    "detector": jsonable(detect_inference(watermark, text)),
                }
            if document_id not in existing_wm:
                seed_all(args.seed + 2 * document_id + 1)
                with torch.inference_mode():
                    generated = watermark.generate_watermarked_text(prompt)
                text = generated_only(prompt, generated, tokenizer)
                existing_wm[document_id] = {
                    "document_id": document_id, "prompt": prompt,
                    "text": text, "sentences": sentences(text),
                    "detector": jsonable(detect_inference(watermark, text)),
                }
            common = {
                "schema_version": 1,
                "experiment": "exp27_markllm_baseline",
                "method": args.algorithm,
                "config": jsonable(vars(args)),
                "source": {"markllm_commit": "c45ddc40f7b761beabe55a1b8dc4690e531d1c6d"},
            }
            atomic_json(null_path, {**common, "variant": "none",
                "documents": [existing_null[k] for k in sorted(existing_null)]})
            atomic_json(wm_path, {**common, "variant": "watermarked",
                "documents": [existing_wm[k] for k in sorted(existing_wm)]})
            progress.update(1)
        progress.close()
        print(f"complete in {time.time() - started:.1f}s: {out_dir}")
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
