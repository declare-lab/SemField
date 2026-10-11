"""Experiment 23: quality, distortion, and measured cost for SemField.

The generic experiment-13 aggregator identifies conditions by semantic cell
map.  SemField has no such map, so distinct bandwidths collide there.  This
script gives every realized feature map its own condition, loads the language
model judge only once, and pairs each SemField condition with the null artifact
having the same generator, prompts, and run seed.

Perplexity is evaluated over every continuation token with overlapping sliding
windows.  GPU judges run in FP16/BF16 by default; CPU judges run in FP32.  Model
files are cache-only unless ``--allow-downloads`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semfield import _hf_env  # noqa: F401,E402


def generated_files(patterns: Sequence[str]) -> list[Path]:
    found: dict[str, Path] = {}
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if not matches and Path(pattern).is_file():
            matches = [pattern]
        for match in matches:
            path = Path(match)
            if path.suffix == ".json":
                found[str(path.resolve())] = path.resolve()
    return sorted(found.values())


def _fingerprint(values: Sequence[str]) -> str:
    digest = hashlib.blake2b(digest_size=12)
    for value in values:
        digest.update(str(value).encode("utf-8", "ignore"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _ngrams(tokens: Sequence[str], width: int):
    return zip(*(tokens[index:] for index in range(width))) \
        if len(tokens) >= width else ()


def surface_metrics(documents: Sequence[dict]) -> dict[str, float | int]:
    texts = [str(row.get("text", "")) for row in documents
             if str(row.get("text", "")).strip()]
    tokens = [token.lower() for text in texts for token in text.split()]
    words = [len(text.split()) for text in texts]
    sentences = [len(row.get("sentences", ())) for row in documents]
    output: dict[str, float | int] = {
        "documents": len(texts),
        "words": len(tokens),
        "mean_words_per_document": float(np.mean(words)) if words else 0.0,
        "mean_sentences_per_document": (
            float(np.mean(sentences)) if sentences else 0.0
        ),
    }
    for width in (1, 2, 3):
        counts = Counter(_ngrams(tokens, width))
        total = sum(counts.values())
        output[f"distinct_{width}"] = len(counts) / max(1, total)
    return output


def load_judge(name: str, device: str | None, precision: str,
               allow_downloads: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not str(resolved_device).startswith("cuda"):
        dtype = torch.float32
    elif precision == "fp32":
        dtype = torch.float32
    elif precision == "bf16":
        dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported()
                 else torch.float16)
    elif precision == "fp16":
        dtype = torch.float16
    else:
        dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported()
                 else torch.float16)
    cache_only = not allow_downloads
    tokenizer = AutoTokenizer.from_pretrained(
        name, local_files_only=cache_only
    )
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=dtype, local_files_only=cache_only
    ).to(resolved_device).eval()
    print(f"judge: {name}, {str(dtype).removeprefix('torch.')} on "
          f"{resolved_device}")
    return model, tokenizer, resolved_device, str(dtype).removeprefix("torch.")


def document_nll(text: str, model, tokenizer, device: str,
                 max_length: int, stride: int) -> tuple[float, int]:
    import torch

    ids = tokenizer(
        text, return_tensors="pt", add_special_tokens=False
    ).input_ids
    sequence_length = int(ids.shape[1])
    if sequence_length < 2:
        return 0.0, 0
    total_nll = 0.0
    total_tokens = 0
    previous_end = 0
    with torch.inference_mode():
        for begin in range(0, sequence_length, stride):
            end = min(begin + max_length, sequence_length)
            new_tokens = end - previous_end
            if new_tokens <= 0:
                continue
            input_ids = ids[:, begin:end].to(device)
            labels = input_ids.clone()
            if new_tokens < labels.shape[1]:
                labels[:, :-new_tokens] = -100
            predicted = int((labels[:, 1:] != -100).sum().item())
            if predicted:
                loss = model(input_ids=input_ids, labels=labels).loss
                total_nll += float(loss) * predicted
                total_tokens += predicted
            previous_end = end
            if end == sequence_length:
                break
    return total_nll, total_tokens


def _bootstrap_perplexity(rows: Sequence[dict], draws: int,
                          seed: int) -> tuple[float, list[float]]:
    total_nll = sum(float(row["nll"]) for row in rows)
    total_tokens = sum(int(row["tokens"]) for row in rows)
    point = math.exp(total_nll / max(1, total_tokens))
    if draws <= 0 or not rows:
        return point, [point, point]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        indices = rng.integers(0, len(rows), size=len(rows))
        nll = sum(float(rows[int(i)]["nll"]) for i in indices)
        tokens = sum(int(rows[int(i)]["tokens"]) for i in indices)
        values.append(math.exp(nll / max(1, tokens)))
    return point, [float(value) for value in np.quantile(values, [.025, .975])]


def score_documents(documents: Sequence[dict], model, tokenizer, device: str,
                    max_length: int, stride: int, bootstrap: int,
                    seed: int) -> dict[str, object]:
    rows = []
    from tqdm.auto import tqdm
    progress = tqdm(total=len(documents), desc="semfield quality",
                    unit="doc", dynamic_ncols=True)
    for position, document in enumerate(documents):
        text = str(document.get("text", ""))
        if not text.strip():
            continue
        nll, tokens = document_nll(
            text, model, tokenizer, device, max_length, stride
        )
        if tokens:
            rows.append({
                "document_id": int(document.get("document_id", position)),
                "nll": float(nll),
                "tokens": int(tokens),
                "mean_nll": float(nll / tokens),
            })
        progress.update(1)
    progress.close()
    perplexity, interval = _bootstrap_perplexity(rows, bootstrap, seed)
    return {
        "perplexity": float(perplexity),
        "perplexity_ci95": interval,
        "scored_documents": len(rows),
        "scored_tokens": int(sum(row["tokens"] for row in rows)),
        "max_length": int(max_length),
        "stride": int(stride),
        "per_document": rows,
    }


def _condition(payload: dict) -> str:
    config = payload.get("config", {})
    summary = payload.get("summary", {})
    feature = payload.get("feature_map", {})
    if "polarity_locked" in config:
        selection = "SemField-PL" if config["polarity_locked"] else "SemField"
    else:
        selection = config.get("selection_objective", "local")
        selection = {"self-normalized": "SemField", "polarity-locked": "SemField-PL"}.get(
            selection, selection
        )
    candidates = config.get("candidates", config.get("semfield_candidates", 2))
    selection_suffix = (
        f"|sel{selection}|c{candidates}"
        if summary.get("variant") == "semfield" else ""
    )
    return (
        f"{config.get('generator', 'na')}|{summary.get('variant', 'na')}|"
        f"{feature.get('type', 'na')}|m{feature.get('n_features', 'na')}|"
        f"l{feature.get('length_scale', 'na')}|f{feature.get('seed', 'na')}|"
        f"key{payload.get('key_seed', 'na')}|run{config.get('seed', 'na')}"
        f"{selection_suffix}"
    )


def _match_key(payload: dict) -> tuple[object, ...]:
    config = payload.get("config", {})
    feature = payload.get("feature_map", {})
    return (
        config.get("generator"), config.get("embedder"),
        config.get("seed"), payload.get("key_seed"),
        _fingerprint(payload.get("prompts", ())),
        feature.get("type"), feature.get("n_features"),
        feature.get("length_scale"), feature.get("seed"),
    )


def _paired_quality(watermarked: dict, null: dict, bootstrap: int,
                    seed: int) -> dict[str, object] | None:
    wm_rows = {int(row["document_id"]): row for row in
               watermarked.get("language_model", {}).get("per_document", ())}
    null_rows = {int(row["document_id"]): row for row in
                 null.get("language_model", {}).get("per_document", ())}
    common = sorted(set(wm_rows) & set(null_rows))
    if not common:
        return None
    differences = np.asarray([
        float(wm_rows[index]["mean_nll"]) -
        float(null_rows[index]["mean_nll"]) for index in common
    ])
    point = float(differences.mean())
    interval = [point, point]
    if bootstrap > 0:
        rng = np.random.default_rng(seed)
        draws = [float(differences[
            rng.integers(0, differences.size, size=differences.size)
        ].mean()) for _ in range(bootstrap)]
        interval = [float(value) for value in np.quantile(draws, [.025, .975])]
    return {
        "paired_documents": len(common),
        "mean_nll_difference_watermarked_minus_null": point,
        "mean_nll_difference_ci95": interval,
        "perplexity_ratio_geometric_document_mean": float(math.exp(point)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", nargs="+", required=True)
    parser.add_argument("--judge", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", choices=["auto", "fp16", "bf16", "fp32"],
                        default="auto")
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--skip-ppl", action="store_true")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--out", default="results_exp23_semfield_quality.json")
    args = parser.parse_args(argv)

    if args.bootstrap < 0:
        parser.error("--bootstrap must be non-negative")
    if args.max_documents is not None and args.max_documents <= 0:
        parser.error("--max-documents must be positive")

    paths = generated_files(args.generations)
    if not paths:
        parser.error("no generation JSON files matched")
    artifacts = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("experiment") != "exp19_semfield_generate":
            continue
        artifacts.append((path, payload))
    if not artifacts:
        parser.error("no exp19 SemField generation JSON files matched")

    model = tokenizer = None
    device = args.device or "cpu"
    dtype = None
    max_length = args.max_length
    if not args.skip_ppl:
        model, tokenizer, device, dtype = load_judge(
            args.judge, args.device, args.precision, args.allow_downloads
        )
        advertised = int(getattr(model.config, "max_position_embeddings", 1024))
        max_length = min(int(max_length or advertised), advertised)
        if args.stride <= 0 or args.stride > max_length:
            parser.error("--stride must be positive and no larger than max length")

    results: dict[str, dict] = {}
    null_by_match: dict[tuple[object, ...], str] = {}
    for artifact_i, (path, payload) in enumerate(artifacts):
        condition = _condition(payload)
        if condition in results:
            raise ValueError(f"duplicate condition {condition}: {path}")
        all_documents = list(payload.get("documents", ()))
        documents = all_documents[:args.max_documents]
        documents = [row for row in documents if str(row.get("text", "")).strip()]
        summary = payload.get("summary", {})
        row: dict[str, object] = {
            "file": str(path),
            "condition": condition,
            "variant": summary.get("variant"),
            "generator": payload.get("config", {}).get("generator"),
            "feature_map": payload.get("feature_map"),
            "surface": surface_metrics(documents),
            "generation_cost": summary,
            "cohort": {
                "documents_total": len(all_documents),
                "documents_in_scope": len(documents),
                "nonempty_fraction": (
                    len(documents) / len(all_documents) if all_documents else 0.0
                ),
            },
            "match_key": [str(value) for value in _match_key(payload)],
        }
        if not args.skip_ppl:
            row["language_model"] = score_documents(
                documents, model, tokenizer, device, int(max_length),
                args.stride, args.bootstrap, args.seed + artifact_i,
            )
        results[condition] = row
        if summary.get("variant") == "none":
            match = _match_key(payload)
            if match in null_by_match:
                raise ValueError("more than one null artifact has the same match key")
            null_by_match[match] = condition

    paired = {}
    for artifact_i, (_, payload) in enumerate(artifacts):
        if payload.get("summary", {}).get("variant") != "semfield":
            continue
        condition = _condition(payload)
        null_condition = null_by_match.get(_match_key(payload))
        if null_condition is None:
            paired[condition] = {"status": "no matched null artifact"}
            continue
        comparison = _paired_quality(
            results[condition], results[null_condition],
            args.bootstrap, args.seed + 100_003 + artifact_i,
        ) if not args.skip_ppl else None
        paired[condition] = {
            "status": "matched",
            "null_condition": null_condition,
            "language_model": comparison,
            "requested_candidates_per_sentence_ratio": (
                float(results[condition]["generation_cost"].get(
                    "requested_candidates_per_sentence", 0.0
                )) / max(1e-15, float(results[null_condition][
                    "generation_cost"
                ].get("requested_candidates_per_sentence", 0.0)))
            ),
        }

    output = {
        "schema_version": 1,
        "experiment": "exp23_semfield_quality",
        "config": {**vars(args), "resolved_device": device,
                   "resolved_dtype": dtype, "resolved_max_length": max_length},
        "results": results,
        "paired": paired,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
