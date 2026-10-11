"""Experiment 16: faithful row-conditioned SemStamp generation.

This experiment is deliberately separate from exp11 so the SemClock path and
its accounting remain unchanged.  It implements the semantic transition
structure used by SemStamp: the previous sentence's LSH cell selects a fixed
size set of valid next cells, and completed-sentence candidates are sampled
one at a time until one enters that set and clears an optional LSH margin.

The default key mapping is an ideal-PRF baseline with independent uniform
fixed-size row masks.  It is faithful to the statistical row law but is NOT
bit-for-bit compatible with the released SemStamp repository's RNG stream.
This is recorded in every output file.

Example:

    python experiments/exp16_semstamp_generate.py \
        --generator opt-1.3b --embedder mpnet \
        --corpus-dir data/c4-realnewslike \
        --documents 100 --sentences 40 --lsh-bits 3 --n-valid 2 \
        --min-margin 0.02 --max-attempts 32
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.generation_utils import build_prompts, sample_completed
from semfield.cells import LSHCells
from semfield.corpora import split_sentences
from semfield.generation import load_generator
from semfield.semstamp_row import IDEAL_PRF_MODE, IdealRowSemStamp


def experiment_key(seed: int) -> bytes:
    """Domain-separated setup key for the ideal row baseline."""
    return hashlib.blake2b(
        f"semstamp-row-ideal-prf|{seed}".encode(), digest_size=32
    ).digest()


def _detector_payload(scheme: IdealRowSemStamp, cells: list[int],
                      initial_cell: int) -> dict:
    result = scheme.detect(cells, initial_cell=initial_cell)
    moments = {
        name: (None if isinstance(value, float) and not math.isfinite(value) else value)
        for name, value in result.moments.to_dict().items()
    }
    return {
        "score": result.score,
        "n_transitions": result.n_transitions,
        "exact_pvalue": result.pvalue,
        "null_moments": moments,
        "mask_mode": result.mask_mode,
        "released_rng_compatible": result.released_rng_compatible,
    }


def generate_row_semstamp(
    model,
    tokenizer,
    prompts: list[str],
    initial_sentences: list[str],
    initial_cells: list[int],
    cell_map,
    embedder,
    scheme: IdealRowSemStamp,
    *,
    n_sentences: int,
    seed: int,
    max_attempts: int,
    on_exhaustion: str,
    max_new_tokens: int,
    max_context_tokens: int,
    temperature: float,
    top_p: float,
    max_rows: int | None,
    max_completion_retry_rounds: int,
) -> tuple[list[dict], dict]:
    """Generate with sequential rejection at an unchanged context.

    One *SemStamp attempt* is one completed sentence whose cell and margin are
    evaluated.  Raw model requests that fail to close a sentence are counted
    separately and never treated as watermark candidates.
    """
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if on_exhaustion not in {"stop", "emit-last"}:
        raise ValueError("on_exhaustion must be 'stop' or 'emit-last'")
    n_documents = len(prompts)
    if not (len(initial_sentences) == len(initial_cells) == n_documents):
        raise ValueError("initial sentence/cell arrays must match prompts")

    contexts = list(prompts)
    records = [{
        "document_id": i,
        "initial_sentence": initial_sentences[i],
        "initial_cell": int(initial_cells[i]),
        "sentences": [],
        "cells": [],
        "transitions": [],
        "exhaustions": [],
    } for i in range(n_documents)]
    previous_cells = [int(cell) for cell in initial_cells]
    active = list(range(n_documents))

    raw_requested = 0
    raw_completed = 0
    completed_attempts = 0
    accepted_transitions = 0
    fallback_transitions = 0
    exhausted_transitions = 0
    completion_failures = 0
    sampling_calls = 0
    started = time.perf_counter()

    def commit(document_i: int, sentence: str, cell: int, attempts: list[dict],
               *, accepted: bool, exhausted: bool, fallback: bool) -> None:
        nonlocal accepted_transitions, fallback_transitions
        source = previous_cells[document_i]
        final = attempts[-1]
        transition = {
            "step": len(records[document_i]["sentences"]),
            "source_cell": int(source),
            "destination_cell": int(cell),
            "green": bool(final["green"]),
            "margin": final["margin"],
            "margin_ok": bool(final["margin_ok"]),
            "accepted": bool(accepted),
            "attempts": len(attempts),
            "exhausted": bool(exhausted),
            "fallback": bool(fallback),
            "attempt_trace": attempts,
        }
        records[document_i]["sentences"].append(sentence)
        records[document_i]["cells"].append(int(cell))
        records[document_i]["transitions"].append(transition)
        contexts[document_i] += " " + sentence
        previous_cells[document_i] = int(cell)
        if accepted:
            accepted_transitions += 1
        if fallback:
            fallback_transitions += 1

    for step in range(n_sentences):
        if not active:
            break
        pending = list(active)
        traces: dict[int, list[dict]] = {i: [] for i in pending}
        last_candidates: dict[int, tuple[str, int]] = {}
        next_active: list[int] = []

        for attempt_index in range(max_attempts):
            if not pending:
                break
            groups, sampling = sample_completed(
                model,
                tokenizer,
                [contexts[i] for i in pending],
                target=1,
                seed=seed + step * 10_000_019 + attempt_index * 1_000_003,
                max_new_tokens=max_new_tokens,
                max_context_tokens=max_context_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.05,
                max_rows=max_rows,
                max_retry_rounds=max_completion_retry_rounds,
                sample_ids=pending,
            )
            sampling_calls += 1
            raw_requested += int(sampling["requested"])
            raw_completed += int(sampling["completed"])
            failed_local = set(sampling["failed_indices"])

            successful: list[tuple[int, str]] = []
            failed_documents: list[int] = []
            for local_i, document_i in enumerate(pending):
                if local_i in failed_local or not groups[local_i]:
                    failed_documents.append(document_i)
                else:
                    successful.append((document_i, groups[local_i][0]))

            for document_i in failed_documents:
                completion_failures += 1
                exhausted_transitions += 1
                records[document_i]["exhaustions"].append({
                    "step": step,
                    "reason": "no_completed_sentence",
                    "completed_attempts": len(traces[document_i]),
                })

            if successful:
                vectors = embedder([sentence for _, sentence in successful])
                cells = np.asarray(cell_map(vectors), dtype=int)
                margins = np.asarray(cell_map.margins(vectors), dtype=float)
            else:
                cells = np.zeros(0, dtype=int)
                margins = np.zeros(0, dtype=float)

            still_pending: list[int] = []
            for row_i, (document_i, sentence) in enumerate(successful):
                cell = int(cells[row_i])
                margin = float(margins[row_i])
                decision = scheme.decide(
                    previous_cells[document_i], cell, margin=margin
                )
                completed_attempts += 1
                trace = {
                    "attempt": len(traces[document_i]) + 1,
                    "candidate_cell": cell,
                    "margin": margin,
                    "green": decision.green,
                    "margin_ok": decision.margin_ok,
                    "accepted": decision.accepted,
                }
                traces[document_i].append(trace)
                last_candidates[document_i] = (sentence, cell)
                if decision.accepted:
                    commit(
                        document_i, sentence, cell, traces[document_i],
                        accepted=True, exhausted=False, fallback=False,
                    )
                    next_active.append(document_i)
                else:
                    still_pending.append(document_i)
            pending = still_pending

        for document_i in pending:
            exhausted_transitions += 1
            records[document_i]["exhaustions"].append({
                "step": step,
                "reason": "max_rejection_attempts",
                "completed_attempts": len(traces[document_i]),
            })
            if on_exhaustion == "emit-last" and document_i in last_candidates:
                sentence, cell = last_candidates[document_i]
                commit(
                    document_i, sentence, cell, traces[document_i],
                    accepted=False, exhausted=True, fallback=True,
                )
                next_active.append(document_i)

        active = sorted(next_active)
        print(
            f"    semstamp-row: sentence {step + 1}/{n_sentences}, "
            f"{len(active)}/{n_documents} documents active"
        )

    elapsed = time.perf_counter() - started
    for i, record in enumerate(records):
        record["text"] = " ".join(record["sentences"])
        record["sentences_emitted"] = len(record["sentences"])
        record["detector"] = _detector_payload(
            scheme, record["cells"], record["initial_cell"]
        )

    total_emitted = sum(record["sentences_emitted"] for record in records)
    summary = {
        "scheme": "semstamp-row-ideal-prf",
        "documents": n_documents,
        "sentences_emitted": total_emitted,
        "raw_model_requests": raw_requested,
        "raw_completed_sentences": raw_completed,
        "completed_candidates_evaluated": completed_attempts,
        "completed_candidates_per_emitted_sentence": (
            completed_attempts / max(1, total_emitted)
        ),
        "raw_requests_per_emitted_sentence": (
            raw_requested / max(1, total_emitted)
        ),
        "sentence_completion_rate": raw_completed / max(1, raw_requested),
        "accepted_transitions": accepted_transitions,
        "fallback_transitions": fallback_transitions,
        "exhausted_transitions": exhausted_transitions,
        "completion_failures": completion_failures,
        "sampling_calls": sampling_calls,
        "on_exhaustion": on_exhaustion,
        "seconds_total": elapsed,
        "seconds_per_document": elapsed / max(1, n_documents),
        "mask": scheme.metadata(),
    }
    return records, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", default="data/c4-realnewslike")
    parser.add_argument("--generator", default="opt-1.3b")
    parser.add_argument("--vllm-python",
                        default=sys.executable)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=192)
    parser.add_argument("--vllm-request-size", type=int, default=512)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float,
                        default=0.90)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--embedder", default="mpnet")
    parser.add_argument("--documents", type=int, default=100)
    parser.add_argument("--prompt-sentences", type=int, default=3)
    parser.add_argument("--sentences", type=int, default=40)
    parser.add_argument("--lsh-bits", type=int, default=3)
    parser.add_argument("--n-valid", type=int, default=2)
    parser.add_argument("--min-margin", type=float, default=0.02)
    parser.add_argument("--map-seed", type=int, default=0)
    parser.add_argument("--key-seed", type=int, default=0)
    parser.add_argument("--mask-mode", choices=[IDEAL_PRF_MODE],
                        default=IDEAL_PRF_MODE)
    parser.add_argument("--max-attempts", type=int, default=32)
    parser.add_argument("--on-exhaustion", choices=["stop", "emit-last"],
                        default="stop")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--max-context-tokens", type=int, default=512)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-completion-retry-rounds", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="generations_semstamp_row")
    args = parser.parse_args()

    n_cells = 2 ** args.lsh_bits
    if not 0 < args.n_valid < n_cells:
        parser.error("--n-valid must lie strictly between 0 and 2**lsh-bits")

    from semfield.embedders import embed, load_embedder, resolve_devices

    prompts, references, source = build_prompts(
        args.corpus_dir, args.documents, args.prompt_sentences
    )
    initial_sentences = [split_sentences(prompt)[-1] for prompt in prompts]
    devices = resolve_devices(args.device)
    encoder = load_embedder(args.embedder, device=devices[0])

    def embedder(sentences):
        return embed(encoder, sentences, devices=devices,
                     model_name=args.embedder)

    args.generation_backend = "vllm"
    model, tokenizer = load_generator(
        args.generator, device=devices[0],
        vllm_python=args.vllm_python,
        vllm_max_num_seqs=args.vllm_max_num_seqs,
        vllm_request_size=args.vllm_request_size,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        max_model_len=args.max_context_tokens + args.max_new_tokens,
    )
    probe = embedder(
        initial_sentences + ["A sentence used to determine embedding width."]
    )
    cell_map = LSHCells(dim=probe.shape[1], d=args.lsh_bits,
                        seed=args.map_seed)
    initial_cells = cell_map(probe[:len(initial_sentences)]).astype(int).tolist()
    scheme = IdealRowSemStamp(
        n_cells,
        args.n_valid,
        experiment_key(args.key_seed),
        min_margin=args.min_margin,
        mask_mode=args.mask_mode,
    )

    documents, summary = generate_row_semstamp(
        model,
        tokenizer,
        prompts,
        initial_sentences,
        initial_cells,
        cell_map,
        embedder,
        scheme,
        n_sentences=args.sentences,
        seed=args.seed,
        max_attempts=args.max_attempts,
        on_exhaustion=args.on_exhaustion,
        max_new_tokens=args.max_new_tokens,
        max_context_tokens=args.max_context_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_rows=args.max_rows,
        max_completion_retry_rounds=args.max_completion_retry_rounds,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.generator}_semstamp-row-ideal-prf.json"
    payload = {
        "schema_version": 1,
        "experiment": "exp16_semstamp_generate",
        "config": vars(args),
        "source": source,
        "cell_map": {
            "type": "lsh",
            "bits": args.lsh_bits,
            "n_cells": n_cells,
            "seed": args.map_seed,
        },
        "baseline_fidelity": scheme.metadata(),
        "key_seed": args.key_seed,
        "summary": summary,
        "prompts": prompts,
        "references": references,
        "documents": documents,
    }
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {path}")
    if getattr(model, "is_vllm_generator", False):
        model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
