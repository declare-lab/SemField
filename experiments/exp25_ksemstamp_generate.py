"""Generate and prepare a calibrated k-SemStamp baseline.

k-SemStamp differs from SemStamp only in its semantic partition: sentence
embeddings are assigned to the closest of K domain-fitted k-means centroids.
The previous cluster selects a keyed fixed-size set of valid next clusters.
Generation reuses experiment 16's rejection sampler and detection reuses the
exact row-conditioned null in experiment 17.

Two modes are provided. ``generate`` creates watermarked generations and
saves the centroids. ``encode-null`` maps an existing matched experiment-11
null generation through those exact centroids, making it suitable for the
experiment-17 evaluator.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.generation_utils import build_prompts
from experiments.exp16_semstamp_generate import (
    experiment_key,
    generate_row_semstamp,
)
from semfield.cells import KMeansCells
from semfield.corpora import load_documents, split_sentences
from semfield.semstamp_row import IDEAL_PRF_MODE, IdealRowSemStamp


class MarginKMeansCells(KMeansCells):
    """Cosine k-means assignment with the paper's cluster-boundary margin."""

    def similarities(self, embeddings) -> np.ndarray:
        if self.centroids is None:
            raise RuntimeError("centroids are not fitted")
        values = np.atleast_2d(np.asarray(embeddings, dtype=float))
        values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
        centers = self.centroids / np.maximum(
            np.linalg.norm(self.centroids, axis=1, keepdims=True), 1e-12
        )
        return values @ centers.T

    def __call__(self, embeddings) -> np.ndarray:
        return np.argmax(self.similarities(embeddings), axis=1).astype(int)

    def margins(self, embeddings) -> np.ndarray:
        """Return second-best distance minus best distance.

        For cosine distance this equals best cosine similarity minus the
        second-best similarity, exactly matching Eq. (1) in k-SemStamp.
        """
        similarities = self.similarities(embeddings)
        top_two = np.partition(similarities, -2, axis=1)[:, -2:]
        return top_two.max(axis=1) - top_two.min(axis=1)


def _fit_sentences(corpus_dir: str, max_documents: int,
                   max_sentences: int) -> tuple[list[str], str]:
    documents, source = load_documents(
        corpus_dir=corpus_dir, max_documents=max_documents
    )
    sentences = [sentence for document in documents
                 for sentence in split_sentences(document)]
    if max_sentences > 0:
        sentences = sentences[:max_sentences]
    if len(sentences) < 8:
        raise ValueError("too few fitting sentences for k-SemStamp")
    return sentences, source


def _metadata(cell_map: MarginKMeansCells, args, fit_source: str) -> dict:
    assert cell_map.centroids is not None
    return {
        "type": "kmeans",
        "family": "kmeans",
        "semstamp_name": "k-SemStamp cosine clusters",
        "n_cells": int(cell_map.n_cells),
        "clusters": int(cell_map.n_cells),
        "seed": int(args.map_seed),
        "fit_sample": int(args.fit_sample),
        "fit_source": fit_source,
        "margin_definition": "best_cosine_similarity-minus-second_best",
        "centroids": cell_map.centroids.tolist(),
    }


def _load_map(reference: Path) -> tuple[dict, MarginKMeansCells]:
    payload = json.loads(reference.read_text(encoding="utf-8"))
    metadata = payload.get("cell_map", {})
    centers = np.asarray(metadata.get("centroids"), dtype=float)
    if centers.ndim != 2 or centers.shape[0] < 2:
        raise ValueError("reference does not contain valid k-means centroids")
    cell_map = MarginKMeansCells(
        n_clusters=centers.shape[0], seed=int(metadata.get("seed", 0))
    )
    cell_map.centroids = centers
    return payload, cell_map


def encode_null(args) -> int:
    reference, cell_map = _load_map(Path(args.centroid_reference))
    source_path = Path(args.null_input)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("documents"), list):
        raise ValueError("--null-input must be a generation JSON with documents")

    from semfield.embedders import embed, load_embedder, resolve_devices

    embedder_name = args.embedder or reference.get("config", {}).get("embedder")
    devices = resolve_devices(args.device)
    encoder = load_embedder(embedder_name, device=devices[0])
    sentences = [str(sentence) for document in payload["documents"]
                 for sentence in document.get("sentences", ())]
    vectors = embed(encoder, sentences, devices=devices, model_name=embedder_name)
    cells = cell_map(vectors).tolist()
    cursor = 0
    for document in payload["documents"]:
        count = len(document.get("sentences", ()))
        document["cells"] = [int(value) for value in cells[cursor:cursor + count]]
        cursor += count
    payload["cell_map"] = reference["cell_map"]
    payload["ksemstamp_null_reencoding"] = {
        "source": str(source_path),
        "centroid_reference": str(args.centroid_reference),
        "embedder": embedder_name,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}")
    return 0


def generate(args) -> int:
    if not 0 < args.n_valid < args.clusters:
        raise ValueError("--n-valid must lie strictly between 0 and --clusters")

    from semfield.embedders import embed, load_embedder, resolve_devices
    from semfield.generation import load_generator

    prompts, references, source = build_prompts(
        args.corpus_dir, args.documents, args.prompt_sentences
    )
    initial_sentences = [split_sentences(prompt)[-1] for prompt in prompts]
    fit_sentences, fit_source = _fit_sentences(
        args.fit_corpus_dir or args.corpus_dir,
        args.fit_documents,
        args.fit_sentences,
    )
    devices = resolve_devices(args.device)
    encoder = load_embedder(args.embedder, device=devices[0])

    def embedder(sentences):
        return embed(encoder, list(sentences), devices=devices,
                     model_name=args.embedder)

    print(f"fitting {args.clusters} clusters on {len(fit_sentences)} sentences")
    fit_vectors = embedder(fit_sentences)
    cell_map = MarginKMeansCells(
        n_clusters=args.clusters, seed=args.map_seed,
        fit_sample=args.fit_sample,
    ).fit(fit_vectors)
    initial_cells = cell_map(embedder(initial_sentences)).astype(int).tolist()
    scheme = IdealRowSemStamp(
        args.clusters, args.n_valid, experiment_key(args.key_seed),
        min_margin=args.min_margin, mask_mode=IDEAL_PRF_MODE,
    )
    model, tokenizer = load_generator(args.generator, device=devices[0])
    documents, summary = generate_row_semstamp(
        model, tokenizer, prompts, initial_sentences, initial_cells,
        cell_map, embedder, scheme,
        n_sentences=args.sentences, seed=args.seed,
        max_attempts=args.max_attempts, on_exhaustion=args.on_exhaustion,
        max_new_tokens=args.max_new_tokens,
        max_context_tokens=args.max_context_tokens,
        temperature=args.temperature, top_p=args.top_p,
        max_rows=args.max_rows,
        max_completion_retry_rounds=args.max_completion_retry_rounds,
    )
    summary["scheme"] = "ksemstamp"
    payload = {
        "schema_version": 1,
        # Keep the established evaluator contract; cell_map.type distinguishes it.
        "experiment": "exp16_semstamp_generate",
        "method": "ksemstamp",
        "config": vars(args),
        "source": source,
        "cell_map": _metadata(cell_map, args, fit_source),
        "baseline_fidelity": {
            **scheme.metadata(),
            "semantic_partition": "domain-fitted cosine k-means",
            "paper": "Hou et al., Findings of ACL 2024",
        },
        "key_seed": args.key_seed,
        "summary": summary,
        "prompts": prompts,
        "references": references,
        "documents": documents,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["generate", "encode-null"],
                        default="generate")
    parser.add_argument("--corpus-dir", default="data/c4-realnewslike")
    parser.add_argument("--fit-corpus-dir")
    parser.add_argument("--generator", default="opt-1.3b")
    parser.add_argument("--embedder", default="mpnet")
    parser.add_argument("--documents", type=int, default=100)
    parser.add_argument("--prompt-sentences", type=int, default=3)
    parser.add_argument("--sentences", type=int, default=40)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--n-valid", type=int, default=2)
    parser.add_argument("--min-margin", type=float, default=0.02)
    parser.add_argument("--fit-documents", type=int, default=1000)
    parser.add_argument("--fit-sentences", type=int, default=100000)
    parser.add_argument("--fit-sample", type=int, default=100000)
    parser.add_argument("--map-seed", type=int, default=0)
    parser.add_argument("--key-seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=32)
    parser.add_argument("--on-exhaustion", choices=["stop", "emit-last"],
                        default="emit-last")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--max-context-tokens", type=int, default=512)
    parser.add_argument("--max-rows", type=int, default=64)
    parser.add_argument("--max-completion-retry-rounds", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--null-input")
    parser.add_argument("--centroid-reference")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.mode == "encode-null":
        if not args.null_input or not args.centroid_reference:
            parser.error("encode-null requires --null-input and --centroid-reference")
        return encode_null(args)
    return generate(args)


if __name__ == "__main__":
    raise SystemExit(main())
