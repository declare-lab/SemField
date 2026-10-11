"""Generate resumable sentence-level paraphrase artifacts for watermark attacks.

The output format is accepted directly by the SemField robustness
evaluators.  Passing ``--cell-reference LABEL=GENERATION.json`` additionally
stores cells computed with that generation's exact semantic map, which is
required by the model-free SemStamp evaluator.

The ``bigram`` selector follows SemStamp's attack: generate multiple beams and
choose the candidate with the least bigram overlap, subject to losing at most
``--max-similarity-drop`` of the first beam's semantic similarity.  Similarity
can be BERTScore (the paper protocol) or a cached sentence embedder.

Examples::

    CUDA_VISIBLE_DEVICES=0 python experiments/exp24_generate_paraphrases.py \
      --model tuner007/pegasus_paraphrase --preset pegasus \
      --input semfield-wm=wm.json --input semfield-null=null.json \
      --selector bigram --candidates 25 --similarity-backend bertscore \
      --output-dir runs/paraphrase/pegasus-bigram --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from semfield.generation import default_precision_for_model  # noqa: E402


LABEL_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _named_path(value: str, option: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"{option} must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not LABEL_RE.fullmatch(label):
        raise argparse.ArgumentTypeError(f"invalid label {label!r}")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"missing input: {path}")
    return label, path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
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


def _load_generation(path: Path, max_documents: int | None) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    documents = payload.get("documents") if isinstance(payload, dict) else None
    if not isinstance(documents, list):
        raise ValueError(f"{path} is not a generation JSON with documents")
    if max_documents is not None:
        documents = documents[:max_documents]
    cleaned = []
    for position, document in enumerate(documents):
        if not isinstance(document, dict):
            raise ValueError(f"{path}: document {position} is not an object")
        sentences = document.get("sentences")
        if not isinstance(sentences, list) or any(
            not isinstance(sentence, str) for sentence in sentences
        ):
            raise ValueError(f"{path}: document {position} lacks sentences")
        cleaned.append({
            "document_id": int(document.get("document_id", position)),
            "sentences": [sentence.strip() for sentence in sentences
                          if sentence.strip()],
        })
    return cleaned


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def _bigram_overlap(source: str, candidate: str) -> int:
    source_words = _words(source)
    candidate_words = _words(candidate)
    source_bigrams = set(zip(source_words, source_words[1:]))
    candidate_bigrams = set(zip(candidate_words, candidate_words[1:]))
    return len(source_bigrams.intersection(candidate_bigrams))


class SimilarityScorer:
    def __init__(self, backend: str, device: str, embedder: str,
                 bertscore_model: str | None):
        self.backend = backend
        self.device = device
        self.embedder_name = embedder
        self.bertscore_model = bertscore_model
        self.encoder = None

    def score(self, source: str, candidates: Sequence[str]) -> np.ndarray:
        if self.backend == "none":
            return np.ones(len(candidates), dtype=float)
        if self.backend == "bertscore":
            try:
                from bert_score import score as bert_score
            except ImportError as exc:
                raise RuntimeError(
                    "BERTScore backend requested; install bert-score with "
                    "`uv pip install --python ~/.venv/bin/python bert-score`"
                ) from exc
            _, _, f1 = bert_score(
                list(candidates), [source] * len(candidates), lang="en",
                model_type=self.bertscore_model, device=self.device,
                verbose=False, rescale_with_baseline=False,
            )
            return f1.detach().float().cpu().numpy().astype(float)

        from semfield.embedders import embed, load_embedder, resolve_devices
        if self.encoder is None:
            resolved = resolve_devices(self.device)[0]
            self.encoder = load_embedder(self.embedder_name, device=resolved)
        vectors = embed(
            self.encoder, [source, *candidates], devices=[self.device],
            model_name=self.embedder_name,
        ).astype(float)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True),
                              1e-12)
        return vectors[1:] @ vectors[0]


def _select(source: str, candidates: Sequence[str], selector: str,
            scorer: SimilarityScorer,
            max_similarity_drop: float) -> tuple[str, dict]:
    cleaned = [candidate.strip() or source for candidate in candidates]
    similarities = scorer.score(source, cleaned)
    overlaps = np.asarray([
        _bigram_overlap(source, candidate) for candidate in cleaned
    ], dtype=int)
    if selector == "first":
        selected = 0
    else:
        reference = float(similarities[0])
        threshold = reference * (1.0 - max_similarity_drop)
        eligible = np.flatnonzero(similarities >= threshold)
        if eligible.size == 0:
            eligible = np.asarray([0])
        selected = int(min(
            eligible,
            key=lambda index: (int(overlaps[index]),
                               -float(similarities[index]), int(index)),
        ))
    return cleaned[selected], {
        "selected_candidate": selected,
        "candidate_count": len(cleaned),
        "bigram_overlap": int(overlaps[selected]),
        "semantic_similarity": float(similarities[selected]),
        "first_beam_similarity": float(similarities[0]),
    }


def _dtype(precision: str, device: str):
    import torch
    if not device.startswith("cuda") or precision == "fp32":
        return torch.float32
    if precision == "bf16":
        return torch.bfloat16
    return torch.float16


def _prompt(sentence: str, preset: str, template: str | None) -> str:
    if template is not None:
        return template.format(sentence=sentence)
    if preset == "parrot":
        return f"paraphrase: {sentence}"
    if preset == "generic":
        return f"Paraphrase this sentence while preserving its meaning: {sentence}"
    return sentence


def _generate_groups(model, tokenizer, sentences: Sequence[str], *,
                     preset: str, template: str | None, candidates: int,
                     max_input_tokens: int, max_new_tokens: int,
                     device: str) -> list[list[str]]:
    prompts = [_prompt(sentence, preset, template) for sentence in sentences]
    model_input_limit = getattr(model.config, "max_position_embeddings", None)
    effective_input_limit = max_input_tokens
    if isinstance(model_input_limit, int) and model_input_limit > 0:
        effective_input_limit = min(effective_input_limit, model_input_limit)
    encoded = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=effective_input_limit,
    ).to(device)
    generation_kwargs = {
        "num_beams": max(candidates, 1),
        "num_return_sequences": max(candidates, 1),
        "do_sample": False,
        "early_stopping": True,
    }
    # The released Pegasus paraphrase checkpoint has a task-specific decoder
    # limit of 60 tokens. Driving beam search beyond it can produce invalid
    # beam indices in recent Transformers versions. Respect that checkpoint
    # limit; other models continue to use the caller.s new-token budget.
    configured_max = getattr(model.generation_config, "max_length", None)
    if preset == "pegasus" and isinstance(configured_max, int) and configured_max > 0:
        generation_kwargs["max_length"] = min(max_new_tokens, configured_max)
    else:
        generation_kwargs["max_new_tokens"] = max_new_tokens
    with torch.inference_mode():
        outputs = model.generate(**encoded, **generation_kwargs)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [decoded[index * candidates:(index + 1) * candidates]
            for index in range(len(sentences))]


def _cell_encoder(reference_path: Path, device: str, embedder: str | None):
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    metadata = payload.get("cell_map", {})
    if metadata.get("type") == "kmeans":
        from experiments.exp25_ksemstamp_generate import MarginKMeansCells
        from semfield.embedders import embed, load_embedder, resolve_devices

        centers = np.asarray(metadata.get("centroids"), dtype=float)
        if centers.ndim != 2 or centers.shape[0] < 2:
            raise ValueError("k-SemStamp reference has no valid centroids")
        model_name = embedder or payload.get("config", {}).get("embedder")
        devices = resolve_devices(device)
        encoder = load_embedder(model_name, device=devices[0])
        cell_map = MarginKMeansCells(n_clusters=centers.shape[0])
        cell_map.centroids = centers

        def encode(sentences):
            if not sentences:
                return ()
            vectors = embed(encoder, list(sentences), devices=devices,
                            model_name=model_name)
            return tuple(int(value) for value in cell_map(vectors))

        return encode
    from experiments.semantic_cells import TextCellEncoder
    return TextCellEncoder(payload, embedder, device)


def _contract(args, label: str, source: Path, cell_reference: Path | None) -> dict:
    return {
        "label": label,
        "source": str(source),
        "source_sha256": _sha256(source),
        "cell_reference": None if cell_reference is None else str(cell_reference),
        "cell_reference_sha256": (
            None if cell_reference is None else _sha256(cell_reference)
        ),
        "model": args.model,
        "preset": args.preset,
        "prompt_template": args.prompt_template,
        "selector": args.selector,
        "candidates": args.candidates,
        "similarity_backend": args.similarity_backend,
        "similarity_embedder": args.similarity_embedder,
        "bertscore_model": args.bertscore_model,
        "max_similarity_drop": args.max_similarity_drop,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "max_documents": args.max_documents,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True,
                        metavar="LABEL=PATH")
    parser.add_argument("--cell-reference", action="append", default=[],
                        metavar="LABEL=GENERATION_JSON")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--preset", choices=["pegasus", "parrot", "generic"],
                        default="pegasus")
    parser.add_argument("--prompt-template", default=None,
                        help="custom template containing {sentence}")
    parser.add_argument("--selector", choices=["first", "bigram"],
                        default="first")
    parser.add_argument("--candidates", type=int, default=25)
    parser.add_argument("--similarity-backend",
                        choices=["bertscore", "embedding", "none"],
                        default="embedding")
    parser.add_argument("--similarity-embedder", default="mpnet")
    parser.add_argument("--bertscore-model", default=None)
    parser.add_argument("--max-similarity-drop", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"],
                        default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.precision is None:
        args.precision = default_precision_for_model(args.model)

    if args.candidates < 1 or args.batch_size < 1:
        parser.error("--candidates and --batch-size must be positive")
    if not 0.0 <= args.max_similarity_drop < 1.0:
        parser.error("--max-similarity-drop must lie in [0, 1)")
    if args.selector == "bigram" and args.candidates < 2:
        parser.error("bigram selection requires at least two candidates")
    if args.prompt_template is not None and "{sentence}" not in args.prompt_template:
        parser.error("--prompt-template must contain {sentence}")

    inputs = dict(_named_path(value, "--input") for value in args.input)
    if len(inputs) != len(args.input):
        parser.error("duplicate --input label")
    references = dict(
        _named_path(value, "--cell-reference") for value in args.cell_reference
    )
    unknown = sorted(set(references).difference(inputs))
    if unknown:
        parser.error(f"cell references have no matching input: {unknown}")

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    local_only = not args.allow_download
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=local_only
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model, local_files_only=local_only,
        dtype=_dtype(args.precision, args.device),
        attn_implementation="eager",
    ).to(args.device).eval()
    scorer = SimilarityScorer(
        args.similarity_backend, args.device, args.similarity_embedder,
        args.bertscore_model,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    for label, source in inputs.items():
        documents = _load_generation(source, args.max_documents)
        reference = references.get(label)
        contract = _contract(args, label, source, reference)
        output_path = output_dir / f"{label}.json"
        existing: dict = {}
        completed: dict[int, dict] = {}
        if output_path.exists():
            if not args.resume:
                parser.error(f"{output_path} exists; pass --resume")
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            if existing.get("contract") != contract:
                parser.error(f"resume contract differs for {output_path}")
            completed = {
                int(document["document_id"]): document
                for document in existing.get("documents", [])
            }
        encoder = _cell_encoder(reference, args.device, None) if reference else None
        result = {
            "schema_version": 1,
            "experiment": "exp24_generate_paraphrases",
            "contract": contract,
            "status": "partial",
            "documents": list(completed.values()),
        }
        from tqdm.auto import tqdm
        progress = tqdm(total=len(documents), initial=len(completed),
                        desc=label, unit="doc", dynamic_ncols=True)
        for position, document in enumerate(documents):
            document_id = int(document["document_id"])
            if document_id in completed:
                continue
            source_sentences = document["sentences"]
            paraphrases: list[str] = []
            selections: list[dict] = []
            with torch.inference_mode():
                for start in range(0, len(source_sentences), args.batch_size):
                    chunk = source_sentences[start:start + args.batch_size]
                    groups = _generate_groups(
                        model, tokenizer, chunk, preset=args.preset,
                        template=args.prompt_template,
                        candidates=args.candidates,
                        max_input_tokens=args.max_input_tokens,
                        max_new_tokens=args.max_new_tokens,
                        device=args.device,
                    )
                    for sentence, candidates in zip(chunk, groups):
                        selected, metadata = _select(
                            sentence, candidates, args.selector, scorer,
                            args.max_similarity_drop,
                        )
                        paraphrases.append(selected)
                        selections.append(metadata)
            row = {
                "document_id": document_id,
                "sentences": paraphrases,
                "source_sentence_count": len(source_sentences),
                "selection": selections,
            }
            if encoder is not None:
                row["cells"] = list(encoder(paraphrases))
            completed[document_id] = row
            result["documents"] = [
                completed[key] for key in sorted(completed)
            ]
            _atomic_json(output_path, result)
            progress.update(1)
        progress.close()
        result["status"] = "complete"
        result["documents"] = [completed[key] for key in sorted(completed)]
        _atomic_json(output_path, result)
        print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
