"""Experiment 20: calibrated sentence-level robustness of SemField.

This evaluator consumes the watermarked and matched null JSON artifacts written
by :mod:`experiments.exp19_semfield_generate`.  Unlike the cell-path attacks in
experiments 12 and 17, every attack here is applied to sentence *text*.  The
exact encoder and public random-feature map recorded by experiment 19 are then
reconstructed, and all distinct attacked sentences are embedded in one cached
pass.

For fixed attacked feature rows ``phi_t`` and a global field vector ``w`` the
ordinary statistic is

    S = w.T @ sum_t phi_t,
    V = ||sum_t phi_t||^2,
    Z = S / sqrt(V).

Thus ``Z`` is exactly standard normal conditional on key-independent null text
in the ideal Gaussian-field model. Detection uses the analytic Gaussian
p-value under the tested key.

Anchored artifacts opt into a continuous global/local field.  Their whole
document statistic is

    [sqrt(rho) w0.T sum(phi_t) + sqrt(1-rho) sum_t u_t.T phi_t]
    / sqrt(rho ||sum(phi_t)||^2 + (1-rho) sum_t ||phi_t||^2),

where ``u_t`` is derived from the same shared key with a position domain
separator.  The local component follows ordinary monotone positions during a
scan; no binary code or Hamming distance is used.

The default structural attacks require no language model.  ``paraphrase``
accepts externally produced sentence text and deliberately requires a separate
null-paraphrase artifact before estimating attacked FPR.

Example::

    python experiments/exp20_semfield_robustness.py \
      --watermarked generations_semfield/opt-1.3b_semfield.json \
      --null generations_semfield/opt-1.3b_none.json \
      --attacks none reorder delete crop dilute substitute split merge \
      --strengths 0.10 0.25 0.50 0.75 \
      --out results_exp20.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Iterable, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semfield._hf_env import EMBEDDING_CACHE  # noqa: E402
from semfield.semfield import (
    block_positions,  # noqa: E402
    anchored_field_scores,
    anchored_field_statistic,
    protected_anchored_field_scores,
    protected_anchored_field_statistic,
    BENCHMARK_KEY_DOMAIN,
    aggregate_features,
    benchmark_ensemble_key,
    decorrelated_field_scores,
    decorrelated_field_statistic,
    field_statistic,
    innovation_span_scores,
    innovation_span_zscores,
    innovation_span_statistic,
    two_sided_field_statistic,
    keyed_gaussian,
    multiscale_window_features,
    SoftSignSemanticField,
    AERSemanticField,
)


from semfield.semfield_kernels import feature_map_from_metadata  # noqa: E402
from semfield.semfield_moment import moment_field_statistic  # noqa: E402

ATTACKS = (
    "none",
    "reorder",
    "delete",
    "insert",
    "crop",
    "dilute",
    "substitute",
    "split",
    "merge",
    "paraphrase",
)

EXP19_SCHEMA_VERSION = 1
KEY_DOMAIN = BENCHMARK_KEY_DOMAIN
_DETECTOR_FIELDS = (
    "score_sum",
    "null_variance",
    "z_score",
    "analytic_pvalue",
    "n_sentences",
    "mean_pair_kernel",
    "design_effect",
)


@dataclass(frozen=True)
class Condition:
    attack: str
    strength: float

    @property
    def slug(self) -> str:
        value = f"{self.attack}-{self.strength:g}"
        return re.sub(r"[^A-Za-z0-9_.-]+", "-", value)


@dataclass(frozen=True)
class AttackedText:
    """Attacked sentences with provenance back to clean sentence indices.

    ``source_groups[j]`` names every clean sentence represented by attacked
    sentence ``j``.  Inserted material is ``None``; splitting produces two
    singleton groups with the same source; merging produces a two-index group.
    ``watermarked_span`` is populated only when dilution preserves a contiguous
    copy of the clean passage.
    """

    sentences: tuple[str, ...]
    source_groups: tuple[tuple[int, ...] | None, ...]
    watermarked_span: tuple[int, int] | None
    details: dict[str, int | float | bool | str | None]


def experiment_key(seed: int) -> bytes:
    """Reconstruct experiment 19's experimental live key."""

    return benchmark_ensemble_key(seed, 0)




def _condition_seed(parts: Sequence[object]) -> int:
    digest = hashlib.blake2b(digest_size=8)
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\x00")
    return int.from_bytes(digest.digest(), "big")


def _sentences(document: dict, label: str) -> tuple[str, ...]:
    values = document.get("sentences")
    if not isinstance(values, list) or any(not isinstance(x, str) for x in values):
        raise ValueError(f"{label} must contain a list of sentence strings")
    if any(not value.strip() for value in values):
        raise ValueError(f"{label} contains an empty sentence")
    return tuple(value.strip() for value in values)


def _load_payload(path: str | Path, label: str) -> tuple[Path, dict]:
    source = Path(path).resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("documents"), list):
        raise ValueError(f"{source} is not an exp19 generation JSON")
    if value.get("experiment") != "exp19_semfield_generate":
        raise ValueError(
            f"{label} input has experiment={value.get('experiment')!r}; "
            "expected exp19_semfield_generate"
        )
    return source, value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _generation_sentence_fingerprint(documents: Sequence[dict]) -> str:
    """Reproduce exp19's document-aware sidecar sentence fingerprint."""

    digest = hashlib.blake2b(digest_size=16)
    for position, document in enumerate(documents):
        document_id = int(document.get("document_id", position))
        digest.update(document_id.to_bytes(8, "big", signed=True))
        sentences = document.get("sentences", ())
        digest.update(len(sentences).to_bytes(8, "big"))
        for sentence in sentences:
            digest.update(str(sentence).encode("utf-8", "ignore"))
            digest.update(b"\x00")
    return digest.hexdigest()


def _compare_stored_detector(
    stored: object,
    computed: object,
    *,
    label: str,
) -> float:
    """Require an exp19 detector record to match a sidecar recomputation."""

    if not isinstance(stored, dict):
        raise ValueError(f"{label} is missing its stored detector")
    values = asdict(computed)
    missing = [name for name in _DETECTOR_FIELDS if name not in stored]
    if missing:
        raise ValueError(f"{label} detector is missing fields {missing}")

    maximum_delta = 0.0
    for name in _DETECTOR_FIELDS:
        expected = values[name]
        observed = stored[name]
        if expected is None:
            if observed is not None:
                raise ValueError(
                    f"{label} detector {name}={observed!r}; expected None"
                )
            continue
        if name == "n_sentences":
            try:
                matches = (
                    not isinstance(observed, bool)
                    and int(observed) == int(expected)
                )
            except (TypeError, ValueError):
                matches = False
            if not matches:
                raise ValueError(
                    f"{label} detector {name}={observed!r}; expected {expected!r}"
                )
            continue
        try:
            observed_number = float(observed)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{label} detector {name} is not numeric"
            ) from exc
        expected_number = float(expected)
        if not math.isfinite(observed_number) or not math.isclose(
            observed_number, expected_number, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(
                f"{label} detector {name}={observed_number!r}; "
                f"sidecar recomputation gives {expected_number!r}"
            )
        maximum_delta = max(
            maximum_delta, abs(observed_number - expected_number)
        )
    return float(maximum_delta)


def validate_feature_sidecar(
    generation_path: Path,
    payload: dict,
    *,
    label: str,
) -> dict[str, object]:
    """Validate exp19's sidecar and clean detectors without loading a model."""

    if payload.get("schema_version") != EXP19_SCHEMA_VERSION:
        raise ValueError(
            f"{label} input has schema_version={payload.get('schema_version')!r}; "
            f"expected {EXP19_SCHEMA_VERSION}"
        )
    documents = payload.get("documents")
    metadata = payload.get("feature_sidecar")
    feature_metadata = payload.get("feature_map")
    field_metadata = payload.get("field")
    if not isinstance(documents, list):
        raise ValueError(f"{label} input has no document list")
    if not isinstance(metadata, dict):
        raise ValueError(f"{label} input is missing feature_sidecar metadata")
    if not isinstance(feature_metadata, dict):
        raise ValueError(f"{label} input is missing feature_map metadata")
    if not isinstance(field_metadata, dict):
        raise ValueError(f"{label} input is missing field metadata")
    if field_metadata.get("key_derivation_domain") != KEY_DOMAIN:
        raise ValueError(
            f"{label} input key_derivation_domain does not match exp19"
        )

    required = (
        "path",
        "format",
        "sha256",
        "features_key",
        "offsets_key",
        "document_ids_key",
        "dtype",
        "shape",
        "documents",
        "sentence_fingerprint_blake2b128",
    )
    missing = [name for name in required if name not in metadata]
    if missing:
        raise ValueError(f"{label} feature_sidecar is missing fields {missing}")
    if metadata["format"] != "numpy-npz":
        raise ValueError(f"{label} feature_sidecar format must be 'numpy-npz'")
    key_names = {
        "features": metadata["features_key"],
        "offsets": metadata["offsets_key"],
        "document_ids": metadata["document_ids_key"],
    }
    if key_names != {
        "features": "features",
        "offsets": "offsets",
        "document_ids": "document_ids",
    }:
        raise ValueError(f"{label} feature_sidecar has an unknown NPZ schema")

    sidecar_path = (generation_path.parent / str(metadata["path"])).resolve()
    computed_sha256 = _sha256(sidecar_path)
    recorded_sha256 = str(metadata["sha256"]).lower()
    if computed_sha256 != recorded_sha256:
        raise ValueError(
            f"{label} feature_sidecar SHA-256 mismatch: "
            f"recorded {recorded_sha256}, computed {computed_sha256}"
        )
    with np.load(sidecar_path, allow_pickle=False) as archive:
        if set(archive.files) != set(key_names.values()):
            raise ValueError(
                f"{label} feature_sidecar NPZ members do not match its schema"
            )
        features = np.asarray(archive[str(key_names["features"])]).copy()
        offsets = np.asarray(archive[str(key_names["offsets"])]).copy()
        document_ids = np.asarray(
            archive[str(key_names["document_ids"])]
        ).copy()

    if features.ndim != 2 or features.dtype != np.dtype("float32"):
        raise ValueError(
            f"{label} sidecar features must be a two-dimensional float32 matrix"
        )
    if np.any(~np.isfinite(features)):
        raise ValueError(f"{label} sidecar features contain non-finite values")
    if (
        offsets.dtype != np.dtype("int64")
        or offsets.shape != (len(documents) + 1,)
    ):
        raise ValueError(f"{label} sidecar offsets have the wrong dtype or shape")
    if (
        document_ids.dtype != np.dtype("int64")
        or document_ids.shape != (len(documents),)
    ):
        raise ValueError(
            f"{label} sidecar document_ids have the wrong dtype or shape"
        )
    if (
        offsets[0] != 0
        or np.any(np.diff(offsets) < 0)
        or offsets[-1] != features.shape[0]
    ):
        raise ValueError(f"{label} sidecar offsets do not partition feature rows")

    expected_shape = [int(value) for value in features.shape]
    if (
        metadata["dtype"] != str(features.dtype)
        or metadata["shape"] != expected_shape
    ):
        raise ValueError(
            f"{label} sidecar dtype/shape metadata does not match the NPZ"
        )
    if int(metadata["documents"]) != len(documents):
        raise ValueError(f"{label} sidecar document count does not match the JSON")
    if features.shape[1] != int(feature_metadata["n_features"]):
        raise ValueError(f"{label} sidecar width does not match feature_map")
    expected_ids = np.asarray(_document_ids(documents), dtype=np.int64)
    if not np.array_equal(document_ids, expected_ids):
        raise ValueError(f"{label} sidecar document ids do not match the JSON")
    fingerprint = _generation_sentence_fingerprint(documents)
    if metadata["sentence_fingerprint_blake2b128"] != fingerprint:
        raise ValueError(f"{label} sidecar sentence fingerprint does not match")

    key = experiment_key(int(payload["key_seed"]))
    channel = int(field_metadata.get("channel", 0))
    field_structure = str(field_metadata.get("structure", "global"))
    field_rho = float(field_metadata.get("rho", 1.0))
    field_blocks = field_metadata.get("blocks")
    if field_blocks is not None:
        field_blocks = int(field_blocks)
        if field_blocks <= 0:
            raise ValueError(f"{label} has an invalid field blocks")
    field_fusion = field_metadata.get("fusion")
    field_type = str(field_metadata.get("type", ""))
    if field_structure not in {"global", "anchored", "aer"}:
        raise ValueError(f"{label} has an unknown field structure {field_structure!r}")
    if field_type == "aer-relative-block-gaussian-v1" and field_structure != "aer":
        raise ValueError(f"{label} AER metadata requires field_structure='aer'")
    if not math.isfinite(field_rho) or not 0.0 <= field_rho <= 1.0:
        raise ValueError(f"{label} has an invalid field rho")
    if field_fusion not in {None, "max-absolute-global-anchored-v1"}:
        raise ValueError(f"{label} has an unknown detector fusion {field_fusion!r}")
    if field_fusion is not None and field_structure != "anchored":
        raise ValueError(f"{label} detector fusion requires an anchored field")
    is_decorrelated = (
        field_type == "sentence-decorrelated-keyed-gaussian-v1"
    )
    is_innovation = field_type == "innovation-span-keyed-gaussian-v1"
    if is_innovation and field_structure != "global":
        raise ValueError(f"{label} innovation fields must be global")
    if is_decorrelated and field_structure != "global":
        raise ValueError(f"{label} decorrelated fields must be global")
    soft_sign_channels = field_metadata.get("soft_sign_channels")
    soft_sign_beta = field_metadata.get("soft_sign_beta")
    if field_type == "soft-sign-semantic-field-v1":
        if field_structure != "global":
            raise ValueError(f"{label} soft-sign fields must be global")
        try:
            soft_sign_channels = int(soft_sign_channels)
            soft_sign_beta = float(soft_sign_beta)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} has invalid soft-sign metadata") from exc
        if soft_sign_channels <= 0 or not math.isfinite(soft_sign_beta) or soft_sign_beta <= 0.0:
            raise ValueError(f"{label} has invalid soft-sign metadata")
    maximum_delta = 0.0
    for position, document in enumerate(documents):
        document_id = int(document.get("document_id", position))
        sentences = _sentences(document, f"{label} document {document_id}")
        start, stop = int(offsets[position]), int(offsets[position + 1])
        if document.get("feature_span") != [start, stop]:
            raise ValueError(
                f"{label} document {document_id} feature_span does not match offsets"
            )
        if stop - start != len(sentences):
            raise ValueError(
                f"{label} document {document_id} feature rows do not align with sentences"
            )
        weights = keyed_gaussian(key, features.shape[1], channel=channel)
        aggregation = str(field_metadata.get("aggregation", "uniform"))
        two_sided = field_metadata.get("test_tail") == "two-sided-absolute"
        moment = field_metadata.get("moment", {})
        if is_innovation:
            computed = innovation_span_statistic(
                features[start:stop], weights, two_sided=two_sided
            )
        elif is_decorrelated:
            computed = decorrelated_field_statistic(
                features[start:stop], weights, two_sided=two_sided
            )
        elif field_type == "soft-sign-semantic-field-v1":
            computed = SoftSignSemanticField(
                features.shape[1], key,
                channels=soft_sign_channels, beta=soft_sign_beta,
                channel=channel,
            ).detect_features(
                features[start:stop], two_sided=two_sided
            )
        elif field_type == "aer-relative-block-gaussian-v1":
            computed = AERSemanticField(
                type("FeatureMap", (), {"n_features": features.shape[1]})(),
                key, blocks=int(field_blocks or 4),
                channel=channel,
                lambda_grid=tuple(field_metadata.get("aer_lambda_grid") or (0.0,.5,1.0,2.0,3.0)),
            ).detect_features(features[start:stop])
        elif field_fusion == "max-absolute-global-anchored-v1":
            computed = protected_anchored_field_statistic(
                features[start:stop], key,
                rho=field_rho, channel=channel,
                positions=block_positions(stop - start, field_blocks),
            )
        elif field_structure == "anchored":
            if isinstance(moment, dict) and bool(moment.get("enabled", False)):
                raise ValueError("anchored fields cannot use moment-hybrid metadata")
            computed = anchored_field_statistic(
                features[start:stop], key,
                rho=field_rho, channel=channel, two_sided=two_sided,
                positions=block_positions(stop - start, field_blocks),
            )
        elif isinstance(moment, dict) and bool(moment.get("enabled", False)):
            moment_channel = int(moment.get("channel", channel + 1))
            moment_lambda = float(moment.get("lambda", 0.5))
            moment_weights = keyed_gaussian(
                key, features.shape[1], channel=moment_channel
            )
            computed = moment_field_statistic(
                features[start:stop], weights, moment_weights,
                moment_lambda=moment_lambda, two_sided=two_sided
            )
        else:
            statistic_fn = two_sided_field_statistic if two_sided else field_statistic
            computed = statistic_fn(
                features[start:stop], weights, aggregation=aggregation
            )
        maximum_delta = max(
            maximum_delta,
            _compare_stored_detector(
                document.get("detector"),
                computed,
                label=f"{label} document {document_id}",
            ),
        )

    return {
        "generation_json": str(generation_path),
        "sidecar_path": str(sidecar_path),
        "format": "numpy-npz",
        "schema_version": EXP19_SCHEMA_VERSION,
        "sha256": computed_sha256,
        "sha256_verified": True,
        "dtype": str(features.dtype),
        "shape": expected_shape,
        "documents": len(documents),
        "document_ids_verified": True,
        "feature_spans_verified": True,
        "sentence_fingerprint_blake2b128": fingerprint,
        "sentence_fingerprint_verified": True,
        "live_key_derivation_domain": KEY_DOMAIN,
        "field_channel": channel,
        "detector_validation": {
            "features": "exact exp19 sidecar rows",
            "documents_checked": len(documents),
            "fields": list(_DETECTOR_FIELDS),
            "relative_tolerance": 1e-12,
            "absolute_tolerance": 1e-12,
            "maximum_absolute_delta": maximum_delta,
            "all_match": True,
        },
        "model_loaded": False,
    }


def _validated_sidecar_rows(
    generation_path: Path,
    payload: dict,
    *,
    label: str,
) -> dict[tuple[int, int], tuple[str, np.ndarray]]:
    """Load exact occurrence rows after sidecar validation succeeds."""

    metadata = payload["feature_sidecar"]
    sidecar_path = (generation_path.parent / str(metadata["path"])).resolve()
    with np.load(sidecar_path, allow_pickle=False) as archive:
        features = np.asarray(archive[str(metadata["features_key"])]).copy()
        offsets = np.asarray(archive[str(metadata["offsets_key"])]).copy()

    rows: dict[tuple[int, int], tuple[str, np.ndarray]] = {}
    for position, document in enumerate(payload["documents"]):
        document_id = int(document.get("document_id", position))
        sentences = _sentences(document, f"{label} document {document_id}")
        start = int(offsets[position])
        for sentence_index, sentence in enumerate(sentences):
            row = np.asarray(
                features[start + sentence_index], dtype=np.float64
            )
            rows[(document_id, sentence_index)] = (sentence, row.copy())
    return rows


def _variant(payload: dict) -> str | None:
    summary = payload.get("summary", {})
    return summary.get("variant") or summary.get("scheme")


def _document_ids(documents: Sequence[dict]) -> list[int]:
    return [int(document.get("document_id", position))
            for position, document in enumerate(documents)]


def _critical_config(payload: dict) -> dict[str, object]:
    config = payload.get("config", {})
    fields = (
        "corpus_dir",
        "generator",
        "embedder",
        "documents",
        "prompt_sentences",
        "sentences",
        "temperature",
        "top_p",
        "max_new_tokens",
        "max_context_tokens",
        "max_retry_rounds",
        "max_embed_length",
        "embed_batch_size",
        "seed",
    )
    return {name: config.get(name) for name in fields}


def validate_matched_inputs(
    watermarked: dict,
    null: dict,
    max_documents: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Validate all comparison-critical generation and feature metadata."""

    wm_variant = str(_variant(watermarked) or "").lower()
    null_variant = str(_variant(null) or "").lower()
    if "semfield" not in wm_variant and wm_variant not in {"field", "rank"}:
        raise ValueError("watermarked input is not the SemField variant")
    if null_variant != "none":
        raise ValueError("matched null input must have variant/scheme 'none'")
    if watermarked.get("encoder") != null.get("encoder"):
        raise ValueError("watermarked and null inputs use different encoders")
    if watermarked.get("feature_map") != null.get("feature_map"):
        raise ValueError("watermarked and null inputs use different feature maps")
    # "test_tail" is excluded deliberately. It records which tail the DETECTOR
    # uses, which follows from the watermarked run's selection objective, and
    # exp20 always reads it from the watermarked side (see the two_sided
    # assignment further down) -- the null's copy is never consulted. The null
    # text itself cannot depend on the objective either: variant "none" draws a
    # single candidate so no selection runs at all, and the sampling seed is
    # (seed, variant, round_key) with the objective absent. So one null
    # generation is a valid matched null for every objective, and sharing it
    # across an LF and a PL run saves a full generation pass per objective.
    # Everything else in the field block -- aggregation, channel, both key
    # key domains and field metadata are still compared exactly, because
    # those do change what the null means.
    _wm_field = {k: v for k, v in watermarked.get("field", {}).items() if k != "test_tail"}
    _null_field = {k: v for k, v in null.get("field", {}).items() if k != "test_tail"}
    if _wm_field != _null_field:
        raise ValueError("watermarked and null inputs use different field metadata")
    if not isinstance(watermarked.get("encoder"), dict):
        raise ValueError("input is missing encoder metadata")
    if not isinstance(watermarked.get("feature_map"), dict):
        raise ValueError("input is missing feature_map metadata")
    if not isinstance(watermarked.get("field"), dict):
        raise ValueError("input is missing field metadata")
    if watermarked["field"].get("key_derivation_domain") != KEY_DOMAIN:
        raise ValueError("input key_derivation_domain does not match exp19")
    if int(watermarked.get("key_seed", -1)) != int(null.get("key_seed", -2)):
        raise ValueError("watermarked and null inputs use different key seeds")
    if _critical_config(watermarked) != _critical_config(null):
        raise ValueError("watermarked and null generation configurations differ")

    wm_documents = list(watermarked["documents"][:max_documents])
    null_documents = list(null["documents"][:max_documents])
    if _document_ids(wm_documents) != _document_ids(null_documents):
        raise ValueError("watermarked and null document ids are not matched and ordered")
    wm_prompts = watermarked.get("prompts")
    null_prompts = null.get("prompts")
    if not isinstance(wm_prompts, list) or not isinstance(null_prompts, list):
        raise ValueError("both inputs must store matched prompts")
    count = len(wm_documents)
    if list(wm_prompts[:count]) != list(null_prompts[:count]):
        raise ValueError("watermarked and null prompts do not match")

    for position, (wm_document, null_document) in enumerate(
        zip(wm_documents, null_documents)
    ):
        document_id = int(wm_document.get("document_id", position))
        _sentences(wm_document, f"watermarked document {document_id}")
        _sentences(null_document, f"null document {document_id}")
    return wm_documents, null_documents


def _external_text_paths(path: str | Path) -> dict[int, tuple[str, ...]]:
    """Load external actual-text attacks; cell-only artifacts are rejected."""

    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(value, dict) and "documents" in value:
        value = value["documents"]
    elif isinstance(value, dict) and "paraphrases" in value:
        value = value["paraphrases"]
    if isinstance(value, dict):
        value = [
            {"document_id": key, **(entry if isinstance(entry, dict)
             else {"sentences": entry})}
            for key, entry in value.items()
        ]
    if not isinstance(value, list):
        raise ValueError("external text JSON must contain a list or document map")

    from semfield.corpora import split_sentences

    output: dict[int, tuple[str, ...]] = {}
    for position, entry in enumerate(value):
        if isinstance(entry, dict):
            document_id = int(entry.get("document_id", position))
            if "sentences" in entry:
                sentences = entry["sentences"]
            elif "text" in entry:
                sentences = split_sentences(str(entry["text"]))
            elif "cells" in entry:
                raise ValueError(
                    f"external document {document_id} has cells but no text; "
                    "SemField robustness requires actual sentences"
                )
            else:
                raise ValueError(
                    f"external document {document_id} needs sentences or text"
                )
        elif isinstance(entry, str):
            document_id = position
            sentences = split_sentences(entry)
        elif isinstance(entry, list):
            document_id = position
            sentences = entry
        else:
            raise ValueError("invalid external text entry")
        if not isinstance(sentences, list) or any(
            not isinstance(sentence, str) for sentence in sentences
        ):
            raise ValueError(f"external document {document_id} has invalid sentences")
        # Generation can legitimately retain a document with zero completed
        # sentences after exhausting its retry budget. Keep that document in
        # the end-to-end denominator; the detector will count it as a
        # nondetection. This also preserves matched WM/null document IDs.
        cleaned = tuple(sentence.strip() for sentence in sentences if sentence.strip())
        if document_id in output:
            raise ValueError(f"duplicate external document id {document_id}")
        output[document_id] = cleaned
    return output


def _validate_external_coverage(
    values: dict[int, tuple[str, ...]],
    documents: Sequence[dict],
    *,
    label: str,
) -> None:
    expected = set(_document_ids(documents))
    missing = sorted(expected.difference(values))
    if missing:
        raise ValueError(
            f"{label} is missing selected document ids {missing}"
        )


def _split_one(sentence: str) -> tuple[str, str] | None:
    words = sentence.split()
    if len(words) < 4:
        return None
    midpoint = len(words) // 2
    # Prefer a nearby clause boundary, then make the new sentence boundary
    # explicit in the text rather than relying only on the Python list split.
    candidates = [
        index + 1 for index, word in enumerate(words[:-1])
        if word.rstrip('"\'’”)]}').endswith((",", ";", ":", "—", "-"))
        and 2 <= index + 1 <= len(words) - 2
    ]
    cut = (
        min(candidates, key=lambda index: abs(index - midpoint))
        if candidates else midpoint
    )
    left = " ".join(words[:cut]).rstrip()
    right = " ".join(words[cut:]).lstrip()
    closing = re.search(r"[\"'’”)\]}]*$", left)
    suffix = closing.group(0) if closing is not None else ""
    stem = left[:len(left) - len(suffix)] if suffix else left
    stem = re.sub(r"[,;:.!?—-]+$", "", stem.rstrip()).rstrip()
    if not stem or not right:
        return None
    return f"{stem}.{suffix}", right


def _merge_pair(left: str, right: str) -> str:
    """Remove a real boundary before joining two sentences as one clause."""

    stem = re.sub(r"[.?!]+[\"'’”]*$", "", left.rstrip()).rstrip()
    return f"{stem}, and {right.lstrip()}"


def attack_text(
    sentences: Sequence[str],
    attack: str,
    strength: float,
    rng: np.random.Generator,
    *,
    replacement_sentences: Sequence[str] | None = None,
    paraphrase_sentences: Sequence[str] | None = None,
) -> AttackedText:
    """Apply a deterministic-by-RNG text attack and retain source provenance."""

    clean = tuple(str(sentence) for sentence in sentences)
    n = len(clean)
    if attack not in ATTACKS:
        raise ValueError(f"unknown attack: {attack}")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("attack strength must lie in [0, 1]")
    if attack == "none" or n == 0 or strength == 0.0:
        return AttackedText(
            clean,
            tuple((index,) for index in range(n)),
            None,
            {"requested_strength": float(strength), "edited_sources": 0},
        )

    if attack == "reorder":
        if n < 2:
            return attack_text(clean, "none", 0.0, rng)
        moved = min(n, max(2, int(round(strength * n))))
        selected = np.sort(rng.choice(n, size=moved, replace=False))
        shift = int(rng.integers(1, moved))
        sources = np.arange(n)
        sources[selected] = np.roll(selected, shift)
        return AttackedText(
            tuple(clean[int(index)] for index in sources),
            tuple((int(index),) for index in sources),
            None,
            {"requested_strength": float(strength), "edited_sources": int(moved)},
        )

    if attack == "delete":
        removed = min(n, max(1, int(round(strength * n))))
        keep = np.ones(n, dtype=bool)
        keep[rng.choice(n, size=removed, replace=False)] = False
        indices = np.flatnonzero(keep)
        return AttackedText(
            tuple(clean[int(index)] for index in indices),
            tuple((int(index),) for index in indices),
            None,
            {"requested_strength": float(strength), "edited_sources": int(removed)},
        )

    if attack == "crop":
        removed = min(n, max(1, int(round(strength * n))))
        retained = n - removed
        if retained <= 0:
            return AttackedText(
                (), (), None,
                {"requested_strength": float(strength), "edited_sources": int(n)},
            )
        start = int(rng.integers(0, n - retained + 1))
        indices = range(start, start + retained)
        return AttackedText(
            tuple(clean[index] for index in indices),
            tuple((index,) for index in indices),
            None,
            {"requested_strength": float(strength), "edited_sources": int(removed),
             "crop_start": start},
        )

    if attack in {"dilute", "substitute"}:
        pool = tuple(str(value) for value in (replacement_sentences or ()))
        if not pool:
            raise ValueError(f"{attack} requires non-empty null replacement text")
        if attack == "substitute":
            replaced = min(n, max(1, int(round(strength * n))))
            chosen = np.sort(rng.choice(n, size=replaced, replace=False))
            output = list(clean)
            for index, value in zip(
                chosen, rng.choice(pool, size=replaced, replace=True)
            ):
                output[int(index)] = str(value)
            return AttackedText(
                tuple(output),
                tuple((index,) for index in range(n)),
                None,
                {"requested_strength": float(strength),
                 "edited_sources": int(replaced)},
            )

        if strength >= 1.0:
            inserted = max(1, n)
            additions = tuple(str(value) for value in rng.choice(
                pool, size=inserted, replace=True
            ))
            return AttackedText(
                additions,
                tuple(None for _ in additions),
                None,
                {"requested_strength": float(strength),
                 "edited_sources": int(n), "inserted_sentences": inserted},
            )
        inserted = max(1, int(round(strength * n / (1.0 - strength))))
        additions = [str(value) for value in rng.choice(
            pool, size=inserted, replace=True
        )]
        cut = int(rng.integers(0, inserted + 1))
        output = additions[:cut] + list(clean) + additions[cut:]
        groups = [None] * cut + [(index,) for index in range(n)] + [None] * (
            inserted - cut
        )
        return AttackedText(
            tuple(output), tuple(groups), (cut, cut + n),
            {"requested_strength": float(strength), "edited_sources": 0,
             "inserted_sentences": inserted, "clean_span_start": cut},
        )

    if attack == "insert":
        # AliMark's probing_insert (paraphraser/watermark_attack.py): drop
        # rate*n bare punctuation marks between sentences. It corrupts
        # segmentation rather than content -- the inserted fragments carry no
        # meaning, so a sentence-level detector sees spurious rows it has to
        # survive. Matching their semantics exactly, including sampling
        # positions from 1..n-1 and flooring the count, keeps our
        # perturbation curves comparable with their published ones.
        inserted = int(strength * n)
        if n > 1 and inserted > 0:
            inserted = min(inserted, n - 1)
            positions = set(int(value) for value in rng.choice(
                np.arange(1, n), size=inserted, replace=False
            ))
        else:
            positions = set()
        marks = (".", "?", "!")
        output: list[str] = []
        groups: list[tuple[int, ...] | None] = []
        for index, sentence in enumerate(clean):
            output.append(sentence)
            groups.append((index,))
            if index in positions:
                output.append(str(marks[int(rng.integers(0, len(marks)))]))
                groups.append(None)
        return AttackedText(
            tuple(output), tuple(groups), None,
            {"requested_strength": float(strength), "edited_sources": 0,
             "inserted_sentences": len(output) - n},
        )

    if attack == "split":
        target = min(n, max(1, int(round(strength * n))))
        eligible = [index for index, sentence in enumerate(clean)
                    if _split_one(sentence) is not None]
        chosen = set(int(value) for value in (
            rng.choice(eligible, size=min(target, len(eligible)), replace=False)
            if eligible else ()
        ))
        output: list[str] = []
        groups: list[tuple[int, ...]] = []
        for index, sentence in enumerate(clean):
            pieces = _split_one(sentence) if index in chosen else None
            if pieces is None:
                output.append(sentence)
                groups.append((index,))
            else:
                output.extend(pieces)
                groups.extend(((index,), (index,)))
        return AttackedText(
            tuple(output), tuple(groups), None,
            {"requested_strength": float(strength),
             "edited_sources": len(chosen), "new_boundaries": len(chosen)},
        )

    if attack == "merge":
        if n < 2:
            return attack_text(clean, "none", 0.0, rng)
        target_pairs = min(n // 2, max(1, int(round(strength * n / 2.0))))
        selected: set[int] = set()
        occupied: set[int] = set()
        for boundary in rng.permutation(n - 1):
            left = int(boundary)
            if left in occupied or left + 1 in occupied:
                continue
            selected.add(left)
            occupied.update((left, left + 1))
            if len(selected) == target_pairs:
                break
        output = []
        groups = []
        index = 0
        while index < n:
            if index in selected:
                output.append(_merge_pair(clean[index], clean[index + 1]))
                groups.append((index, index + 1))
                index += 2
            else:
                output.append(clean[index])
                groups.append((index,))
                index += 1
        return AttackedText(
            tuple(output), tuple(groups), None,
            {"requested_strength": float(strength),
             "edited_sources": int(2 * len(selected)),
             "removed_boundaries": len(selected)},
        )

    if paraphrase_sentences is None:
        raise ValueError("paraphrase requires external actual-text sentences")
    paraphrased = tuple(str(value).strip() for value in paraphrase_sentences)
    if strength >= 1.0:
        groups = tuple((index,) if index < n else None
                       for index in range(len(paraphrased)))
        return AttackedText(
            paraphrased, groups, None,
            {"requested_strength": float(strength), "edited_sources": int(n),
             "alignment": "index-only"},
        )
    if len(paraphrased) != n:
        raise ValueError(
            "partial paraphrase requires exactly one external sentence per clean sentence"
        )
    replaced = min(n, max(1, int(round(strength * n))))
    chosen = set(int(value) for value in rng.choice(
        n, size=replaced, replace=False
    ))
    output = tuple(paraphrased[index] if index in chosen else clean[index]
                   for index in range(n))
    return AttackedText(
        output, tuple((index,) for index in range(n)), None,
        {"requested_strength": float(strength), "edited_sources": replaced,
         "alignment": "one-to-one"},
    )


class FeatureEncoder:
    """Reconstruct exp19's exact encoder/RFF map and memoize sentence rows."""

    def __init__(
        self,
        encoder_metadata: dict,
        feature_metadata: dict,
        *,
        device: str | None,
        cache_dir: str | Path | None,
        max_embed_length: int = 512,
        embed_batch_size: int | None = None,
    ):
        self.encoder_metadata = dict(encoder_metadata)
        self.feature_map = feature_map_from_metadata(feature_metadata)
        self.device = device
        self.cache_dir = cache_dir
        self.max_embed_length = int(max_embed_length)
        self.embed_batch_size = (
            None if embed_batch_size is None else int(embed_batch_size)
        )
        self._rows: dict[str, np.ndarray] = {}
        # Stateful feature maps (currently residual SemField) cannot cache a
        # transformed row by sentence text alone: its innovation depends on
        # the preceding sentences in the document.  Keep normalized raw
        # embeddings and transform complete sequences at evaluation time.
        self._raw_rows: dict[str, np.ndarray] = {}
        self._occurrence_rows: dict[tuple[str, int, int], tuple[str, np.ndarray]] = {}

    def seed_rows(
        self,
        rows: dict,
        *,
        source: str,
        split_name: str,
    ) -> dict[str, int]:
        """Seed exact sidecar occurrences and text shortcuts for edits."""

        added = 0
        reused = 0
        conflicts = 0
        for (document_id, sentence_index), pair in rows.items():
            sentence, value = pair
            sentence = str(sentence)
            row = np.asarray(value, dtype=np.float64).ravel()
            if (
                row.size != self.feature_map.n_features
                or np.any(~np.isfinite(row))
            ):
                raise ValueError(f"{source} has an invalid feature row")
            occurrence = (
                str(split_name), int(document_id), int(sentence_index)
            )
            self._occurrence_rows[occurrence] = (sentence, row.copy())
            existing = self._rows.get(sentence)
            if existing is None:
                self._rows[sentence] = row.copy()
                added += 1
            elif np.array_equal(existing, row):
                reused += 1
            else:
                conflicts += 1
        return {
            "occurrence_rows_added": len(rows),
            "unique_rows_added": added,
            "matching_rows_reused": reused,
            "conflicting_text_shortcuts_ignored": conflicts,
        }

    def prepare(self, sentences: Iterable[str]) -> None:
        ordered = list(dict.fromkeys(str(value) for value in sentences))
        cache = self._raw_rows if getattr(self.feature_map, "stateful", False) else self._rows
        missing = [value for value in ordered if value not in cache]
        if not missing:
            return
        from semfield.embedders import embed, load_embedder, resolve_devices

        name = str(self.encoder_metadata.get("name"))
        names = self.encoder_metadata.get("model_ids")
        if not names:
            names = [name]
        names = [str(value) for value in names if str(value)]
        if not names or names == ["None"]:
            raise ValueError("encoder metadata has no model name")
        precision = str(
            self.encoder_metadata.get("precision_requested", "auto")
        )
        output_dim_value = self.encoder_metadata.get("output_dim_requested")
        output_dim = None if output_dim_value is None else int(output_dim_value)
        resolved_device = self.device
        devices = resolve_devices(resolved_device)
        blocks = []
        for model_name in names:
            model = load_embedder(
                model_name,
                output_dim=output_dim,
                device=devices[0],
                max_seq_length=self.max_embed_length,
                precision=precision,
            )
            block = embed(
                model,
                missing,
                batch_size=self.embed_batch_size,
                normalize=True,
                devices=devices,
                model_name=model_name,
                cache_dir=self.cache_dir,
                output_dim=output_dim,
                precision=precision,
                show_progress_bar=False,
            )
            block = np.asarray(block, dtype=np.float64)
            block /= np.maximum(np.linalg.norm(block, axis=1, keepdims=True),
                                1e-12)
            blocks.append(block / math.sqrt(len(names)))
        vectors = np.concatenate(blocks, axis=1)
        if vectors.shape[1] != self.feature_map.input_dim:
            raise ValueError(
                f"encoder returned width {vectors.shape[1]}, but exp19 feature "
                f"map requires {self.feature_map.input_dim}"
            )
        if getattr(self.feature_map, "stateful", False):
            for sentence, row in zip(missing, vectors):
                self._raw_rows[sentence] = np.asarray(row, dtype=np.float64)
        else:
            features = self.feature_map.transform(vectors)
            for sentence, row in zip(missing, features):
                self._rows[sentence] = np.asarray(row, dtype=np.float64)

    def matrix(self, sentences: Sequence[str]) -> np.ndarray:
        if not sentences:
            return np.zeros((0, self.feature_map.n_features), dtype=np.float64)
        if getattr(self.feature_map, "stateful", False):
            self.prepare(sentences)
            raw = np.asarray(
                [self._raw_rows[str(sentence)] for sentence in sentences],
                dtype=np.float64,
            )
            return self.feature_map.transform_sequence(raw)
        absent = [sentence for sentence in sentences if sentence not in self._rows]
        if absent:
            self.prepare(absent)
        return np.asarray([self._rows[str(sentence)] for sentence in sentences],
                          dtype=np.float64)

    def occurrence_matrix(
        self, split_name: str, document_id: int, sentences: Sequence[str]
    ) -> np.ndarray:
        rows: list[np.ndarray] = []
        for sentence_index, sentence in enumerate(sentences):
            key = (str(split_name), int(document_id), int(sentence_index))
            stored = self._occurrence_rows.get(key)
            if stored is None:
                raise ValueError(f"missing exact sidecar occurrence {key}")
            stored_sentence, row = stored
            if stored_sentence != str(sentence):
                raise ValueError(f"sidecar occurrence text mismatch at {key}")
            rows.append(row)
        if not rows:
            return np.zeros((0, self.feature_map.n_features), dtype=np.float64)
        return np.asarray(rows, dtype=np.float64)

    def attacked_matrix(
        self,
        clean_sentences: Sequence[str],
        clean_features: np.ndarray,
        attacked: AttackedText,
    ) -> np.ndarray:
        if getattr(self.feature_map, "stateful", False):
            # Reordering changes the running residual even when every sentence
            # is textually unchanged.  Recompute the complete attacked
            # sequence whenever its order/content differs from clean text;
            # the exact clean sidecar remains valid for the no-attack case.
            if tuple(attacked.sentences) == tuple(clean_sentences):
                return np.asarray(clean_features, dtype=np.float64)
            if not attacked.sentences:
                return np.zeros(
                    (0, self.feature_map.n_features), dtype=np.float64
                )
            self.prepare(attacked.sentences)
            raw = np.asarray(
                [self._raw_rows[str(sentence)] for sentence in attacked.sentences],
                dtype=np.float64,
            )
            return self.feature_map.transform_sequence(raw)
        changed: list[str] = []
        for sentence, group in zip(attacked.sentences, attacked.source_groups):
            exact = (
                group is not None
                and len(group) == 1
                and 0 <= group[0] < len(clean_sentences)
                and sentence == clean_sentences[group[0]]
            )
            if not exact:
                changed.append(sentence)
        self.prepare(changed)

        rows: list[np.ndarray] = []
        for sentence, group in zip(attacked.sentences, attacked.source_groups):
            exact = (
                group is not None
                and len(group) == 1
                and 0 <= group[0] < len(clean_sentences)
                and sentence == clean_sentences[group[0]]
            )
            rows.append(
                clean_features[group[0]] if exact else self._rows[str(sentence)]
            )
        if not rows:
            return np.zeros((0, self.feature_map.n_features), dtype=np.float64)
        return np.asarray(rows, dtype=np.float64)


def _weights(
    live_key: bytes,
    controls: Sequence[bytes],
    dimension: int,
    channel: int,
    *,
    moment: bool = False,
) -> np.ndarray:
    keys = [live_key, *controls]
    if moment:
        return np.asarray([
            [
                keyed_gaussian(key, dimension, channel=channel),
                keyed_gaussian(key, dimension, channel=channel + 1),
            ]
            for key in keys
        ], dtype=np.float64)
    return np.asarray([
        keyed_gaussian(key, dimension, channel=channel) for key in keys
    ], dtype=np.float64)


def _anchored_weights(
    live_key: bytes,
    controls: Sequence[bytes],
    dimension: int,
    channel: int,
    rho: float,
    fusion: str | None = None,
    blocks: int | None = None,
) -> dict[str, object]:
    """Describe live/control keys for an anchored-field document.

    Local position fields are derived lazily by ``anchored_field_scores``;
    keeping only the keys here avoids allocating a controls x positions x
    embedding-dimension tensor for every document.
    """

    return {
        "keys": [live_key, *controls],
        "dimension": int(dimension),
        "channel": int(channel),
        "rho": float(rho),
        "blocks": blocks,
        "fusion": fusion,
    }






def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return None
    return float(numerator / denominator) if abs(denominator) > 1e-15 else None


def retention_metrics(
    clean_sentences: Sequence[str],
    clean_features: np.ndarray,
    attacked: AttackedText,
    attacked_features: np.ndarray,
    live_weights: np.ndarray,
) -> dict[str, object]:
    """Measure text, feature, and keyed-contribution survival without clocks."""

    clean = tuple(clean_sentences)
    n_clean = len(clean)
    represented: set[int] = set()
    exact: set[int] = set()
    aligned_kernels: list[float] = []
    for attacked_index, group in enumerate(attacked.source_groups):
        if group is None:
            continue
        for source in group:
            if not 0 <= source < n_clean:
                continue
            represented.add(source)
            aligned_kernels.append(float(np.dot(
                attacked_features[attacked_index], clean_features[source]
            )))
        if (len(group) == 1 and 0 <= group[0] < n_clean and
                attacked.sentences[attacked_index] == clean[group[0]]):
            exact.add(group[0])

    clean_aggregate = clean_features.sum(axis=0)
    attacked_aggregate = attacked_features.sum(axis=0)
    clean_norm = float(np.linalg.norm(clean_aggregate))
    attacked_norm = float(np.linalg.norm(attacked_aggregate))
    cross = float(np.dot(clean_aggregate, attacked_aggregate))
    cosine = (_safe_ratio(cross, clean_norm * attacked_norm)
              if clean_norm > 1e-15 and attacked_norm > 1e-15 else None)
    projection = (_safe_ratio(cross, clean_norm * clean_norm)
                  if clean_norm > 1e-15 else None)
    clean_score = float(np.dot(live_weights, clean_aggregate))
    attacked_score = float(np.dot(live_weights, attacked_aggregate))
    exact_score = float(sum(
        np.dot(live_weights, clean_features[index]) for index in exact
    ))
    return {
        "clean_sentences": n_clean,
        "attacked_sentences": len(attacked.sentences),
        "represented_clean_sentences": len(represented),
        "exactly_retained_clean_sentences": len(exact),
        "source_coverage": float(len(represented) / n_clean) if n_clean else None,
        "exact_text_retention": float(len(exact) / n_clean) if n_clean else None,
        "mean_aligned_feature_kernel": (
            float(np.mean(aligned_kernels)) if aligned_kernels else None
        ),
        "aggregate_feature_cosine": cosine,
        "aggregate_feature_projection": projection,
        "aggregate_norm_ratio": _safe_ratio(attacked_norm, clean_norm),
        "aggregate_l2_change": float(np.linalg.norm(
            attacked_aggregate - clean_aggregate
        )),
        "live_score_ratio": _safe_ratio(attacked_score, clean_score),
        "exact_contribution_fraction": _safe_ratio(exact_score, clean_score),
        "sentence_multiset_unchanged": Counter(clean) == Counter(attacked.sentences),
    }


def _localization(scan: dict[str, object] | None,
                  true_span: tuple[int, int] | None) -> dict[str, object] | None:
    if scan is None or true_span is None:
        return None
    start = scan.get("window_start")
    stop = scan.get("window_end")
    if start is None or stop is None:
        return None
    a, b = int(start), int(stop)
    u, v = true_span
    overlap = max(0, min(b, v) - max(a, u))
    union = max(b, v) - min(a, u)
    return {
        "true_start": int(u),
        "true_end": int(v),
        "intersection": int(overlap),
        "iou": float(overlap / union) if union else None,
        "watermarked_recall": float(overlap / (v - u)) if v > u else None,
        "selected_precision": float(overlap / (b - a)) if b > a else None,
    }


def evaluate_one(
    document: dict,
    clean_sentences: tuple[str, ...],
    attacked: AttackedText,
    encoder: FeatureEncoder,
    weights: np.ndarray | dict[str, object],
    *,
    split_name: str,
    condition: Condition,
    alpha: float,
    scan_min: int,
    scan_max: int | None,
    scan_sizes: Sequence[int] | None,
    scan_stride_fraction: float,
    run_controls: bool,
    run_scan: bool,
    two_sided: bool,
    moment_lambda: float | None = None,
    anchored: dict[str, object] | None = None,
    aer: dict[str, object] | None = None,
    soft_sign: dict[str, object] | None = None,
    decorrelated: bool = False,
    innovation: bool = False,
) -> dict[str, object]:
    document_id = int(document.get("document_id", 0))
    clean_features = encoder.occurrence_matrix(
        split_name, document_id, clean_sentences
    )
    attacked_features = encoder.attacked_matrix(
        clean_sentences, clean_features, attacked
    )
    if aer is not None:
        field = AERSemanticField(
            type("FeatureMap", (), {"n_features": attacked_features.shape[1]})(),
            aer["keys"][0], blocks=int(aer["blocks"]),
            channel=int(aer["channel"]),
            lambda_grid=tuple(aer.get("lambda_grid", (0.0,.5,1.0,2.0,3.0))),
        )
        statistic = field.detect_features(attacked_features)
        live_weights = field.weights(0)
    elif innovation:
        statistic = innovation_span_statistic(
            attacked_features, weights[0], two_sided=two_sided
        )
        live_weights = weights[0]
    elif decorrelated:
        statistic = decorrelated_field_statistic(
            attacked_features, weights[0], two_sided=two_sided
        )
        live_weights = weights[0]
    elif soft_sign is not None:
        field = SoftSignSemanticField(
            attacked_features.shape[1], soft_sign["keys"][0],
            channels=int(soft_sign["channels"]),
            beta=float(soft_sign["beta"]), channel=int(soft_sign["channel"]),
        )
        statistic = field.detect_features(
            attacked_features, two_sided=two_sided
        )
        # Retention diagnostics use one fixed keyed direction; this is only a
        # content-survival diagnostic and is not the soft-sign detector.
        live_weights = field.directions[0]
    elif anchored is not None:
        if anchored.get("fusion") == "max-absolute-global-anchored-v1":
            statistic = protected_anchored_field_statistic(
                attacked_features,
                anchored["keys"][0],
                rho=float(anchored["rho"]),
                channel=int(anchored["channel"]),
                positions=block_positions(
                    len(attacked_features), anchored.get("blocks")
                ),
            )
        else:
            statistic = anchored_field_statistic(
                attacked_features,
                anchored["keys"][0],
                rho=float(anchored["rho"]),
                channel=int(anchored["channel"]),
                two_sided=two_sided,
                positions=block_positions(
                    len(attacked_features), anchored.get("blocks")
                ),
            )
        # Retention diagnostics intentionally use the global component only;
        # they describe semantic content survival rather than the full keyed
        # detector score.
        live_weights = keyed_gaussian(
            anchored["keys"][0],
            int(anchored["dimension"]),
            channel=int(anchored["channel"]),
        )
    elif moment_lambda is not None:
        statistic = moment_field_statistic(
            attacked_features, weights[0][0], weights[0][1],
            moment_lambda=float(moment_lambda), two_sided=two_sided,
        )
        live_weights = weights[0][0]
    else:
        statistic_fn = two_sided_field_statistic if two_sided else field_statistic
        statistic = statistic_fn(attacked_features, weights[0])
        live_weights = weights[0]
    retention = retention_metrics(
        clean_sentences, clean_features, attacked, attacked_features, live_weights
    )
    document_id = int(document.get("document_id", 0))
    return {
        "document_id": document_id,
        "split": split_name,
        "attack": condition.attack,
        "strength": float(condition.strength),
        "clean_sentence_count": len(clean_sentences),
        "attacked_sentence_count": len(attacked.sentences),
        "analytic": {
            **asdict(statistic),
            "detected": bool(statistic.analytic_pvalue <= alpha),
        },
        "alignment": {
            "source_groups": [
                None if group is None else [int(value) for value in group]
                for group in attacked.source_groups
            ],
            "watermarked_span": (
                None
                if attacked.watermarked_span is None
                else [int(value) for value in attacked.watermarked_span]
            ),
        },
        "localization": _localization(scan, attacked.watermarked_span),
        "retention": retention,
        "attack_details": attacked.details,
    }


def _wilson_interval(successes: int, total: int,
                     confidence: float = 0.95) -> list[float] | None:
    if total <= 0:
        return None
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    rate = successes / total
    scale = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / scale
    radius = z / scale * math.sqrt(
        rate * (1.0 - rate) / total + z * z / (4.0 * total * total)
    )
    return [float(max(0.0, center - radius)),
            float(min(1.0, center + radius))]


def _mean_optional(values: Iterable[object]) -> float | None:
    numeric = [float(value) for value in values
               if value is not None and math.isfinite(float(value))]
    return float(np.mean(numeric)) if numeric else None


def summarize(rows: Sequence[dict], alpha: float) -> dict[str, object]:
    total = len(rows)

    def detection_summary(section: str) -> dict[str, object]:
        eligible = [row for row in rows if row.get(section) is not None]
        hits = sum(bool(row[section]["detected"]) for row in eligible)
        return {
            "documents": len(eligible),
            "detections": int(hits),
            "rate": float(hits / len(eligible)) if eligible else None,
            "rate_wilson95": _wilson_interval(hits, len(eligible)),
            "mean_pvalue": _mean_optional(
                row[section].get("analytic_pvalue", row[section].get("pvalue"))
                for row in eligible
            ),
        }

    retention_fields = (
        "source_coverage",
        "exact_text_retention",
        "mean_aligned_feature_kernel",
        "aggregate_feature_cosine",
        "aggregate_feature_projection",
        "aggregate_norm_ratio",
        "aggregate_l2_change",
        "live_score_ratio",
        "exact_contribution_fraction",
    )
    localizations = [row["localization"] for row in rows
                     if row.get("localization") is not None]
    return {
        "documents": total,
        "alpha": float(alpha),
        "analytic": detection_summary("analytic"),
        "mean_z_analytic": _mean_optional(
            row["analytic"]["z_score"] for row in rows
        ),
        "mean_design_effect": _mean_optional(
            row["analytic"]["design_effect"] for row in rows
        ),
        "mean_retention": {
            field: _mean_optional(row["retention"].get(field) for row in rows)
            for field in retention_fields
        },
        "localization": {
            "documents": len(localizations),
            "mean_iou": _mean_optional(row.get("iou") for row in localizations),
            "mean_watermarked_recall": _mean_optional(
                row.get("watermarked_recall") for row in localizations
            ),
            "mean_selected_precision": _mean_optional(
                row.get("selected_precision") for row in localizations
            ),
        },
        "interval_note": (
            "Wilson intervals are descriptive document-level intervals; "
            "shared model prompts and one master key can induce dependence."
        ),
    }


def _condition_grid(attacks: Sequence[str], strengths: Sequence[float]) -> list[Condition]:
    output = []
    for attack in attacks:
        values = [0.0] if attack == "none" else strengths
        output.extend(Condition(attack, float(value)) for value in values)
    return output


def _prepare_attacks(
    documents: Sequence[dict],
    conditions: Sequence[Condition],
    *,
    split_name: str,
    seed: int,
    replacements: Sequence[str],
    paraphrases: dict[int, tuple[str, ...]] | None,
) -> dict[tuple[int, Condition], AttackedText]:
    output: dict[tuple[int, Condition], AttackedText] = {}
    for position, document in enumerate(documents):
        document_id = int(document.get("document_id", position))
        clean = _sentences(document, f"{split_name} document {document_id}")
        for condition in conditions:
            external = ((paraphrases or {}).get(document_id)
                        if condition.attack == "paraphrase" else None)
            if condition.attack == "paraphrase" and external is None:
                continue
            rng = np.random.default_rng(_condition_seed((
                seed, "semfield-robustness", condition.attack,
                condition.strength, document_id,
            )))
            output[(document_id, condition)] = attack_text(
                clean,
                condition.attack,
                condition.strength,
                rng,
                replacement_sentences=replacements,
                paraphrase_sentences=external,
            )
    return output


def _sentence_fingerprint(sentences: Iterable[str]) -> str:
    digest = hashlib.blake2b(digest_size=16)
    count = 0
    for sentence in sentences:
        digest.update(str(sentence).encode("utf-8", "ignore"))
        digest.update(b"\x00")
        count += 1
    digest.update(count.to_bytes(8, "big"))
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--watermarked", required=True,
                        help="exp19 SemField generation JSON")
    parser.add_argument("--null", required=True,
                        help="matched exp19 none generation JSON")
    parser.add_argument("--attacks", nargs="+", choices=ATTACKS,
                        default=["none", "reorder", "delete", "crop",
                                 "dilute", "substitute", "split", "merge"])
    parser.add_argument("--strengths", nargs="+", type=float,
                        default=[0.10, 0.25, 0.50, 0.75])
    parser.add_argument("--watermarked-paraphrases", "--paraphrases",
                        dest="watermarked_paraphrases", default=None,
                        help="external watermarked JSON with sentences/text")
    parser.add_argument("--null-paraphrases", default=None,
                        help="separate external null JSON with sentences/text")
    parser.add_argument(
        "--positive-only-attacks", action="store_true",
        help="attack watermarked positives only; retain clean nulls (AliMark probing protocol)",
    )
    parser.add_argument("--alpha", type=float, default=0.01,
                        help="analytic Gaussian-test significance level (default: 0.01)")
    parser.add_argument("--scan-min", type=int, default=8)
    parser.add_argument("--scan-max", type=int, default=None)
    parser.add_argument("--scan-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--scan-stride-fraction", type=float, default=0.5)
    parser.add_argument("--scan-max-documents", type=int, default=None,
                        help="run scans only for the first N matched document pairs")
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--device", default=None,
                        help="embedding device; defaults to exp19 config.device")
    parser.add_argument("--cache-dir", default=EMBEDDING_CACHE)
    parser.add_argument("--seed", type=int, default=0,
                        help="key-independent structural-attack seed")
    parser.add_argument("--out", default="results_exp20_semfield_robustness.json")
    args = parser.parse_args(argv)

    if not 0.0 < args.alpha < 1.0:
        parser.error("--alpha must lie in (0, 1)")
    if any(not 0.0 <= value <= 1.0 for value in args.strengths):
        parser.error("all --strengths must lie in [0, 1]")
    if args.scan_min < 1:
        parser.error("--scan-min must be positive")
    if args.scan_max is not None and args.scan_max < 1:
        parser.error("--scan-max must be positive")
    if not 0.0 < args.scan_stride_fraction <= 1.0:
        parser.error("--scan-stride-fraction must lie in (0, 1]")
    if args.max_documents is not None and args.max_documents < 1:
        parser.error("--max-documents must be positive")
    if args.scan_max_documents is not None and args.scan_max_documents < 0:
        parser.error("--scan-max-documents must be non-negative")
    if "paraphrase" in args.attacks and args.watermarked_paraphrases is None:
        parser.error("paraphrase requires --watermarked-paraphrases")
    if args.null_paraphrases is not None and "paraphrase" not in args.attacks:
        parser.error("--null-paraphrases is only used with paraphrase")

    try:
        wm_path, watermarked = _load_payload(args.watermarked, "watermarked")
        null_path, null = _load_payload(args.null, "null")
        wm_documents, null_documents = validate_matched_inputs(
            watermarked, null, args.max_documents
        )
        generation_config = watermarked["config"]
        recorded_aggregation = str(watermarked.get("field", {}).get("aggregation", "uniform"))
        if recorded_aggregation not in {"uniform", "sentence-decorrelated", "span-energy"}:
            raise ValueError("unknown SemField sentence aggregation")
        two_sided = (
            watermarked.get("field", {}).get("test_tail")
            == "two-sided-absolute"
        )
        feature_map = feature_map_from_metadata(watermarked["feature_map"])
        key_seed = int(watermarked["key_seed"])
        field_metadata = watermarked.get("field", {})
        channel = int(field_metadata.get("channel", 0))
        field_structure = str(field_metadata.get("structure", "global"))
        field_rho = float(field_metadata.get("rho", 1.0))
        field_blocks = field_metadata.get("blocks")
        if field_blocks is not None:
            field_blocks = int(field_blocks)
        field_fusion = field_metadata.get("fusion")
        field_type = str(field_metadata.get("type", ""))
        is_decorrelated = (
            field_type == "sentence-decorrelated-keyed-gaussian-v1"
        )
        is_innovation = field_type == "innovation-span-keyed-gaussian-v1"
        if is_innovation and field_structure != "global":
            raise ValueError("innovation detector requires an unfused global field")
        if is_innovation:
            if recorded_aggregation != "span-energy":
                raise ValueError("innovation detector metadata is inconsistent")
        elif is_decorrelated:
            if field_structure != "global" or field_fusion is not None:
                raise ValueError(
                    "decorrelated detector requires an unfused global field"
                )
            if recorded_aggregation != "sentence-decorrelated":
                raise ValueError("decorrelated detector metadata is inconsistent")
        is_soft_sign = field_type == "soft-sign-semantic-field-v1"
        soft_sign_channels = field_metadata.get("soft_sign_channels")
        soft_sign_beta = field_metadata.get("soft_sign_beta")
        if is_soft_sign:
            if field_structure != "global" or field_fusion is not None:
                raise ValueError(
                    "soft-sign detector requires an unfused global field"
                )
            try:
                soft_sign_channels = int(soft_sign_channels)
                soft_sign_beta = float(soft_sign_beta)
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid soft-sign detector metadata") from exc
            if (soft_sign_channels <= 0 or not math.isfinite(soft_sign_beta)
                    or soft_sign_beta <= 0.0):
                raise ValueError("invalid soft-sign detector metadata")
        is_aer = field_type == "aer-relative-block-gaussian-v1"
        if field_structure not in {"global", "anchored", "aer"}:
            raise ValueError(f"unknown field structure {field_structure!r}")
        if is_aer and field_structure != "aer":
            raise ValueError("AER metadata requires field_structure='aer'")
        if not math.isfinite(field_rho) or not 0.0 <= field_rho <= 1.0:
            raise ValueError("field rho must lie in [0, 1]")
        if field_fusion not in {None, "max-absolute-global-anchored-v1"}:
            raise ValueError(f"unknown detector fusion {field_fusion!r}")
        if field_fusion is not None and field_structure != "anchored":
            raise ValueError("detector fusion requires an anchored field")
        live_key = experiment_key(key_seed)
        controls = ()
        moment_metadata = field_metadata.get("moment", {})
        is_moment = (
            isinstance(moment_metadata, dict)
            and bool(moment_metadata.get("enabled", False))
        )
        moment_lambda = (
            float(moment_metadata.get("lambda", 0.5)) if is_moment else None
        )
        effective_scan_max = (
            int(args.scan_max)
            if args.scan_max is not None
            else int(generation_config["sentences"])
        )
        max_embed_length = int(generation_config["max_embed_length"])
        embed_batch_value = generation_config.get("embed_batch_size")
        embed_batch_size = (
            None if embed_batch_value is None else int(embed_batch_value)
        )
        if effective_scan_max < 1 or max_embed_length < 1:
            raise ValueError("exp19 sentence/embed limits must be positive")
        if embed_batch_size is not None and embed_batch_size < 1:
            raise ValueError("exp19 embed_batch_size must be positive")
        wm_sidecar_validation = validate_feature_sidecar(
            wm_path, watermarked, label="watermarked"
        )
        null_sidecar_validation = validate_feature_sidecar(
            null_path, null, label="null"
        )
        wm_sidecar_rows = _validated_sidecar_rows(
            wm_path, watermarked, label="watermarked"
        )
        null_sidecar_rows = _validated_sidecar_rows(
            null_path, null, label="null"
        )
        wm_paraphrases = (
            _external_text_paths(args.watermarked_paraphrases)
            if args.watermarked_paraphrases else None
        )
        null_paraphrases = (
            _external_text_paths(args.null_paraphrases)
            if args.null_paraphrases else None
        )
        if "paraphrase" in args.attacks:
            available = set((wm_paraphrases or {}).keys())
            missing = sorted(set(_document_ids(wm_documents)) - available)
            if missing:
                print(f"warning: excluding {len(missing)} watermarked documents missing non-empty paraphrases: {missing}", file=sys.stderr, flush=True)
                wm_documents = [d for d in wm_documents if int(d["document_id"]) in available]
                retained = {int(d["document_id"]) for d in wm_documents}
                null_documents = [d for d in null_documents if int(d["document_id"]) in retained]
                null_by_id = {int(d["document_id"]): d for d in null_documents}
                null_documents = [null_by_id[int(d["document_id"])] for d in wm_documents]
            if null_paraphrases is not None:
                _validate_external_coverage(
                    null_paraphrases,
                    null_documents,
                    label="null paraphrases",
                )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    for split_name, validation in (
        ("watermarked", wm_sidecar_validation),
        ("null", null_sidecar_validation),
    ):
        print(
            f"validated {split_name} exp19 sidecar: "
            f"sha256={validation['sha256']}, "
            f"detectors={validation['detector_validation']['documents_checked']}"
        )

    replacements = tuple(
        sentence for position, document in enumerate(null_documents)
        for sentence in _sentences(
            document,
            f"null document {int(document.get('document_id', position))}",
        )
    )
    if any(attack in {"dilute", "substitute"} for attack in args.attacks) and not replacements:
        parser.error("dilute/substitute requires non-empty matched null text")

    conditions = _condition_grid(args.attacks, args.strengths)
    try:
        wm_attacks = _prepare_attacks(
            wm_documents,
            conditions,
            split_name="watermarked",
            seed=args.seed,
            replacements=replacements,
            paraphrases=wm_paraphrases,
        )
        null_attacks = _prepare_attacks(
            null_documents,
            conditions,
            split_name="null",
            seed=args.seed,
            replacements=replacements,
            paraphrases=null_paraphrases,
        )
        if args.positive_only_attacks:
            for position, document in enumerate(null_documents):
                document_id = int(document.get("document_id", position))
                clean = _sentences(document, f"null document {document_id}")
                for condition in conditions:
                    null_attacks[(document_id, condition)] = attack_text(
                        clean, "none", 0.0, np.random.default_rng(0)
                    )
    except ValueError as exc:
        parser.error(str(exc))

    all_sentences: list[str] = []
    for collection in (wm_attacks, null_attacks):
        for attacked in collection.values():
            all_sentences.extend(attacked.sentences)
    device = args.device
    if device is None:
        configured = watermarked.get("config", {}).get("device")
        device = None if configured in (None, "None") else str(configured)
    encoder = FeatureEncoder(
        watermarked["encoder"], watermarked["feature_map"],
        device=device, cache_dir=(args.cache_dir or None),
        max_embed_length=max_embed_length,
        embed_batch_size=embed_batch_size,
    )
    wm_seed_result = encoder.seed_rows(
        wm_sidecar_rows, source="watermarked exp19 sidecar",
        split_name="watermarked"
    )
    null_seed_result = encoder.seed_rows(
        null_sidecar_rows, source="null exp19 sidecar",
        split_name="null"
    )
    sidecar_encoder_seed = {
        "watermarked": {
            "unique_sentence_rows": len(wm_sidecar_rows),
            **wm_seed_result,
        },
        "null": {
            "unique_sentence_rows": len(null_sidecar_rows),
            **null_seed_result,
        },
        "combined_unique_sentence_rows": (
            wm_seed_result["unique_rows_added"]
            + null_seed_result["unique_rows_added"]
        ),
        "conflicting_exact_text_rows": (
            wm_seed_result["conflicting_text_shortcuts_ignored"]
            + null_seed_result["conflicting_text_shortcuts_ignored"]
        ),
    }
    encoder.prepare(all_sentences)

    rows_by_condition = {
        condition: {"watermarked": [], "null": []} for condition in conditions
    }
    for position, (wm_document, null_document) in enumerate(
        zip(wm_documents, null_documents)
    ):
        wm_id = int(wm_document.get("document_id", position))
        null_id = int(null_document.get("document_id", position))
        if wm_id != null_id:
            raise RuntimeError("matched document ids changed after validation")
        run_controls = False
        key_weights = ({
            "keys": [live_key, *(controls if run_controls else ())],
            "dimension": feature_map.n_features,
            "channel": channel,
            "channels": soft_sign_channels,
            "beta": soft_sign_beta,
        } if is_soft_sign else _weights(
            live_key,
            controls if run_controls else (),
            feature_map.n_features,
            channel,
            moment=is_moment,
        )) if field_structure == "global" else _anchored_weights(
            live_key,
            controls if run_controls else (),
            feature_map.n_features,
            channel,
            field_rho,
            field_fusion,
            field_blocks,
        )
        aer_weights = ({
            "keys": [live_key, *(controls if run_controls else ())],
            "dimension": feature_map.n_features,
            "channel": channel,
            "blocks": int(field_blocks or 4),
            "lambda_grid": [0.0, 0.5, 1.0, 2.0, 3.0],
        } if is_aer else None)
        if aer_weights is not None:
            key_weights = aer_weights
        run_scan = run_controls and (
            args.scan_max_documents is None
            or position < args.scan_max_documents
        )
        for condition in conditions:
            for split_name, document, attacks in (
                ("watermarked", wm_document, wm_attacks),
                ("null", null_document, null_attacks),
            ):
                attacked = attacks.get((wm_id, condition))
                if attacked is None:
                    continue
                clean = _sentences(document, f"{split_name} document {wm_id}")
                row = evaluate_one(
                    document,
                    clean,
                    attacked,
                    encoder,
                    key_weights,
                    split_name=split_name,
                    condition=condition,
                    alpha=args.alpha,
                    scan_min=args.scan_min,
                    scan_max=effective_scan_max,
                    scan_sizes=args.scan_sizes,
                    scan_stride_fraction=args.scan_stride_fraction,
                    run_controls=run_controls,
                    run_scan=run_scan,
                    two_sided=two_sided,
                    moment_lambda=moment_lambda,
                    anchored=(key_weights if field_structure == "anchored" else None),
                    aer=(key_weights if is_aer else None),
                    soft_sign=(key_weights if is_soft_sign else None),
                    decorrelated=is_decorrelated,
                    innovation=is_innovation,
                )
                rows_by_condition[condition][split_name].append(row)

    output_conditions = []
    for condition in conditions:
        wm_rows = rows_by_condition[condition]["watermarked"]
        null_rows = rows_by_condition[condition]["null"]
        wm_summary = summarize(wm_rows, args.alpha)
        null_summary = summarize(null_rows, args.alpha)
        fpr_status = "estimated"
        if condition.attack == "paraphrase" and null_paraphrases is None:
            fpr_status = (
                "not_estimated: separate --null-paraphrases required; "
                "watermarked paraphrases are never reused as null"
            )
        print(
            f"{condition.attack:<11} strength={condition.strength:g} "
            f"TPR analytic={wm_summary['analytic']['rate']} "
            f"FPR={null_summary['analytic']['rate']}"
        )
        output_conditions.append({
            "attack": condition.attack,
            "strength": float(condition.strength),
            "watermarked": wm_summary,
            "null": null_summary,
            "rates": {
                "tpr_analytic": wm_summary["analytic"]["rate"],
                "fpr_analytic": null_summary["analytic"]["rate"],
                "fpr_status": fpr_status,
            },
            "documents": {"watermarked": wm_rows, "null": null_rows},
        })

    output = {
        "schema_version": 1,
        "experiment": "exp20_semfield_robustness",
        "config": vars(args),
        "input": {
            "watermarked": str(wm_path),
            "null": str(null_path),
            "key_seed": key_seed,
            "documents": len(wm_documents),
            "encoder": watermarked["encoder"],
            "embedding_reconstruction": {
                "precision_requested": watermarked["encoder"].get(
                    "precision_requested"
                ),
                "max_embed_length": max_embed_length,
                "embed_batch_size": embed_batch_size,
            },
            "feature_map": watermarked["feature_map"],
            "feature_sidecars": {
                "watermarked": {
                    "recorded": watermarked.get("feature_sidecar"),
                    "validation": wm_sidecar_validation,
                },
                "null": {
                    "recorded": null.get("feature_sidecar"),
                    "validation": null_sidecar_validation,
                },
                "usage": {
                    "clean_validation": (
                        "stored detectors were recomputed model-free from exact "
                        "exp19 sidecar rows"
                    ),
                    "attacked_evaluation": (
                        "exact clean sentence text reused validated sidecar rows; "
                        "only novel attacked text was embedded with the recorded "
                        "encoder and public feature map"
                    ),
                    "encoder_seed": sidecar_encoder_seed,
                },
            },
            "attacked_sentence_fingerprint_blake2b128": _sentence_fingerprint(
                all_sentences
            ),
        },
        "detector": {
            "aggregation": recorded_aggregation,
            "recorded_generation_aggregation": recorded_aggregation,
            "field_type": field_type,
            "field_structure": field_structure,
            "rho": float(field_rho),
            "blocks": field_blocks,
            "fusion": field_fusion,
            "soft_sign_channels": (
                int(soft_sign_channels) if is_soft_sign else None
            ),
            "soft_sign_beta": (
                float(soft_sign_beta) if is_soft_sign else None
            ),
            "analytic_statistic": (
                "||U.T w||^2, where U is an orthonormal basis of the "
                "observed sentence-feature span"
                if is_innovation
                else
                "w^T Phi^T (Phi Phi^T)^(-1/2) 1 / "
                "||Phi^T (Phi Phi^T)^(-1/2) 1||"
                if is_decorrelated
                else
                "sum_t sum_m s[t,m] tanh(beta * <phi_t,d_m> / "
                "(1/sqrt(d))) / Q"
                if is_soft_sign
                else
                "max(|w0^T sum(phi)| / ||sum(phi)||, "
                "|Z_anchored|)"
                if field_fusion == "max-absolute-global-anchored-v1"
                else
                "[sqrt(rho) w0^T sum(phi) + sqrt(1-rho) "
                "sum_t ut^T phi_t] / "
                "sqrt(rho ||sum(phi)||^2 + (1-rho) sum_t ||phi_t||^2)"
                if field_structure == "anchored"
                else "w^T sum(phi) / ||sum(phi)||"
            ),
            "analytic_pvalue": (
                "upper normal approximation to span-energy null; controls "
                "are authoritative"
                if is_innovation
                else
                "Bonferroni bound for two two-sided Gaussian tests"
                if field_fusion == "max-absolute-global-anchored-v1"
                else "two-sided Gaussian tail"
                if (is_soft_sign or is_decorrelated) and two_sided
                else "upper standard-normal tail"
            ),
            "local_channel": (
                int(channel + 1) if field_structure == "anchored" else None
            ),
            "analytic_matched_pairs_evaluated": len(wm_documents),
            "scan": {
                "family": "preregistered multiscale contiguous windows",
                "minimum": args.scan_min,
                "maximum_requested": args.scan_max,
                "maximum_effective": effective_scan_max,
                "maximum_default_source": (
                    None if args.scan_max is not None
                    else "exp19 config.sentences"
                ),
                "sizes": args.scan_sizes,
                "stride_fraction": args.scan_stride_fraction,
                "multiplicity": "complete maximum rerun for live and every control key",
            },
        },
        "assumptions": [
            "The public feature map is fixed independently of watermark keys.",
            "Bandwidth, feature seed/dimension, channel, and scan family are fixed before inspecting the live statistic.",
            "Null text and attack randomness are independent of the tested key.",
            "The ideal analytic theorem treats keyed field weights as iid N(0,I).",
            "A single shared keyed field is fixed before generation and reused for every document and attack condition.",
            "Reordering invariance assumes the sentence segmentation itself is unchanged.",
            "Paraphrase FPR requires a separately attacked null split.",
        ],
        "retention_definitions": {
            "source_coverage": "clean source indices represented after attack / clean sentences",
            "exact_text_retention": "clean sentences surviving byte-for-byte / clean sentences",
            "mean_aligned_feature_kernel": "mean realized phi inner product over provenance links",
            "aggregate_feature_cosine": "cosine between clean and attacked summed feature vectors",
            "aggregate_feature_projection": "<v_clean,v_attack>/||v_clean||^2",
            "split_merge_alignment": "source groups allow one-to-many and many-to-one provenance",
            "paraphrase_alignment": "index-only when an external full passage changes segmentation",
        },
        "conditions": output_conditions,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
