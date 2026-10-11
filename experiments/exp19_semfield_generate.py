"""Experiment 19: matched generation with the calibrated SemField proposal.

SemField assigns a sentence a content-local keyed Gaussian score.  For a
public random-feature vector ``phi(s)`` and a shared document field ``w``
derived from the secret key,

    g(s) = w.T phi(s).

The watermarked condition samples the configured number of completed-sentence
candidates and selects using the prospective normalized document score.
The unwatermarked condition samples exactly one completed sentence. Candidates that do
not close a sentence within the token budget are retried using experiment
11's completion-conditioned sampler; they are never padded or silently used.

Selected feature rows are written to a checksummed NPZ sidecar.  This lets the
structural robustness experiment reproduce the exact vectors used during
selection without loading a generator or re-embedding unchanged sentences.

When ``--trajectory-paths B`` is greater than one, the optional
Trajectory-PL mode uses a matched ``B*R`` budget (``R`` is
``--trajectory-candidates``).  It keeps ``B`` complete prefixes alive: the
first round samples ``B*R`` roots, later rounds sample ``R`` continuations for
each live prefix, and only the best final prefix is emitted.  The feature map,
keyed Gaussian field, two-sided detector, temperature, and top-p policy are
unchanged from ordinary polarity-locked PL.

Example::

    python experiments/exp19_semfield_generate.py \
      --generator opt-1.3b --embedder mpnet \
      --corpus-dir data/c4-realnewslike \
      --documents 100 --sentences 40 \
      --feature-dim 2048 --length-scale 0.5 \
      --watermark true --out-dir generations_semfield
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.generation_utils import (  # noqa: E402
    build_prompts,
    sample_completed,
)
from semfield._hf_env import EMBEDDING_CACHE  # noqa: E402
from semfield.generation import load_generator  # noqa: E402
from semfield.semfield import (  # noqa: E402
    BENCHMARK_KEY_DOMAIN,
    AnchoredSemanticField,
    AERSemanticField,
    InnovationSemanticField,
    KeyedSemanticField,
    SoftSignSemanticField,
    benchmark_ensemble_key,
    block_positions,
    decorrelated_field_statistic,
    field_statistic,
    innovation_span_statistic,
    prospective_decorrelated_scores,
    protected_anchored_field_statistic,
    two_sided_field_statistic,
    keyed_gaussian,
)
from semfield.semfield_moment import (  # noqa: E402
    fused_components,
    prospective_fused_scores,
    moment_field_statistic,
    centered_scatter,
)


from semfield.semfield_kernels import build_feature_map  # noqa: E402

SCHEMA_VERSION = 1
KEY_DOMAIN = BENCHMARK_KEY_DOMAIN
VARIANTS = ("none", "semfield")


def experiment_key(seed: int) -> bytes:
    """Derive the reproducible benchmark key (not production key handling)."""

    return benchmark_ensemble_key(seed, 0)


def relational_candidate_scores(
    candidates: np.ndarray,
    history: Sequence[np.ndarray],
    key: bytes,
    harmonics: int = 4,
) -> np.ndarray:
    """Score candidates by their keyed continuous relations to the prefix.

    The frequencies are fixed before seeing any text.  Only the phases are
    derived from the shared key.  This is a continuous carrier and is
    deliberately separate from the projection/sign carriers used by PL.
    """
    values = np.asarray(candidates, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("candidates must be a non-empty matrix")
    if len(history) == 0:
        return np.zeros(values.shape[0], dtype=np.float64)
    prior = np.asarray(history, dtype=np.float64)
    if prior.ndim != 2 or prior.shape[1] != values.shape[1]:
        raise ValueError("history and candidates must have matching widths")
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    prior /= np.maximum(np.linalg.norm(prior, axis=1, keepdims=True), 1e-12)
    relations = values @ prior.T
    count = int(harmonics)
    if count <= 0:
        raise ValueError("relational harmonics must be positive")
    digest = hashlib.sha256(
        bytes.fromhex("706163742d73656d6669656c642d72656c6174696f6e616c2d636172726965722d763100") + bytes(key)
    ).digest()
    phase = np.random.default_rng(
        int.from_bytes(digest[:8], "little")
    ).uniform(-math.pi, math.pi, count)
    omega = np.linspace(0.75, 2.25, count, dtype=np.float64)
    values = math.sqrt(2.0 / count) * np.cos(
        relations[:, :, None] * omega[None, None, :] + phase[None, None, :]
    ).sum(axis=2)
    return values.mean(axis=1)


def relational_document_score(
    embeddings: Sequence[np.ndarray], key: bytes,
    harmonics: int = 4,
) -> float:
    """Return the equal-weight RC score over every unordered sentence pair."""
    if len(embeddings) < 2:
        return 0.0
    rows = np.asarray(embeddings, dtype=np.float64)
    total = 0.0
    pairs = 0
    for index in range(1, rows.shape[0]):
        total += float(relational_candidate_scores(
            rows[index:index + 1], rows[:index], key, harmonics
        )[0]) * index
        pairs += index
    return total / float(max(1, pairs))


def _parse_candidate_schedule(value: str) -> list[int]:
    """Per-sentence candidate counts, as given on the command line."""

    return [int(part) for part in value.split(",") if part.strip()]


def _polarity_objective(
    candidate_phi: np.ndarray,
    aggregate: np.ndarray,
    weights: np.ndarray,
    polarity: int,
    consensus_neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return PL candidate scores and the polarity preferred by each row.

    With one neighbor this is exactly ordinary polarity locking.  With more
    neighbors, each candidate is also represented by the normalized mean of
    its nearest candidates under cosine similarity.  The conservative score
    is the weaker of the candidate's own prospective score and its local
    prototype's prospective score.  This prevents an isolated embedding
    outlier from winning solely because of a large keyed projection.
    """

    values = np.asarray(candidate_phi, dtype=np.float64)
    current = np.asarray(aggregate, dtype=np.float64)
    field_weights = np.asarray(weights, dtype=np.float64)
    prospective = values + current
    norms = np.linalg.norm(prospective, axis=1)
    signed = np.divide(
        prospective @ field_weights,
        norms,
        out=np.full(values.shape[0], -np.inf, dtype=np.float64),
        where=norms > 1e-15,
    )

    k = min(int(consensus_neighbors), values.shape[0])
    if k <= 1:
        preferred = np.where(signed < 0.0, -1, 1).astype(np.int8)
        objective = np.abs(signed) if polarity == 0 else float(polarity) * signed
        return objective, signed, preferred

    row_norms = np.linalg.norm(values, axis=1, keepdims=True)
    unit = np.divide(
        values,
        row_norms,
        out=np.zeros_like(values),
        where=row_norms > 1e-15,
    )
    similarity = unit @ unit.T
    nearest = np.argsort(-similarity, axis=1, kind="stable")[:, :k]
    prototypes = values[nearest].mean(axis=1)
    prototype_norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    prototypes = np.divide(
        prototypes,
        prototype_norms,
        out=values.copy(),
        where=prototype_norms > 1e-15,
    )
    prototype_prospective = prototypes + current
    prototype_document_norms = np.linalg.norm(prototype_prospective, axis=1)
    prototype_signed = np.divide(
        prototype_prospective @ field_weights,
        prototype_document_norms,
        out=np.full(values.shape[0], -np.inf, dtype=np.float64),
        where=prototype_document_norms > 1e-15,
    )

    if polarity == 0:
        positive = np.minimum(signed, prototype_signed)
        negative = np.minimum(-signed, -prototype_signed)
        use_negative = negative > positive
        objective = np.where(use_negative, negative, positive)
        preferred = np.where(use_negative, -1, 1).astype(np.int8)
    else:
        objective = np.minimum(
            float(polarity) * signed,
            float(polarity) * prototype_signed,
        )
        preferred = np.full(values.shape[0], int(polarity), dtype=np.int8)
    return objective, signed, preferred


def _stable_seed(*parts: object) -> int:
    """Order-independent seed for model sampling and tie breaking."""

    digest = hashlib.blake2b(digest_size=8)
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\x00")
    # Leave ample room for sample_completed's deterministic retry offsets.
    return int.from_bytes(digest.digest(), "big") % (2**62)


def _sentence_fingerprint(documents: Sequence[dict]) -> str:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _optional_mean(values: Sequence[float | None]) -> float | None:
    finite = [
        float(value) for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return float(np.mean(finite)) if finite else None


def _distribution(
    values: Sequence[float | None],
) -> dict[str, float | int | None]:
    finite = np.asarray(
        [
            float(value) for value in values
            if value is not None and math.isfinite(float(value))
        ],
        dtype=np.float64,
    )
    if finite.size == 0:
        return {"count": 0, "mean": None, "median": None,
                "minimum": None, "maximum": None}
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "minimum": float(finite.min()),
        "maximum": float(finite.max()),
    }


def _encoder_metadata(encoder, *, name: str, precision: str,
                      output_dim: int | None, input_dim: int) -> dict:
    try:
        resolved_dtype = str(next(encoder.parameters()).dtype).removeprefix(
            "torch."
        )
    except (AttributeError, StopIteration):
        resolved_dtype = None

    revision = None
    try:
        first_module = encoder[0]
        auto_model = getattr(first_module, "auto_model", None)
        revision = getattr(getattr(auto_model, "config", None),
                           "_commit_hash", None)
    except (KeyError, TypeError, IndexError):
        pass

    return {
        "name": name,
        "revision": revision,
        "precision_requested": precision,
        "precision_resolved": resolved_dtype,
        "output_dim_requested": output_dim,
        "input_dim": int(input_dim),
        "normalized_embeddings": True,
    }


def _write_feature_sidecar(path: Path, documents: Sequence[dict],
                           per_document_features: Sequence[np.ndarray],
                           n_features: int) -> dict:
    """Atomically write the exact selected feature rows used by generation."""

    if len(documents) != len(per_document_features):
        raise ValueError("feature arrays do not align with documents")
    offsets = [0]
    rows: list[np.ndarray] = []
    document_ids = []
    for position, (document, values) in enumerate(
        zip(documents, per_document_features)
    ):
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.size == 0:
            matrix = np.zeros((0, n_features), dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != n_features:
            raise ValueError("selected features have an inconsistent width")
        if matrix.shape[0] != len(document.get("sentences", ())):
            raise ValueError("selected features do not align with sentences")
        start = offsets[-1]
        stop = start + matrix.shape[0]
        document["feature_span"] = [int(start), int(stop)]
        offsets.append(stop)
        rows.append(matrix)
        document_ids.append(int(document.get("document_id", position)))

    joined = (
        np.concatenate(rows, axis=0)
        if rows else np.zeros((0, n_features), dtype=np.float32)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            features=joined,
            offsets=np.asarray(offsets, dtype=np.int64),
            document_ids=np.asarray(document_ids, dtype=np.int64),
        )
    temporary.replace(path)
    return {
        "path": path.name,
        "format": "numpy-npz",
        "sha256": _sha256(path),
        "features_key": "features",
        "offsets_key": "offsets",
        "document_ids_key": "document_ids",
        "dtype": str(joined.dtype),
        "shape": [int(value) for value in joined.shape],
        "documents": len(documents),
        "sentence_fingerprint_blake2b128": _sentence_fingerprint(documents),
    }


def _trajectory_append_context(context: str, sentence: str) -> str:
    """Append one completed sentence without damaging chat-template spacing."""

    return context + (
        sentence
        if context.endswith(("\n", " "))
        else " " + sentence
    )


def _trajectory_sample_groups(
    model,
    tokenizer,
    contexts: list[str],
    target: int,
    *,
    seed: int,
    sample_ids: list[int],
    max_new_tokens: int,
    max_context_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_rows: int | None,
    max_retry_rounds: int,
) -> tuple[list[list[str]], dict[str, int]]:
    """Get complete candidate groups for trajectory particles.

    ``sample_completed`` normally finishes every group in one call.  Keeping
    a small outer retry here is important for particle generation: a failed
    child must not silently shorten one path while the other paths advance.
    Each retry uses a fresh seed and asks only for the still-pending contexts.
    The first retry forces a minimum completion length, and later retries let
    the model finish naturally before extracting the first sentence.  These
    recovery modes address instruct prompts that otherwise emit only EOS or a
    fragment after a selected prefix.
    """

    if len(contexts) != len(sample_ids):
        raise ValueError("trajectory contexts and sample_ids must align")
    if target < 1:
        raise ValueError("trajectory target must be positive")
    collected: list[list[str]] = [[] for _ in contexts]
    pending = list(range(len(contexts)))
    requested = completed = rounds = 0
    for retry in range(max_retry_rounds):
        if not pending:
            break
        pending_contexts = [contexts[index] for index in pending]
        groups, accounting = sample_completed(
            model,
            tokenizer,
            pending_contexts,
            target,
            seed=seed + retry * 1_000_003,
            max_new_tokens=max_new_tokens,
            max_context_tokens=max_context_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_rows=max_rows,
            # Retry at this level; nested retries would multiply the budget.
            max_retry_rounds=1,
            sample_ids=[sample_ids[index] for index in pending],
            # Instruct prefixes occasionally emit a fragment with no sentence
            # boundary at all.  Repeating the same unconstrained request can
            # therefore fail forever even though the model can complete a
            # sentence when it is not allowed to stop immediately.  Keep the
            # first draw identical to ordinary PL; on the first retry use the
            # same forced-minimum completion used by the normal catch-up pass.
            # This changes only the recovery path, never the normal sampling
            # distribution or the selected candidate objective.
            min_tokens=(
                0 if retry == 0 else min(64, max(1, max_new_tokens - 1))
            ),
            # If the model still emits no boundary, let it finish naturally
            # and recover the first sentence from the returned text.  This is
            # a recovery-only path; ordinary candidates still use sentence
            # stopping, while native-mode candidates are accepted only after
            # the ordinary path has failed for this context.
            stop_mode="sentence" if retry < 2 else "native",
            proposal_mode="iid",
        )
        requested += int(accounting["requested"])
        completed += int(accounting["completed"])
        rounds += 1
        next_pending: list[int] = []
        for local_index, original_index in enumerate(pending):
            collected[original_index].extend(groups[local_index])
            if len(collected[original_index]) < target:
                next_pending.append(original_index)
        pending = next_pending

    if pending:
        bad = [int(sample_ids[index]) for index in pending]
        raise RuntimeError(
            "trajectory generation could not complete candidate groups for "
            f"particle IDs {bad} after {max_retry_rounds} retries"
        )
    return [group[:target] for group in collected], {
        "requested": requested,
        "completed": completed,
        "rounds": rounds,
    }


def generate_trajectory_variant(
    model,
    tokenizer,
    prompts: list[str],
    feature_map,
    embedder,
    *,
    n_sentences: int,
    key: bytes,
    document_ids: Sequence[int],
    channel: int,
    seed: int,
    max_new_tokens: int,
    max_context_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_rows: int | None,
    max_retry_rounds: int,
    trajectory_paths: int,
    trajectory_candidates: int,
) -> tuple[list[dict], list[np.ndarray], dict]:
    """Generate SemField using several complete document trajectories.

    The ordinary sampler, MPNet feature map, keyed field, and two-sided
    detector are unchanged.  A fixed rollout budget is spread across several
    complete prefixes: each path receives ``trajectory_candidates`` candidates
    per sentence, and the best final path is emitted.  Keeping one survivor
    per parent prevents all expansions from collapsing back onto one prefix.
    """

    if trajectory_paths < 2:
        raise ValueError("trajectory_paths must be at least two")
    if trajectory_candidates < 2:
        raise ValueError("trajectory_candidates must be at least two")
    if len(prompts) != len(document_ids):
        raise ValueError("prompts and document_ids must align")
    if getattr(feature_map, "stateful", False):
        raise ValueError("trajectory PL requires the ordinary stateless feature map")
    n_documents = len(prompts)
    n_features = int(feature_map.n_features)
    fields = [
        KeyedSemanticField(feature_map, key, channel=channel)
        for index in range(n_documents)
    ]

    # A state is one fluent document prefix.  Only its selected sentence rows
    # are retained; unselected candidates are diagnostic values, not output.
    states: list[list[dict]] = [[] for _ in range(n_documents)]
    total_requested = total_completed = total_usable = 0
    retry_context_rounds = 0
    sampling_seconds = embedding_seconds = feature_seconds = 0.0
    all_objective_ranges: list[float] = []
    all_objective_stds: list[float] = []
    all_objective_unique_counts: list[float] = []
    all_objective_top_gaps: list[float] = []
    all_candidate_means: list[float | None] = []
    all_selected_scores: list[float] = []
    path_final_scores: list[float] = []

    def record_pool(values: np.ndarray, features: np.ndarray) -> None:
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            all_objective_ranges.append(float(np.ptp(finite)))
            all_objective_stds.append(float(np.std(finite)))
            all_objective_unique_counts.append(float(
                np.unique(np.round(finite, decimals=10)).size
            ))
            if finite.size > 1:
                ordered = np.partition(finite, finite.size - 2)
                all_objective_top_gaps.append(float(
                    ordered[-1] - ordered[-2]
                ))
        if features.shape[0] > 1:
            values64 = features.astype(np.float64)
            pair_numerator = (
                float(np.square(values64.sum(axis=0)).sum())
                - float(np.square(values64).sum())
            )
            all_candidate_means.append(
                pair_numerator / float(features.shape[0] * (features.shape[0] - 1))
            )
        else:
            all_candidate_means.append(None)

    # First sample: use the full matched budget, then retain B roots.  This
    # preserves PL's strong initial order statistic while opening multiple
    # independently conditioned future contexts.
    root_target = int(trajectory_paths * trajectory_candidates)
    sampling_started = time.perf_counter()
    root_groups, accounting = _trajectory_sample_groups(
        model, tokenizer, list(prompts), root_target,
        seed=_stable_seed("exp19-trajectory", seed, "root"),
        sample_ids=[int(value) for value in document_ids],
        max_new_tokens=max_new_tokens,
        max_context_tokens=max_context_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_rows=max_rows,
        max_retry_rounds=max_retry_rounds,
    )
    sampling_seconds += time.perf_counter() - sampling_started
    total_requested += int(accounting["requested"])
    total_completed += int(accounting["completed"])
    retry_context_rounds += max(0, int(accounting["rounds"]) - 1)
    total_usable += n_documents * root_target
    flat_root = [sentence for group in root_groups for sentence in group]
    embedding_started = time.perf_counter()
    root_vectors = np.asarray(embedder(flat_root), dtype=np.float64)
    embedding_seconds += time.perf_counter() - embedding_started
    feature_started = time.perf_counter()
    root_features = np.asarray(feature_map.transform(root_vectors), dtype=np.float32)
    feature_seconds += time.perf_counter() - feature_started
    cursor = 0
    for document_index in range(n_documents):
        candidates = root_features[cursor:cursor + root_target]
        vectors = root_vectors[cursor:cursor + root_target]
        cursor += root_target
        scores = fields[document_index].score_features(candidates)
        root_objective = np.abs(scores)
        record_pool(root_objective, candidates)
        # Stable sorting makes ties reproducible without adding a key-dependent
        # tie breaker.  Each retained root gets its own polarity lock.
        roots = np.argsort(-root_objective, kind="stable")[:trajectory_paths]
        for root_index in roots:
            root_index = int(root_index)
            score = float(scores[root_index])
            polarity = 1 if score >= 0.0 else -1
            feature = candidates[root_index].copy()
            sentence = str(root_groups[document_index][root_index])
            states[document_index].append({
                "context": _trajectory_append_context(prompts[document_index], sentence),
                "sentences": [sentence],
                "features": [feature],
                "aggregate": feature.astype(np.float64).copy(),
                "polarity": polarity,
                "selected_scores": [float(root_objective[root_index])],
                "candidate_scores": [
                    [float(value) for value in root_objective]
                ],
                "candidate_means": [
                    all_candidate_means[-1]
                ],
                "selected_indices": [root_index],
                "ties": [bool(np.count_nonzero(
                    np.isclose(root_objective, root_objective[root_index],
                               rtol=1e-12, atol=1e-12)
                ) > 1)],
            })

    # Every later round expands all live paths in one batched vLLM request.
    # One child is retained per parent, so the between-prefix term cannot be
    # erased by a global top-B prune.
    for step in range(1, n_sentences):
        flat_states: list[tuple[int, int]] = []
        contexts: list[str] = []
        sample_ids: list[int] = []
        for document_index in range(n_documents):
            for path_index, state in enumerate(states[document_index]):
                flat_states.append((document_index, path_index))
                contexts.append(state["context"])
                sample_ids.append(
                    int(document_ids[document_index]) * trajectory_paths
                    + path_index
                )
        sampling_started = time.perf_counter()
        groups, accounting = _trajectory_sample_groups(
            model, tokenizer, contexts, trajectory_candidates,
            seed=_stable_seed("exp19-trajectory", seed, "step", step),
            sample_ids=sample_ids,
            max_new_tokens=max_new_tokens,
            max_context_tokens=max_context_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_rows=max_rows,
            max_retry_rounds=max_retry_rounds,
        )
        sampling_seconds += time.perf_counter() - sampling_started
        total_requested += int(accounting["requested"])
        total_completed += int(accounting["completed"])
        retry_context_rounds += max(0, int(accounting["rounds"]) - 1)
        total_usable += len(contexts) * trajectory_candidates
        flat = [sentence for group in groups for sentence in group]
        embedding_started = time.perf_counter()
        vectors = np.asarray(embedder(flat), dtype=np.float64)
        embedding_seconds += time.perf_counter() - embedding_started
        feature_started = time.perf_counter()
        transformed = np.asarray(feature_map.transform(vectors), dtype=np.float32)
        feature_seconds += time.perf_counter() - feature_started
        cursor = 0
        for local_index, (document_index, path_index) in enumerate(flat_states):
            state = states[document_index][path_index]
            candidates = transformed[
                cursor:cursor + trajectory_candidates
            ]
            group = groups[local_index]
            cursor += trajectory_candidates
            values = candidates.astype(np.float64)
            prospective = values + state["aggregate"][None, :]
            norms = np.linalg.norm(prospective, axis=1)
            signed = np.divide(
                prospective @ fields[document_index].weights,
                norms,
                out=np.full(trajectory_candidates, -np.inf, dtype=np.float64),
                where=norms > 1e-15,
            )
            objective = float(state["polarity"]) * signed
            record_pool(objective, candidates)
            best = np.flatnonzero(np.isclose(
                objective, float(objective.max()), rtol=1e-12, atol=1e-12
            ))
            chosen = int(best[0])
            chosen_feature = candidates[chosen].copy()
            chosen_sentence = str(group[chosen])
            state["context"] = _trajectory_append_context(
                state["context"], chosen_sentence
            )
            state["sentences"].append(chosen_sentence)
            state["features"].append(chosen_feature)
            state["aggregate"] += chosen_feature.astype(np.float64)
            state["selected_scores"].append(float(objective[chosen]))
            state["candidate_scores"].append([
                float(value) for value in objective
            ])
            state["candidate_means"].append(
                all_candidate_means[-1]
            )
            state["selected_indices"].append(chosen)
            state["ties"].append(bool(best.size > 1))
        print(
            f"    trajectory: sentence {step + 1}/{n_sentences}, "
            f"{n_documents * trajectory_paths} paths active",
            flush=True,
        )

    documents: list[dict] = []
    feature_matrices: list[np.ndarray] = []
    design_effects: list[float] = []
    for document_index in range(n_documents):
        final_scores = np.asarray([
            float(state["polarity"])
            * float(fields[document_index].weights @ state["aggregate"])
            / max(float(np.linalg.norm(state["aggregate"])), 1e-15)
            for state in states[document_index]
        ], dtype=np.float64)
        chosen_path = int(np.argmax(final_scores))
        state = states[document_index][chosen_path]
        matrix = np.asarray(state["features"], dtype=np.float32)
        statistic = two_sided_field_statistic(
            matrix, fields[document_index].weights
        )
        design_effects.append(float(statistic.design_effect))
        path_final_scores.extend(float(value) for value in final_scores)
        all_selected_scores.extend(float(value) for value in state["selected_scores"])
        documents.append({
            "document_id": int(document_ids[document_index]),
            "text": " ".join(state["sentences"]),
            "sentences": list(state["sentences"]),
            "sentences_emitted": len(state["sentences"]),
            "selected_scores": list(state["selected_scores"]),
            "candidate_scores": list(state["candidate_scores"]),
            "candidate_kernels": [None] * len(state["sentences"]),
            "candidate_mean_pair_cosines": list(state["candidate_means"]),
            "predicted_one_step_lifts": [None] * len(state["sentences"]),
            "selected_indices": list(state["selected_indices"]),
            "ties": list(state["ties"]),
            "raw_embeddings": None,
            "polarity": int(state["polarity"]),
            "moment": None,
            "detector": asdict(statistic),
            "trajectory_path": chosen_path,
            "trajectory_final_score": float(final_scores[chosen_path]),
            "trajectory_path_scores": [float(value) for value in final_scores],
        })
        feature_matrices.append(matrix)

    elapsed = sampling_seconds + embedding_seconds + feature_seconds
    total_emitted = sum(row["sentences_emitted"] for row in documents)
    summary = {
        "scheme": "semfield-trajectory-pl",
        "variant": "semfield",
        "feature_map": feature_map.metadata(),
        "selection_objective": "trajectory-polarity-locked",
        "field_structure": "global",
        "field_rho": 1.0,
        "field_blocks": None,
        "semfield_candidates": int(trajectory_candidates),
        "trajectory": {
            "paths": int(trajectory_paths),
            "candidates_per_path": int(trajectory_candidates),
            "root_candidates": int(root_target),
            "candidates_per_round": int(root_target),
            "selection": "best child per path, then best final path",
            "sampling": "ordinary iid complete-sentence candidates",
            "matched_budget_note": (
                "B*R=64 is the matched PL-64 budget when paths=8 and "
                "candidates_per_path=8"
            ),
        },
        "documents": int(n_documents),
        "documents_complete": int(sum(
            row["sentences_emitted"] == n_sentences for row in documents
        )),
        "document_completion_rate": float(sum(
            row["sentences_emitted"] == n_sentences for row in documents
        ) / max(1, n_documents)),
        "sentences_target": int(n_documents * n_sentences),
        "sentences_emitted": int(total_emitted),
        "sentence_target_completion_rate": float(
            total_emitted / max(1, n_documents * n_sentences)
        ),
        "requested_candidates": int(total_requested),
        "completed_candidates": int(total_completed),
        "usable_candidates": int(total_usable),
        "discarded_completed_candidates": int(max(
            0, total_completed - total_usable
        )),
        "requested_candidates_per_sentence": float(
            total_requested / max(1, total_emitted)
        ),
        "completed_candidates_per_sentence": float(
            total_completed / max(1, total_emitted)
        ),
        "completion_rate": float(total_completed / max(1, total_requested)),
        "retry_context_rounds": int(retry_context_rounds),
        "candidate_objective_range": _distribution(all_objective_ranges),
        "candidate_objective_std": _distribution(all_objective_stds),
        "candidate_objective_unique_count": _distribution(
            all_objective_unique_counts
        ),
        "candidate_objective_top_gap": _distribution(all_objective_top_gaps),
        "candidate_mean_pair_cosine": _distribution(all_candidate_means),
        "selected_field_score": _distribution(all_selected_scores),
        "trajectory_final_score": _distribution(path_final_scores),
        "field_design_effect": _distribution(design_effects),
        "seconds_total": float(elapsed),
        "seconds_sampling": float(sampling_seconds),
        "seconds_embedding": float(embedding_seconds),
        "seconds_feature_transform": float(feature_seconds),
        "seconds_per_document": float(elapsed / max(1, n_documents)),
    }
    return documents, feature_matrices, summary


def generate_variant(
    model,
    tokenizer,
    prompts: list[str],
    variant: str,
    feature_map,
    embedder,
    *,
    n_sentences: int,
    key: bytes,
    document_ids: Sequence[int] | None = None,
    channel: int,
    seed: int,
    max_new_tokens: int,
    max_context_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_rows: int | None,
    max_retry_rounds: int,
    semfield_candidates: int = 2,
    field_structure: str = "global",
    field_rho: float = 1.0,
    field_blocks: int | None = None,
    candidate_schedule: Sequence[int] | None = None,
    consensus_neighbors: int = 1,
    selection_objective: str = "local",
    moment_lambda: float = 0.5,
    correlation_epsilon: float = 0.05,
    protected_global_epsilon: float = 0.10,
    soft_sign_channels: int = 8,
    soft_sign_beta: float = 4.0,
    proposal_mode: str = "iid",
    proposal_branches: int = 8,
    proposal_prefix_tokens: int = 8,
    proposal_pool_size: int = 32,
    proposal_temperature: float = 1.3,
    proposal_top_k: int = 50,
    proposal_first_token_logprob_gap: float = 5.0,
    relational_harmonics: int = 4,
    sampling_seed_offset: int = 0,
    resume: Sequence[dict] | None = None,
    trajectory_paths: int = 1,
    trajectory_candidates: int = 8,
    capture_candidate_texts: bool = False,
) -> tuple[list[dict], list[np.ndarray], dict]:
    """Generate a matched condition with local or detector-aligned selection.

    ``resume``, if given, is one dict per prompt holding sentences already
    selected for that document in an earlier run (same keys as one entry of
    the returned ``documents`` list, at minimum "sentences" and "features" --
    a (k, n_features) float32 array). Documents are seeded from that state
    instead of empty, the ordinary step-by-step main pass is skipped (since
    it already ran to produce that state), and generation goes straight to
    the catch-up pass below. Because every per-document seed used anywhere in
    this function is a pure function of (seed, variant, round_key, that
    document's own document_id) -- never of what else is in the batch or
    what ran before it -- resuming a subset of documents this way computes
    exactly what a single full run over all documents would have computed
    for those same documents.
    """

    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    if capture_candidate_texts and resume is not None:
        raise ValueError(
            "candidate-text capture is only supported for a fresh generation"
        )
    if document_ids is None:
        document_ids = list(range(len(prompts)))
    if len(prompts) != len(document_ids):
        raise ValueError("prompts and document_ids must have equal length")
    n_documents = len(prompts)
    if resume is not None and len(resume) != n_documents:
        raise ValueError("resume state must align with prompts")
    if trajectory_paths > 1 and variant == "semfield":
        if resume is not None:
            raise ValueError("trajectory PL does not support resume state yet")
        if field_structure != "global":
            raise ValueError("trajectory PL requires field_structure='global'")
        if selection_objective != "polarity-locked":
            raise ValueError(
                "trajectory PL requires selection_objective='polarity-locked'"
            )
        if proposal_mode != "iid":
            raise ValueError("trajectory PL requires ordinary iid proposals")
        return generate_trajectory_variant(
            model,
            tokenizer,
            prompts,
            feature_map,
            embedder,
            n_sentences=n_sentences,
            key=key,
            document_ids=document_ids,
            channel=channel,
            seed=seed,
            max_new_tokens=max_new_tokens,
            max_context_tokens=max_context_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_rows=max_rows,
            max_retry_rounds=max_retry_rounds,
            trajectory_paths=trajectory_paths,
            trajectory_candidates=trajectory_candidates,
        )
    if semfield_candidates < 2:
        raise ValueError("semfield_candidates must be at least two")
    if field_structure not in {"global", "anchored", "aer"}:
        raise ValueError("field_structure must be 'global', 'anchored', or 'aer'")
    if not math.isfinite(field_rho) or not 0.0 <= float(field_rho) <= 1.0:
        raise ValueError("field_rho must lie in [0, 1]")
    if field_blocks is not None and int(field_blocks) <= 0:
        raise ValueError("field_blocks must be positive")
    if field_structure == "anchored" and selection_objective == "moment-hybrid":
        raise ValueError("moment-hybrid is not supported with anchored fields")
    if (selection_objective == "protected-anchored-pl"
            and field_structure != "anchored"):
        raise ValueError(
            "protected-anchored-pl requires field_structure='anchored'"
        )
    if consensus_neighbors <= 0:
        raise ValueError("consensus_neighbors must be positive")
    if not math.isfinite(moment_lambda) or moment_lambda < 0.0:
        raise ValueError("moment_lambda must be finite and non-negative")
    if not math.isfinite(correlation_epsilon) or correlation_epsilon < 0.0:
        raise ValueError("correlation_epsilon must be finite and non-negative")
    if (not math.isfinite(protected_global_epsilon)
            or protected_global_epsilon < 0.0):
        raise ValueError(
            "protected_global_epsilon must be finite and non-negative"
        )
    if soft_sign_channels <= 0:
        raise ValueError("soft_sign_channels must be positive")
    if not math.isfinite(soft_sign_beta) or soft_sign_beta <= 0.0:
        raise ValueError("soft_sign_beta must be finite and positive")
    if int(relational_harmonics) <= 0:
        raise ValueError("relational_harmonics must be positive")
    if selection_objective == "soft-sign" and field_structure != "global":
        raise ValueError("soft-sign requires field_structure='global'")
    if (selection_objective == "decorrelated-pl"
            and field_structure != "global"):
        raise ValueError("decorrelated-pl requires field_structure='global'")
    if proposal_mode not in {
        "iid", "arithmetic", "branch-stratified", "entropy-stratified",
        "hot-first-token",
    }:
        raise ValueError("unknown proposal_mode")
    if proposal_branches < 2:
        raise ValueError("proposal_branches must be at least two")
    if proposal_prefix_tokens <= 0 or proposal_prefix_tokens >= max_new_tokens:
        raise ValueError(
            "proposal_prefix_tokens must lie in [1, max_new_tokens)"
        )
    if proposal_mode == "hot-first-token":
        largest_target = max(
            [semfield_candidates, *(candidate_schedule or ())]
        )
        if proposal_pool_size < largest_target:
            raise ValueError("proposal_pool_size must cover candidate counts")
        if proposal_top_k < proposal_pool_size:
            raise ValueError("proposal_top_k must cover proposal_pool_size")
        if (not math.isfinite(proposal_temperature)
                or proposal_temperature <= 0.0):
            raise ValueError("proposal_temperature must be finite and positive")
        if (not math.isfinite(proposal_first_token_logprob_gap)
                or proposal_first_token_logprob_gap < 0.0):
            raise ValueError(
                "proposal_first_token_logprob_gap must be finite and non-negative"
            )
    if selection_objective not in {
        "local", "self-normalized", "polarity-locked", "correlation-aware-pl",
        "protected-anchored-pl",
        "moment-hybrid", "soft-sign", "decorrelated-pl", "innovation-pl",
        "relational-continuous",
    }:
        raise ValueError("unknown selection_objective")
    target = 1 if variant == "none" else int(semfield_candidates)
    if not getattr(model, "is_vllm_generator", False):
        raise TypeError("SemField generation requires VLLMCandidateGenerator")
    if resume is None:
        contexts = list(prompts)
        sentences: list[list[str]] = [[] for _ in prompts]
        features: list[list[np.ndarray]] = [[] for _ in prompts]
        raw_embeddings: list[list[np.ndarray]] = [[] for _ in prompts]
        selected_scores: list[list[float]] = [[] for _ in prompts]
        candidate_scores: list[list[list[float]]] = [[] for _ in prompts]
        candidate_kernels: list[list[float | None]] = [[] for _ in prompts]
        predicted_lifts: list[list[float | None]] = [[] for _ in prompts]
        selected_indices: list[list[int]] = [[] for _ in prompts]
        ties: list[list[bool]] = [[] for _ in prompts]
        candidate_mean_pair_cosines: list[list[float | None]] = [
            [] for _ in prompts
        ]
        candidate_texts: list[list[list[str]]] = [[] for _ in prompts]
    else:
        sentences = [list(r["sentences"]) for r in resume]
        features = [
            list(np.asarray(r["features"], dtype=np.float32)) for r in resume
        ]
        # Stateful innovation maps need the original normalized sentence
        # embeddings to reconstruct the running residual context.  They are
        # deliberately re-embedded from text on resume; transformed feature
        # rows alone cannot recover the raw vectors.
        raw_embeddings = [[] for _ in resume]
        if selection_objective == "relational-continuous":
            for document_i, row in enumerate(resume):
                raw_embeddings[document_i] = [
                    np.asarray(value, dtype=np.float64)
                    for value in row.get("raw_embeddings", [])
                ]
        selected_scores = [list(r.get("selected_scores", [])) for r in resume]
        candidate_scores = [list(r.get("candidate_scores", [])) for r in resume]
        candidate_kernels = [list(r.get("candidate_kernels", [])) for r in resume]
        predicted_lifts = [
            list(r.get("predicted_one_step_lifts", [])) for r in resume
        ]
        selected_indices = [list(r.get("selected_indices", [])) for r in resume]
        ties = [list(r.get("ties", [])) for r in resume]
        candidate_mean_pair_cosines = [
            list(r.get("candidate_mean_pair_cosines", [])) for r in resume
        ]
        candidate_texts = [
            [list(map(str, values)) for values in r.get("candidate_texts", [])]
            for r in resume
        ]
        contexts = [
            prompts[i] + (" " + " ".join(sentences[i]) if sentences[i] else "")
            for i in range(n_documents)
        ]
    fields = [
        (
            AERSemanticField(
                feature_map, key, blocks=(field_blocks or 4),
                channel=channel,
            )
            if field_structure == "aer" else
            AnchoredSemanticField(
                feature_map, key, rho=field_rho, channel=channel,
                blocks=field_blocks,
            )
            if field_structure in {"anchored", "aer"}
            else InnovationSemanticField(
                feature_map, key, channel=channel,
            ) if selection_objective == "innovation-pl"
            else SoftSignSemanticField(
                feature_map.n_features, key,
                channels=soft_sign_channels, beta=soft_sign_beta,
                channel=channel,
            ) if selection_objective == "soft-sign"
            else KeyedSemanticField(feature_map, key, channel=channel)
        )
        for i in range(n_documents)
    ]
    soft_sign_score_sums = [0.0 for _ in range(n_documents)]
    soft_sign_variances = [0.0 for _ in range(n_documents)]
    if selection_objective == "soft-sign":
        for document_i, field in enumerate(fields):
            if not features[document_i]:
                continue
            resumed = field.detect_features(
                np.asarray(features[document_i], dtype=np.float64)
            )
            soft_sign_score_sums[document_i] = float(resumed.score_sum)
            soft_sign_variances[document_i] = float(resumed.null_variance)
    moment_weights = [
        keyed_gaussian(field.key, feature_map.n_features,
                       channel=channel + 1)
        for field in fields
    ]
    aggregates = [
        (np.asarray(features[i], dtype=np.float64).sum(axis=0)
         if resume is not None and len(features[i])
         else np.zeros(feature_map.n_features, dtype=np.float64))
        for i in range(n_documents)
    ]
    anchored_local_scores = [0.0 for _ in range(n_documents)]
    anchored_local_variances = [0.0 for _ in range(n_documents)]
    # Rows sharing a relative-position block share one keyed weight, so the
    # local variance is the energy of each block's running sum, not the
    # row-wise energy.  These track that sum per block.
    anchored_block_sums: list[dict[int, np.ndarray]] = [
        {} for _ in range(n_documents)
    ]

    def local_position(row_index: int) -> int:
        """Relative-position block for the row about to be emitted.

        The document length is known up front (``n_sentences``), so the block
        index is the same one the detector recomputes from the observed
        sentence count.  ``field_blocks=None`` reproduces absolute indexing."""
        block_count = field_blocks if field_blocks is not None else (
            fields[0].blocks if field_structure == "aer" else None
        )
        return block_positions(int(n_sentences), block_count)[
            min(int(row_index), int(n_sentences) - 1)
        ]

    def accumulate_local(document_i: int, position: int, value: np.ndarray):
        running = anchored_block_sums[document_i].get(position)
        if running is None:
            anchored_local_variances[document_i] += float(np.dot(value, value))
            anchored_block_sums[document_i][position] = value.copy()
        else:
            anchored_local_variances[document_i] += float(
                2.0 * np.dot(running, value) + np.dot(value, value)
            )
            running += value

    if field_structure in {"anchored", "aer"} and resume is not None:
        for document_i, rows in enumerate(features):
            for row_index, row in enumerate(rows):
                value = np.asarray(row, dtype=np.float64)
                position = local_position(row_index)
                anchored_local_scores[document_i] += float(
                    np.dot(fields[document_i].local_weights(position), value)
                )
                accumulate_local(document_i, position, value)
    moment_means = []
    moment_scatters = []
    feature_energies = []
    for rows in features:
        matrix = (np.asarray(rows, dtype=np.float64)
                  if len(rows) else np.zeros((0, feature_map.n_features), dtype=np.float64))
        if matrix.shape[0]:
            moment_means.append(matrix.mean(axis=0))
            moment_scatters.append(centered_scatter(matrix))
            feature_energies.append(float(np.square(matrix).sum()))
        else:
            moment_means.append(np.zeros(feature_map.n_features, dtype=np.float64))
            moment_scatters.append(np.zeros(feature_map.n_features, dtype=np.float64))
            feature_energies.append(0.0)
    polarities = (
        np.asarray(
            [int(r.get("polarity") or 0) for r in resume], dtype=np.int8
        )
        if resume is not None
        else np.zeros(n_documents, dtype=np.int8)
    )
    started = time.perf_counter()
    embedding_seconds = 0.0
    if getattr(feature_map, "stateful", False) and resume is not None:
        resume_texts = [
            sentence
            for document_sentences in sentences
            for sentence in document_sentences
        ]
        if resume_texts:
            resume_started = time.perf_counter()
            resume_vectors = np.asarray(embedder(resume_texts), dtype=np.float64)
            embedding_seconds += time.perf_counter() - resume_started
            if resume_vectors.ndim != 2 or resume_vectors.shape[0] != len(resume_texts):
                raise ValueError(
                    "stateful resume embedding count does not match saved sentences"
                )
            cursor = 0
            for document_i, document_sentences in enumerate(sentences):
                count = len(document_sentences)
                raw_embeddings[document_i] = [
                    np.asarray(row, dtype=np.float64)
                    for row in resume_vectors[cursor:cursor + count]
                ]
                cursor += count
    total_requested = 0
    total_completed = 0
    total_usable = 0
    retry_context_rounds = 0
    sampling_seconds = 0.0
    feature_seconds = 0.0
    def run_round(round_active: list[int], round_key: object) -> list[int]:
        """Try to add one sentence to each document in round_active, using
        round_key for seeding (an int step number for the main pass, a
        distinct marker for the catch-up pass below). Returns the documents
        that still need more sentences afterward, whether because they
        failed this round or simply have not reached n_sentences yet."""
        nonlocal total_requested, total_completed, total_usable
        nonlocal retry_context_rounds, sampling_seconds, embedding_seconds, feature_seconds

        round_contexts = [contexts[index] for index in round_active]
        sampling_started = time.perf_counter()
        # The catch-up pass already retries a stuck document across many
        # outer rounds (up to max_retry_rounds catch-up rounds, each with
        # its own fresh seed). Letting sample_completed ALSO retry up to
        # max_retry_rounds times internally, within a single one of those
        # rounds, multiplies the two budgets together -- up to
        # max_retry_rounds^2 total attempts for one stubborn document. The
        # main pass (round_key is the int step) keeps the original,
        # unchanged budget; only catch-up rounds (round_key is a tuple)
        # use a small inner budget so a hard document is retried across
        # many *rounds* instead of grinding inside one.
        inner_retry_rounds = (
            min(max_retry_rounds, 5) if isinstance(round_key, tuple)
            else max_retry_rounds
        )
        # A document stuck in catch-up either has a context that looks like
        # the end of an article, making the model emit nothing at all every
        # time, or a continuation that opens on an abbreviation ("Gen.",
        # "Rep.") and gets cut off there before its real sentence is ever
        # generated. A minimum token count fixes both: it forbids stopping
        # (EOS or the sentence-boundary stop strings below) until that many
        # tokens are out, so the model is forced past an empty response and
        # past an abbreviation's period alike. Temperature and top_p stay at
        # the run's own configured values on every pass, main or catch-up,
        # so a recovered sentence is sampled the same way as the rest of the
        # document.
        # A longer forced minimum costs nothing on documents that would have
        # closed a sentence early anyway: first_sentence() still returns the
        # first valid boundary from the start of the text, so extra forced
        # tokens past that point are simply discarded, never change what
        # gets kept. It only matters for a document contaminated enough
        # (several sentences of boilerplate in a row, say) that 32 wasn't
        # enough room to push past it.
        round_min_tokens = 64 if isinstance(round_key, tuple) else 0
        # Per-step candidate budget. The value of one more candidate falls as
        # 1/sqrt(t * design_effect): measured across all three selection
        # objectives the gain at the last sentence is ~0.14 of the gain at the
        # first (r > 0.997 against 1/sqrt(t+1)), so a flat budget spends most
        # of its samples where they buy least. The main pass advances every
        # document in lockstep, so round_key is the step and indexes the
        # schedule directly. Catch-up rounds mix documents sitting at
        # different steps, so they take the schedule's largest entry: the
        # point of that pass is to finish a document at all, and it touches
        # few of them. Actual spend is recorded in the summary's
        # requested_candidates_per_sentence, so the budget reported for a run
        # is measured rather than assumed.
        if variant == "none" or candidate_schedule is None:
            round_target = target
        elif isinstance(round_key, int):
            round_target = int(candidate_schedule[round_key])
        else:
            round_target = int(max(candidate_schedule))
        groups, accounting = sample_completed(
            model,
            tokenizer,
            round_contexts,
            round_target,
            seed=_stable_seed("exp19-model", seed, variant, round_key) + sampling_seed_offset,
            max_new_tokens=max_new_tokens,
            max_context_tokens=max_context_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_rows=max_rows,
            max_retry_rounds=inner_retry_rounds,
            sample_ids=[int(document_ids[index]) for index in round_active],
            min_tokens=round_min_tokens,
            proposal_mode=(
                proposal_mode if variant == "semfield" else "iid"
            ),
            proposal_branches=proposal_branches,
            proposal_prefix_tokens=proposal_prefix_tokens,
            proposal_pool_size=proposal_pool_size,
            proposal_temperature=proposal_temperature,
            proposal_top_k=proposal_top_k,
            proposal_first_token_logprob_gap=proposal_first_token_logprob_gap,
        )
        sampling_seconds += time.perf_counter() - sampling_started
        total_requested += int(accounting["requested"])
        total_completed += int(accounting["completed"])
        baseline_requests = len(round_contexts) * (
            proposal_pool_size
            if variant == "semfield" and proposal_mode == "hot-first-token"
            else round_target
        )
        retry_context_rounds += max(
            0, (int(accounting["requested"]) - baseline_requests) // round_target
        )

        failed_local = set(int(value) for value in accounting["failed_indices"])
        spans: list[tuple[int, int, int, list[str]]] = []
        flat: list[str] = []
        still_needs_more: list[int] = []
        for local_i, document_i in enumerate(round_active):
            group = groups[local_i]
            if local_i in failed_local or len(group) != round_target:
                still_needs_more.append(document_i)
                continue
            start, stop = len(flat), len(flat) + round_target
            spans.append((start, stop, document_i, group))
            flat.extend(group)

        if not flat:
            return still_needs_more

        embedding_started = time.perf_counter()
        vectors = embedder(flat)
        embedding_seconds += time.perf_counter() - embedding_started
        feature_started = time.perf_counter()
        # The float32 rows are both scored now and persisted later, preventing
        # a sidecar round trip from changing an attack statistic in its last
        # bits relative to generation-time selection.
        if getattr(feature_map, "stateful", False):
            # A residual feature depends on the selected prefix of this
            # document.  Compute each candidate pool against that prefix; the
            # embedding model is still called once for the flattened batch.
            phi_by_document = {
                document_i: np.ascontiguousarray(
                    feature_map.transform_candidates(
                        vectors[start:stop],
                        np.asarray(raw_embeddings[document_i], dtype=np.float64),
                    ),
                    dtype=np.float32,
                )
                for start, stop, document_i, _group in spans
            }
            phi_all = None
        else:
            phi_all = np.ascontiguousarray(
                feature_map.transform(vectors), dtype=np.float32
            )
            phi_by_document = None
        feature_seconds += time.perf_counter() - feature_started

        for start, stop, document_i, group in spans:
            if capture_candidate_texts:
                candidate_texts[document_i].append([str(value) for value in group])
            candidate_phi = (
                phi_by_document[document_i]
                if phi_by_document is not None
                else phi_all[start:stop]
            )
            if round_target > 1:
                values64 = candidate_phi.astype(np.float64)
                pair_numerator = (
                    float(np.square(values64.sum(axis=0)).sum())
                    - float(np.square(values64).sum())
                )
                candidate_mean_pair_cosines[document_i].append(
                    pair_numerator / float(round_target * (round_target - 1))
                )
            else:
                # JSON has no NaN value.  A singleton pool has no pairwise
                # cosine, so persist the diagnostic as null.
                candidate_mean_pair_cosines[document_i].append(None)
            field = fields[document_i]
            if variant == "semfield" and selection_objective == "relational-continuous":
                # Relational-continuous carrier: score a candidate by its
                # keyed continuous relations to the already emitted prefix.
                # The first sentence is unmodified because it has no edge.
                if not raw_embeddings[document_i]:
                    objective = np.zeros(round_target, dtype=np.float64)
                    chosen = 0
                    tied = round_target > 1
                else:
                    signed_scores = relational_candidate_scores(
                        vectors[start:stop], raw_embeddings[document_i],
                        key, relational_harmonics,
                    )
                    first_relational = len(raw_embeddings[document_i]) == 1
                    if first_relational and polarities[document_i] == 0:
                        objective = np.abs(signed_scores)
                    else:
                        if polarities[document_i] == 0:
                            polarities[document_i] = 1
                        objective = float(polarities[document_i]) * signed_scores
                    largest = float(objective.max())
                    best = np.flatnonzero(np.isclose(
                        objective, largest, rtol=1e-12, atol=1e-12
                    ))
                    chosen = int(np.random.default_rng(_stable_seed(
                        "exp19-tie", seed, variant, round_key,
                        int(document_ids[document_i])
                    )).choice(best))
                    tied = best.size > 1
                    if first_relational and polarities[document_i] == 0:
                        polarities[document_i] = (
                            1 if signed_scores[chosen] >= 0.0 else -1
                        )
                        objective = float(polarities[document_i]) * signed_scores
                scores = [float(value) for value in objective]
                kernel = (float(np.dot(candidate_phi[0], candidate_phi[1]))
                          if round_target == 2 else None)
            elif variant == "semfield" and selection_objective == "innovation-pl":
                # Full-span innovation carrier.  The stateful feature map has
                # already removed the complete prior sentence span.  Select
                # the largest squared keyed projection at every step; this is
                # the one-row version of the span-energy detector and does not
                # impose a sign/polarity constraint.
                signed_objective = field.score_features(candidate_phi)
                objective = np.square(signed_objective)
                largest = float(objective.max())
                best = np.flatnonzero(objective == largest)
                tied = best.size > 1
                chosen = int(np.random.default_rng(_stable_seed(
                    "exp19-tie", seed, variant, round_key,
                    int(document_ids[document_i])
                )).choice(best))
                scores = [float(value) for value in objective]
                kernel = (float(np.dot(candidate_phi[0], candidate_phi[1]))
                          if round_target == 2 else None)
            elif variant == "semfield" and selection_objective == "soft-sign":
                # One continuous, bounded score over several independent
                # keyed semantic directions. The candidate pool and sampling
                # distribution stay identical to ordinary SemField-PL.
                position = len(features[document_i])
                objective = field.score_candidates(
                    candidate_phi,
                    position,
                    score_sum=soft_sign_score_sums[document_i],
                    null_variance=soft_sign_variances[document_i],
                )
                # PL polarity locking: choose the orientation on the first
                # sentence, then keep that orientation fixed for the document.
                signed_objective = objective
                if polarities[document_i] == 0:
                    objective = np.abs(signed_objective)
                    preferred_polarities = np.where(
                        signed_objective < 0.0, -1, 1
                    ).astype(np.int8)
                else:
                    objective = float(polarities[document_i]) * signed_objective
                largest = float(objective.max())
                best = np.flatnonzero(objective == largest)
                tied = best.size > 1
                chosen = int(np.random.default_rng(_stable_seed(
                    "exp19-tie", seed, variant, round_key,
                    int(document_ids[document_i])
                )).choice(best))
                if polarities[document_i] == 0:
                    polarities[document_i] = int(
                        preferred_polarities[chosen]
                    )
                scores = [float(value) for value in objective]
                kernel = (float(np.dot(candidate_phi[0], candidate_phi[1]))
                          if round_target == 2 else None)
            elif variant == "semfield" and field_structure == "aer":
                # AER selects on the keyed continuous field for the current
                # relative-position block.  Detection later converts each
                # block score to an erasure-safe e-value.
                position = fields[document_i].position(
                    len(features[document_i]), n_sentences
                )
                signed_objective = fields[document_i].prospective_scores(
                    candidate_phi, position,
                    anchored_block_sums[document_i].get(position),
                )
                objective = signed_objective
                largest = float(objective.max())
                best = np.flatnonzero(objective == largest)
                tied = best.size > 1
                chosen = int(np.random.default_rng(_stable_seed(
                    "exp19-aer-tie", seed, variant, round_key,
                    int(document_ids[document_i])
                )).choice(best))
                scores = [float(value) for value in objective]
                kernel = (float(np.dot(candidate_phi[0], candidate_phi[1]))
                          if round_target == 2 else None)
            elif variant == "semfield" and field_structure == "anchored":
                # The anchored field keeps the original prospective PL
                # tournament, but adds a position-local Gaussian component.
                # Its denominator is the exact mixed conditional variance,
                # including the already-emitted local-field energy.
                prospective_position = local_position(len(features[document_i]))
                signed_objective = field.prospective_scores(
                    candidate_phi,
                    aggregates[document_i],
                    anchored_local_scores[document_i],
                    anchored_local_variances[document_i],
                    position=prospective_position,
                    block_sum=anchored_block_sums[document_i].get(
                        prospective_position
                    ),
                )
                if selection_objective == "protected-anchored-pl":
                    # Global PL is the protected primary objective.  The
                    # anchored field can only break ties inside a fixed
                    # prospective-z margin, so low-diversity instruct
                    # contexts fall back to ordinary PL automatically.
                    objective, _global_signed, preferred_polarities = (
                        _polarity_objective(
                            candidate_phi,
                            aggregates[document_i],
                            field.global_weights,
                            int(polarities[document_i]),
                            consensus_neighbors,
                        )
                    )
                    if polarities[document_i] == 0:
                        anchored_objective = (
                            preferred_polarities.astype(np.float64)
                            * signed_objective
                        )
                    else:
                        anchored_objective = (
                            float(polarities[document_i]) * signed_objective
                        )
                elif selection_objective in {
                    "polarity-locked", "correlation-aware-pl"
                }:
                    if polarities[document_i] == 0:
                        objective = np.abs(signed_objective)
                        preferred_polarities = np.where(
                            signed_objective < 0.0, -1, 1
                        ).astype(np.int8)
                    else:
                        objective = (
                            float(polarities[document_i]) * signed_objective
                        )
                        preferred_polarities = np.full(
                            round_target, int(polarities[document_i]),
                            dtype=np.int8,
                        )
                else:
                    objective = signed_objective
                largest = float(objective.max())
                if selection_objective == "protected-anchored-pl":
                    safe = (
                        objective
                        >= largest - float(protected_global_epsilon)
                    )
                    secondary_best = float(anchored_objective[safe].max())
                    eligible = np.flatnonzero(
                        safe & np.isclose(
                            anchored_objective, secondary_best,
                            rtol=1e-12, atol=1e-12,
                        )
                    )
                elif selection_objective == "correlation-aware-pl" and len(features[document_i]):
                    near_tie = objective >= largest - float(correlation_epsilon)
                    values = candidate_phi.astype(np.float64)
                    prospective_global = (
                        values + aggregates[document_i][None, :]
                    )
                    prospective_variance = (
                        field.rho * np.square(prospective_global).sum(axis=1)
                        + (1.0 - field.rho) * (
                            float(anchored_local_variances[document_i])
                            + np.square(values).sum(axis=1)
                        )
                    )
                    current_variance = (
                        field.rho * np.dot(
                            aggregates[document_i], aggregates[document_i]
                        )
                        + (1.0 - field.rho)
                        * float(anchored_local_variances[document_i])
                    )
                    design_effect = np.divide(
                        prospective_variance,
                        max(current_variance, 1e-15),
                    )
                    design_effect[~near_tie] = np.inf
                    best_design = float(design_effect.min())
                    eligible = np.flatnonzero(
                        np.isclose(
                            design_effect, best_design,
                            rtol=1e-12, atol=1e-12,
                        )
                    )
                else:
                    best = np.flatnonzero(objective == largest)
                    eligible = best
                tied = eligible.size > 1
                chosen = int(np.random.default_rng(_stable_seed(
                    "exp19-tie", seed, variant, field_structure, round_key,
                    int(document_ids[document_i])
                )).choice(eligible))
                if (selection_objective in {
                        "polarity-locked", "correlation-aware-pl",
                        "protected-anchored-pl",
                    } and polarities[document_i] == 0):
                    polarities[document_i] = int(preferred_polarities[chosen])
                scores = [float(value) for value in objective]
                kernel = (float(np.dot(candidate_phi[0], candidate_phi[1]))
                          if round_target == 2 else None)
            elif variant == "semfield" and selection_objective == "local":
                selection = field.select_features(
                    candidate_phi,
                    rng=np.random.default_rng(_stable_seed(
                        "exp19-tie", seed, variant, round_key,
                        int(document_ids[document_i])
                    )),
                )
                chosen = int(selection.index)
                scores = [float(value) for value in selection.candidate_scores]
                kernel = (None if selection.candidate_kernel is None else
                          float(selection.candidate_kernel))
                tied = bool(selection.tied)
            elif variant == "semfield":
                if selection_objective == "moment-hybrid":
                    z1, z2, objective = prospective_fused_scores(
                        aggregates[document_i],
                        moment_means[document_i],
                        moment_scatters[document_i],
                        len(features[document_i]),
                        candidate_phi,
                        field.weights,
                        moment_weights[document_i],
                        moment_lambda,
                    )
                    signed_objective = objective
                elif selection_objective == "decorrelated-pl":
                    signed_objective = prospective_decorrelated_scores(
                        np.asarray(features[document_i], dtype=np.float64),
                        candidate_phi,
                        field.weights,
                    )
                    if polarities[document_i] == 0:
                        objective = np.abs(signed_objective)
                        preferred_polarities = np.where(
                            signed_objective < 0.0, -1, 1
                        ).astype(np.int8)
                    else:
                        objective = (
                            float(polarities[document_i]) * signed_objective
                        )
                elif selection_objective in {
                    "self-normalized", "polarity-locked", "correlation-aware-pl",
                }:
                    prospective = (candidate_phi.astype(np.float64)
                                   + aggregates[document_i])
                    norms = np.linalg.norm(prospective, axis=1)
                    signed_objective = np.divide(
                        prospective @ field.weights, norms,
                        out=np.full(round_target, -np.inf, dtype=np.float64),
                        where=norms > 1e-15,
                    )
                    if selection_objective in {
                        "polarity-locked", "correlation-aware-pl",
                    }:
                        objective, signed_objective, preferred_polarities = (
                            _polarity_objective(
                                candidate_phi,
                                aggregates[document_i],
                                field.weights,
                                int(polarities[document_i]),
                                consensus_neighbors,
                            )
                        )
                    else:
                        objective = signed_objective
                else:
                    raise ValueError(f"unknown selection objective {selection_objective!r}")
                if selection_objective == "correlation-aware-pl":
                    # Keep PL's keyed score as the primary objective.  Only
                    # near-ties may be broken by prospective design effect.
                    # The first sentence must be selected by PL because no
                    # previous feature exists to define correlation.
                    largest = float(objective.max())
                    near_tie = objective >= largest - float(correlation_epsilon)
                    if len(features[document_i]) == 0:
                        eligible = np.flatnonzero(objective == largest)
                    else:
                        values = candidate_phi.astype(np.float64)
                        prospective_energy = np.square(
                            values + aggregates[document_i][None, :]
                        ).sum(axis=1)
                        candidate_energy = np.square(values).sum(axis=1)
                        denominator = (
                            float(feature_energies[document_i]) + candidate_energy
                        )
                        design_effect = np.divide(
                            prospective_energy,
                            denominator,
                            out=np.full(round_target, np.inf, dtype=np.float64),
                            where=denominator > 1e-15,
                        )
                        design_effect[~near_tie] = np.inf
                        best_design = float(design_effect.min())
                        eligible = np.flatnonzero(
                            np.isclose(
                                design_effect, best_design,
                                rtol=1e-12, atol=1e-12,
                            )
                        )
                    tied = eligible.size > 1
                    chosen = int(np.random.default_rng(_stable_seed(
                        "exp19-tie", seed, variant, round_key,
                        int(document_ids[document_i])
                    )).choice(eligible))
                else:
                    largest = float(objective.max())
                    best = np.flatnonzero(objective == largest)
                    tied = best.size > 1
                    chosen = int(np.random.default_rng(_stable_seed(
                        "exp19-tie", seed, variant, round_key,
                        int(document_ids[document_i])
                    )).choice(best))
                if (selection_objective in {
                        "polarity-locked", "correlation-aware-pl",
                        "decorrelated-pl",
                    } and polarities[document_i] == 0):
                    polarities[document_i] = int(preferred_polarities[chosen])
                scores = [float(value) for value in objective]
                kernel = (float(np.dot(candidate_phi[0], candidate_phi[1]))
                          if round_target == 2 else None)
            else:
                chosen = 0
                scores = [float(
                    field.score_candidates(
                        candidate_phi,
                        len(features[document_i]),
                        score_sum=soft_sign_score_sums[document_i],
                        null_variance=soft_sign_variances[document_i],
                    )[0]
                    if selection_objective == "soft-sign"
                    else field.score_features(candidate_phi)[0]
                )]
                kernel = None
                tied = False

            predicted = (
                math.sqrt(max(0.0, 1.0 - kernel) / math.pi)
                if kernel is not None else None
            )
            chosen_sentence = group[chosen]
            chosen_feature = candidate_phi[chosen].copy()
            if selection_objective == "soft-sign":
                contribution = field.contributions(
                    chosen_feature[None, :], len(features[document_i])
                )[0]
                soft_sign_score_sums[document_i] += float(
                    contribution.sum() / field.channels
                )
                soft_sign_variances[document_i] += float(
                    np.square(contribution).sum() / (field.channels ** 2)
                )
            if getattr(feature_map, "stateful", False) or selection_objective == "relational-continuous":
                raw_embeddings[document_i].append(
                    np.asarray(vectors[start + chosen], dtype=np.float64).copy()
                )
            aggregates[document_i] += chosen_feature.astype(np.float64)
            if field_structure in {"anchored", "aer"}:
                chosen_position = local_position(len(features[document_i]))
                chosen_value = chosen_feature.astype(np.float64)
                anchored_local_scores[document_i] += float(
                    np.dot(field.local_weights(chosen_position), chosen_value)
                )
                accumulate_local(document_i, chosen_position, chosen_value)
            feature_energies[document_i] += float(
                np.dot(chosen_feature.astype(np.float64),
                       chosen_feature.astype(np.float64))
            )
            if selection_objective == "moment-hybrid":
                count = len(features[document_i])
                if count == 0:
                    moment_means[document_i] = chosen_feature.astype(np.float64)
                    moment_scatters[document_i] = np.zeros_like(
                        moment_means[document_i]
                    )
                else:
                    delta = chosen_feature.astype(np.float64) - moment_means[document_i]
                    moment_scatters[document_i] += (
                        float(count) / float(count + 1) * np.square(delta)
                    )
                    moment_means[document_i] += delta / float(count + 1)
            sentences[document_i].append(chosen_sentence)
            features[document_i].append(chosen_feature)
            selected_scores[document_i].append(scores[chosen])
            candidate_scores[document_i].append(scores)
            candidate_kernels[document_i].append(kernel)
            predicted_lifts[document_i].append(predicted)
            selected_indices[document_i].append(chosen)
            ties[document_i].append(tied)
            # A chat-templated prompt ends with a newline ("...assistant\n").
            # Joining with a space there would put the first sentence one
            # space into the assistant turn; every later append still needs
            # the space.
            contexts[document_i] += (
                chosen_sentence
                if contexts[document_i].endswith(("\n", " "))
                else " " + chosen_sentence
            )
            total_usable += round_target
            if len(sentences[document_i]) < n_sentences:
                still_needs_more.append(document_i)

        return still_needs_more

    # Main pass: identical to the original single-pass loop, so a document
    # that never fails is generated with exactly the seeds it always was.
    # Skipped entirely when resuming -- that state already reflects however
    # far the main pass got for these documents in the run being recovered.
    if resume is None:
        active = list(range(n_documents))
        for step in range(n_sentences):
            if not active:
                break
            active = run_round(active, step)
            print(
                f"    {variant}: sentence {step + 1}/{n_sentences}, "
                f"{len(active)}/{n_documents} documents active",
                flush=True,
            )

    # Catch-up pass: documents that fell short of n_sentences get further
    # attempts here, using their own seed namespace (the "catchup" marker),
    # so this can never collide with or change a seed used above. A document
    # that never needed this pass is completely unaffected by it.
    incomplete = [i for i in range(n_documents) if len(sentences[i]) < n_sentences]
    if incomplete:
        print(
            f"    {variant}: {len(incomplete)}/{n_documents} documents fell short "
            f"of {n_sentences} sentences after the main pass; retrying them",
            flush=True,
        )
        catchup_round = 0
        while incomplete and catchup_round < max_retry_rounds:
            incomplete = run_round(incomplete, ("catchup", catchup_round))
            catchup_round += 1
            print(
                f"    {variant}: catch-up round {catchup_round}, "
                f"{len(incomplete)}/{n_documents} documents still short",
                flush=True,
            )
        for document_i in incomplete:
            have = len(sentences[document_i])
            print(
                f"WARNING: {variant} document {int(document_ids[document_i])} only "
                f"reached {have}/{n_sentences} sentences after the catch-up pass.",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    documents: list[dict] = []
    feature_matrices: list[np.ndarray] = []
    design_effects: list[float] = []
    all_kernels: list[float] = []
    all_lifts: list[float] = []
    all_gaps: list[float] = []
    all_selected_scores: list[float] = []
    all_objective_ranges: list[float] = []
    all_objective_stds: list[float] = []
    all_objective_unique_counts: list[float] = []
    all_objective_top_gaps: list[float] = []
    all_candidate_mean_pair_cosines: list[float | None] = []
    for document_i in range(n_documents):
        matrix = (
            np.asarray(features[document_i], dtype=np.float32)
            if features[document_i]
            else np.zeros((0, feature_map.n_features), dtype=np.float32)
        )
        aggregation = (
            "sentence-decorrelated"
            if selection_objective == "decorrelated-pl" else "uniform"
        )
        statistic = None if selection_objective == "relational-continuous" else (
            innovation_span_statistic(
                matrix, fields[document_i].weights
            )
            if selection_objective == "innovation-pl" else
            fields[document_i].detect_features(
                matrix,
                two_sided=(selection_objective == "soft-sign"),
            )
            if selection_objective == "soft-sign" else
            fields[document_i].detect_features(matrix)
            if field_structure == "aer" else
            protected_anchored_field_statistic(
                matrix,
                fields[document_i].key,
                rho=fields[document_i].rho,
                channel=fields[document_i].channel,
                positions=block_positions(
                    matrix.shape[0], fields[document_i].blocks
                ),
            )
            if (field_structure == "anchored"
                and selection_objective == "protected-anchored-pl") else
            fields[document_i].detect_features(
                matrix,
                two_sided=(selection_objective in {
                    "polarity-locked", "correlation-aware-pl", "soft-sign"
                }),
            )
            if field_structure == "anchored" else
            moment_field_statistic(
                matrix, fields[document_i].weights,
                moment_weights[document_i], moment_lambda=moment_lambda
            )
            if selection_objective == "moment-hybrid" else
            decorrelated_field_statistic(
                matrix, fields[document_i].weights, two_sided=True
            )
            if selection_objective == "decorrelated-pl" else
            two_sided_field_statistic(
                matrix, fields[document_i].weights, aggregation=aggregation
            )
            if selection_objective in {
                "polarity-locked", "correlation-aware-pl"
            }
            else field_statistic(
                matrix, fields[document_i].weights, aggregation=aggregation
            )
        )
        design_effect = getattr(statistic, "design_effect", None)
        if design_effect is not None:
            design_effects.append(float(design_effect))
        all_kernels.extend(
            float(value) for value in candidate_kernels[document_i]
            if value is not None
        )
        all_lifts.extend(
            float(value) for value in predicted_lifts[document_i]
            if value is not None
        )
        all_gaps.extend(
            abs(float(row[0]) - float(row[1]))
            for row in candidate_scores[document_i] if len(row) == 2
        )
        all_selected_scores.extend(selected_scores[document_i])
        all_candidate_mean_pair_cosines.extend(
            candidate_mean_pair_cosines[document_i]
        )
        for row in candidate_scores[document_i]:
            values = np.asarray(row, dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            all_objective_ranges.append(float(np.ptp(values)))
            all_objective_stds.append(float(np.std(values)))
            all_objective_unique_counts.append(float(
                np.unique(np.round(values, decimals=10)).size
            ))
            if values.size > 1:
                ordered = np.partition(values, values.size - 2)
                all_objective_top_gaps.append(float(
                    ordered[-1] - ordered[-2]
                ))
        moment_record = None
        if selection_objective == "moment-hybrid" and matrix.shape[0]:
            z_semantic, z_moment, z_fused = fused_components(
                matrix, fields[document_i].weights,
                moment_weights[document_i], moment_lambda
            )
            moment_record = {
                "z_semantic": float(z_semantic),
                "z_moment": float(z_moment),
                "z_fused": float(z_fused),
                "lambda": float(moment_lambda),
            }
        detector_record = (
            {
                "carrier": "relational-continuous",
                "score": float((
                    (int(polarities[document_i]) or 1)
                    * relational_document_score(
                        raw_embeddings[document_i], key,
                        relational_harmonics,
                    )
                )),
                "n_pairs": int(max(0, len(raw_embeddings[document_i]) *
                                    (len(raw_embeddings[document_i]) - 1) // 2)),
            }
            if selection_objective == "relational-continuous"
            else asdict(statistic)
        )
        document_record = {
            "document_id": document_i,
            "text": " ".join(sentences[document_i]),
            "sentences": sentences[document_i],
            "sentences_emitted": len(sentences[document_i]),
            "selected_scores": selected_scores[document_i],
            "candidate_scores": candidate_scores[document_i],
            "candidate_kernels": candidate_kernels[document_i],
            "candidate_mean_pair_cosines": (
                candidate_mean_pair_cosines[document_i]
            ),
            "predicted_one_step_lifts": predicted_lifts[document_i],
            "selected_indices": selected_indices[document_i],
            "ties": ties[document_i],
            "raw_embeddings": (
                [np.asarray(row, dtype=np.float32).tolist()
                 for row in raw_embeddings[document_i]]
                if selection_objective == "relational-continuous" else None
            ),
            "polarity": (
                int(polarities[document_i])
                if selection_objective in {
                    "polarity-locked", "correlation-aware-pl",
                    "protected-anchored-pl", "soft-sign", "decorrelated-pl",
                    "relational-continuous",
                } else None
            ),
            "moment": moment_record,
            "detector": detector_record,
        }
        if capture_candidate_texts:
            document_record["candidate_texts"] = candidate_texts[document_i]
        documents.append(document_record)
        feature_matrices.append(matrix)

    total_emitted = sum(row["sentences_emitted"] for row in documents)
    full_documents = sum(
        row["sentences_emitted"] == n_sentences for row in documents
    )
    summary = {
        "scheme": (
            "semfield-relational-continuous"
            if selection_objective == "relational-continuous"
            else
            "semfield-innovation-pl"
            if selection_objective == "innovation-pl"
            else
            "semfield-soft-sign"
            if selection_objective == "soft-sign"
            else "semfield-decorrelated-pl"
            if selection_objective == "decorrelated-pl"
            else "semfield-moment"
            if selection_objective == "moment-hybrid"
            else "semfield-protected-anchored-pl"
            if selection_objective == "protected-anchored-pl"
            else "semfield-anchored"
            if field_structure in {"anchored", "aer"}
            else "semfield-" + str(
                {
                    "linear-spherical-v1": "linear",
                    "residual-linear-spherical-v1": "residual",
                    "hybrid-residual-linear-spherical-v1": "hybrid-residual",
                    "span-residual-linear-spherical-v1": "span-residual",
                }.get(
                    feature_map.metadata().get("type"),
                    feature_map.metadata().get("kernel", "linear"),
                )
            )
        ) if variant == "semfield" else "none",
        "variant": variant,
        "feature_map": feature_map.metadata(),
        "selection_objective": selection_objective,
        "field_structure": field_structure,
        "field_rho": float(field_rho),
        "field_blocks": (None if field_blocks is None else int(field_blocks)),
        "semfield_candidates": int(semfield_candidates),
        "candidate_schedule": (list(int(v) for v in candidate_schedule)
                               if candidate_schedule is not None else None),
        "consensus_neighbors": int(consensus_neighbors),
        "moment_lambda": float(moment_lambda),
        "correlation_epsilon": float(correlation_epsilon),
        "protected_global_epsilon": float(protected_global_epsilon),
        "soft_sign_channels": int(soft_sign_channels),
        "soft_sign_beta": float(soft_sign_beta),
        "relational_harmonics": int(relational_harmonics),
        "proposal": {
            "mode": proposal_mode if variant == "semfield" else "iid",
            "branches": (
                int(proposal_branches)
                if variant == "semfield" and proposal_mode in {
                    "branch-stratified", "entropy-stratified"
                }
                else None
            ),
            "prefix_tokens": (
                int(proposal_prefix_tokens)
                if variant == "semfield" and proposal_mode in {
                    "branch-stratified", "entropy-stratified"
                }
                else None
            ),
            "pool_size": (
                int(proposal_pool_size)
                if variant == "semfield" and proposal_mode == "hot-first-token"
                else None
            ),
            "opening_temperature": (
                float(proposal_temperature)
                if variant == "semfield" and proposal_mode == "hot-first-token"
                else None
            ),
            "opening_top_k": (
                int(proposal_top_k)
                if variant == "semfield" and proposal_mode == "hot-first-token"
                else None
            ),
            "opening_base_logprob_gap": (
                float(proposal_first_token_logprob_gap)
                if variant == "semfield" and proposal_mode == "hot-first-token"
                else None
            ),
            "likelihood_filter": (
                "top-length-normalized-processed-logprob-v1"
                if variant == "semfield" and proposal_mode == "hot-first-token"
                else None
            ),
        },
        "documents": n_documents,
        "documents_complete": int(full_documents),
        "document_completion_rate": float(full_documents / max(1, n_documents)),
        "sentences_target": int(n_documents * n_sentences),
        "sentences_emitted": int(total_emitted),
        "sentence_target_completion_rate": float(
            total_emitted / max(1, n_documents * n_sentences)
        ),
        "requested_candidates": int(total_requested),
        "completed_candidates": int(total_completed),
        "usable_candidates": int(total_usable),
        "discarded_completed_candidates": int(max(0, total_completed - total_usable)),
        "requested_candidates_per_sentence": float(
            total_requested / max(1, total_emitted)
        ),
        "completed_candidates_per_sentence": float(
            total_completed / max(1, total_emitted)
        ),
        "completion_rate": float(total_completed / max(1, total_requested)),
        "retry_context_rounds": int(retry_context_rounds),
        "candidate_kernel": _distribution(all_kernels),
        "candidate_score_gap": _distribution(all_gaps),
        "candidate_objective_range": _distribution(all_objective_ranges),
        "candidate_objective_std": _distribution(all_objective_stds),
        "candidate_objective_unique_count": _distribution(
            all_objective_unique_counts
        ),
        "candidate_objective_top_gap": _distribution(
            all_objective_top_gaps
        ),
        "candidate_mean_pair_cosine": _distribution(
            all_candidate_mean_pair_cosines
        ),
        "selected_field_score": _distribution(all_selected_scores),
        "predicted_one_step_lift": _distribution(all_lifts),
        "predicted_one_step_lift_scope": (
            "sqrt((1-kappa)/pi) is exact for a fixed candidate pair and an "
            "unconditioned Gaussian field; field reuse makes later contexts "
            "adaptive, so this is a benchmark diagnostic, not a sequence-level "
            "power proof"
        ),
        "field_design_effect": _distribution(design_effects),
        "seconds_total": float(elapsed),
        "seconds_sampling": float(sampling_seconds),
        "seconds_embedding": float(embedding_seconds),
        "seconds_feature_transform": float(feature_seconds),
        "seconds_per_document": float(elapsed / max(1, n_documents)),
    }
    return documents, feature_matrices, summary


def _parse_boolean(value: str) -> bool:
    value = value.lower()
    if value not in {"true", "false"}:
        raise argparse.ArgumentTypeError("expected true or false")
    return value == "true"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--corpus-dir", default="data/c4-realnewslike")
    parser.add_argument("--input", help="shared JSON manifest with document_id/prompt/reference")
    parser.add_argument("--generator", default="opt-1.3b")
    parser.add_argument("--vllm-python",
                        default=sys.executable)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=192)
    parser.add_argument("--vllm-request-size", type=int, default=512)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float,
                        default=0.90)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--embedder", default="mpnet")
    parser.add_argument(
        "--watermark", type=_parse_boolean, default=True, metavar="{true,false}",
        help="generate watermarked text (default: true); false ignores watermark-specific options",
    )
    parser.add_argument("--documents", type=int, default=100)
    parser.add_argument("--document-ids", nargs="+", type=int,
                        help="optional original manifest document IDs")
    parser.add_argument("--prompt-sentences", type=int, default=3)
    parser.add_argument("--sentences", type=int, default=40)
    parser.add_argument("--feature-dim", type=int, default=2048)
    parser.add_argument("--kernel", choices=["linear"], default="linear")
    parser.add_argument(
        "--feature-map",
        choices=[
            "linear", "content-routed", "residual", "hybrid-residual",
            "span-residual", "whitened", "projected", "poly2", "poly3", "rbf",
            "cone-equalized", "mean-axis-shrunk",
        ],
        default="linear",
        help=(
            "sentence feature map: linear is the existing LF/PL map; poly2/poly3 "
            "use fixed-width random Maclaurin features; rbf uses fixed-width "
            "random Fourier features; "
            "content-routed uses a public frozen semantic router to reduce "
            "cross-sentence field correlation while keeping a continuous "
            "Gaussian field; "
            "residual removes the running topic direction; hybrid-residual "
            "keeps equal absolute and residual channels; whitened applies the "
            "public whitening matrix so the keyed direction sees the "
            "directions candidates actually differ in; projected applies a "
            "frozen clean-only candidate-PCA basis and reduces the keyed "
            "feature dimension"
        ),
    )
    parser.add_argument(
        "--whitener-path", default=None,
        help=("public whitening matrix for --feature-map whitened, built by "
              "build_whitening_matrix.py from unwatermarked text only"),
    )
    parser.add_argument(
        "--cone-equalizer-path", default=None,
        help=("public frozen one-axis cone equalizer NPZ for "
              "--feature-map cone-equalized; fitted independently from "
              "unwatermarked text"),
    )
    parser.add_argument(
        "--mean-axis-shrink-path", default=None,
        help=("public frozen mean-axis shrink transform for "
              "--feature-map mean-axis-shrunk; fitted from independent "
              "unwatermarked text"),
    )
    parser.add_argument(
        "--router-path", default=None,
        help=("public, frozen content-router NPZ for --feature-map "
              "content-routed; it is built from an external generic corpus "
              "and is never fitted on the evaluation documents or attacks"),
    )
    parser.add_argument(
        "--projection-path", default=None,
        help=("public frozen candidate-PCA NPZ for --feature-map projected; "
              "the selected array must have shape (embedding_dim, k)"),
    )
    parser.add_argument(
        "--projection-key", default="basis_k64",
        help="array name inside --projection-path (default: basis_k64)",
    )
    parser.add_argument(
        "--local-blocks", dest="field_blocks", type=int, default=None,
        help=(
            "index the local field by relative-position block instead of "
            "absolute sentence index. Rewrites change sentence counts "
            "(dipper 12 -> 13.07, llm-rewrite 12 -> 11.50) which shifts every "
            "later absolute index onto the wrong weight, but they preserve "
            "order, so a relative block survives. Omit for absolute indexing."
        ),
    )
    parser.add_argument(
        "--residual-weight", type=float, default=0.5,
        help="weight of the residual channel in hybrid-residual",
    )
    parser.add_argument("--length-scale", type=float, default=0.5)
    parser.add_argument("--kernel-alpha", type=float, default=1.0,
                        help="shape parameter for rational-quadratic")
    parser.add_argument("--feature-seed", type=int, default=0)
    parser.add_argument("--field-channel", type=int, default=0)
    parser.add_argument(
        "--field-structure", choices=["global", "anchored", "aer"], default="global",
        help=(
            "global is the existing SemField field; anchored mixes it with "
            "independent continuous position-local fields"
        ),
    )
    parser.add_argument(
        "--field-rho", type=float, default=1.0,
        help=(
            "global-field weight for anchored structure in [0,1]; 1.0 "
            "recovers the existing global SemField"
        ),
    )
    parser.add_argument("--key-seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--max-context-tokens", type=int, default=512)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-retry-rounds", type=int, default=5)
    parser.add_argument("--candidates", dest="semfield_candidates", type=int, default=2)
    parser.add_argument(
        "--candidate-pools-output", type=str, default=None,
        help=(
            "write every fresh SemField candidate text to a separate JSON "
            "sidecar; intended for offline controllability audits"
        ),
    )
    parser.add_argument(
        "--trajectory-paths", type=int, default=1,
        help=(
            "number of complete SemField document paths to keep alive; "
            "values greater than one enable Trajectory-PL"
        ),
    )
    parser.add_argument(
        "--trajectory-candidates", type=int, default=8,
        help=(
            "ordinary complete-sentence candidates sampled per trajectory "
            "at each round (B*R is the matched rollout budget)"
        ),
    )
    parser.add_argument(
        "--candidate-schedule", default=None,
        help="comma-separated per-sentence candidate counts, e.g. "
             "64,64,64,32,32,32,16,16,16,8,8,8. Must have exactly "
             "--sentences entries. Omit for a flat --candidates "
             "budget at every step (the default, unchanged).")
    parser.add_argument(
        "--consensus-neighbors", type=int, default=1,
        help="nearest candidates used by polarity-locked consensus selection; "
             "one reproduces ordinary polarity locking exactly")
    parser.add_argument(
        "--polarity-locked", type=_parse_boolean, default=False,
        metavar="{true,false}",
        help="false selects SemField; true selects SemField-PL (default: false)",
    )
    parser.add_argument(
        "--soft-sign-channels", type=int, default=8,
        help="number of independent keyed directions for soft-sign selection",
    )
    parser.add_argument(
        "--soft-sign-beta", type=float, default=4.0,
        help="fixed tanh sharpness for soft-sign selection",
    )
    parser.add_argument(
        "--relational-harmonics", type=int, default=4,
        help="fixed low-frequency components in the SemField-RC carrier",
    )
    parser.add_argument(
        "--moment-lambda", type=float, default=0.5,
        help="weight of the centered-scatter channel for moment-hybrid",
    )
    parser.add_argument(
        "--correlation-epsilon", type=float, default=0.05,
        help=("PL score margin allowed when minimizing prospective design "
              "effect for correlation-aware-pl"),
    )
    parser.add_argument(
        "--protected-global-epsilon", type=float, default=0.10,
        help=("maximum prospective global PL z-score sacrificed before the "
              "anchored score may break the tie"),
    )
    parser.add_argument(
        "--proposal-mode",
        choices=[
            "iid", "arithmetic", "branch-stratified", "entropy-stratified",
            "hot-first-token",
        ],
        default="iid",
        help="key-independent candidate proposal mechanism",
    )
    parser.add_argument(
        "--proposal-branches", type=int, default=8,
        help="distinct prefix branches in branch-stratified mode",
    )
    parser.add_argument(
        "--proposal-prefix-tokens", type=int, default=8,
        help="token length of each branch prefix",
    )
    parser.add_argument(
        "--proposal-pool-size", type=int, default=32,
        help="raw opening-branch pool before hot-first-token likelihood filtering",
    )
    parser.add_argument(
        "--proposal-temperature", type=float, default=1.3,
        help="temperature used only to choose hot-first-token branches",
    )
    parser.add_argument(
        "--proposal-top-k", type=int, default=50,
        help="hard cap on opening tokens in hot-first-token mode",
    )
    parser.add_argument(
        "--proposal-first-token-logprob-gap", type=float, default=5.0,
        help="maximum base-temperature log-probability gap for an opening token",
    )
    parser.add_argument("--output-dim", type=int, default=None)
    parser.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"],
                        default="auto")
    parser.add_argument("--cache-dir", default=str(EMBEDDING_CACHE))
    parser.add_argument("--embed-batch-size", type=int, default=None)
    parser.add_argument("--max-embed-length", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampling-seed-offset", type=int, default=0)
    parser.add_argument("--out-dir", default="generations_semfield")
    args = parser.parse_args(argv)
    args.variants = ["semfield" if args.watermark else "none"]
    if not args.watermark:
        # Keep ordinary sampling settings, but do not let watermark options
        # affect validation, proposal sampling, or auxiliary scoring.
        watermark_options = {
            "embedder", "feature_dim", "kernel", "feature_map",
            "whitener_path", "cone_equalizer_path", "mean_axis_shrink_path",
            "router_path", "projection_path", "projection_key", "field_blocks",
            "residual_weight", "length_scale", "kernel_alpha", "feature_seed",
            "field_channel", "field_structure", "field_rho", "key_seed",
            "semfield_candidates", "candidate_pools_output", "candidate_schedule",
            "consensus_neighbors", "polarity_locked", "moment_lambda",
            "correlation_epsilon", "protected_global_epsilon", "output_dim",
            "embed_batch_size", "max_embed_length", "cache_dir",
        }
        for action in parser._actions:
            if (action.dest in watermark_options
                    or action.dest.startswith(("trajectory_", "proposal_", "soft_sign_", "relational_"))):
                setattr(args, action.dest, action.default)
    # Preserve the existing selectors and artifact schema; only the CLI changes.
    args.selection_objective = "polarity-locked" if args.polarity_locked else "self-normalized"
    args.candidates = args.semfield_candidates
    return args


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.documents <= 0 or args.prompt_sentences <= 0 or args.sentences <= 0:
        parser.error("document and sentence counts must be positive")
    if args.feature_dim <= 0:
        parser.error("--feature-dim must be positive")
    if (
        args.feature_map == "hybrid-residual"
        and (not math.isfinite(args.residual_weight)
             or not 0.0 < args.residual_weight < 1.0)
    ):
        parser.error("--residual-weight must lie strictly between zero and one")
    if args.feature_map == "content-routed" and not args.router_path:
        parser.error("--feature-map content-routed requires --router-path")
    if args.feature_map == "mean-axis-shrunk" and not args.mean_axis_shrink_path:
        parser.error("--feature-map mean-axis-shrunk requires --mean-axis-shrink-path")
    if not math.isfinite(args.length_scale) or args.length_scale <= 0.0:
        parser.error("--length-scale must be finite and positive")
    if not math.isfinite(args.kernel_alpha) or args.kernel_alpha <= 0.0:
        parser.error("--kernel-alpha must be finite and positive")
    if args.field_channel < 0:
        parser.error("--field-channel must be non-negative")
    if (not math.isfinite(args.field_rho)
            or not 0.0 <= args.field_rho <= 1.0):
        parser.error("--field-rho must lie in [0, 1]")
    if args.field_structure == "anchored" and args.selection_objective == "moment-hybrid":
        parser.error("moment-hybrid selection is not supported with anchored fields")
    if not math.isfinite(args.temperature) or args.temperature <= 0.0:
        parser.error("--temperature must be finite and positive")
    if not math.isfinite(args.top_p) or not 0.0 < args.top_p <= 1.0:
        parser.error("--top-p must lie in (0, 1]")
    if not math.isfinite(args.repetition_penalty) or args.repetition_penalty <= 0.0:
        parser.error("--repetition-penalty must be finite and positive")
    if args.max_new_tokens <= 0 or args.max_context_tokens <= 0:
        parser.error("token budgets must be positive")
    if args.max_rows is not None and args.max_rows <= 0:
        parser.error("--max-rows must be positive")
    if args.max_retry_rounds <= 0:
        parser.error("--max-retry-rounds must be positive")
    if args.semfield_candidates < 2:
        parser.error("--candidates must be at least two")
    if args.trajectory_paths < 1:
        parser.error("--trajectory-paths must be positive")
    if args.trajectory_candidates < 2:
        parser.error("--trajectory-candidates must be at least two")
    if args.trajectory_paths > 1:
        if args.field_structure != "global":
            parser.error("Trajectory-PL requires --field-structure global")
        if args.selection_objective != "polarity-locked":
            parser.error(
                "Trajectory-PL requires --polarity-locked true"
            )
        if args.proposal_mode != "iid":
            parser.error("Trajectory-PL requires ordinary iid proposals")
        if args.candidate_pools_output is not None:
            parser.error(
                "--candidate-pools-output is not supported with trajectory PL"
            )
    if args.consensus_neighbors <= 0:
        parser.error("--consensus-neighbors must be positive")
    if not math.isfinite(args.moment_lambda) or args.moment_lambda < 0.0:
        parser.error("--moment-lambda must be finite and non-negative")
    if not math.isfinite(args.correlation_epsilon) or args.correlation_epsilon < 0.0:
        parser.error("--correlation-epsilon must be finite and non-negative")
    if (not math.isfinite(args.protected_global_epsilon)
            or args.protected_global_epsilon < 0.0):
        parser.error(
            "--protected-global-epsilon must be finite and non-negative"
        )
    if args.soft_sign_channels <= 0:
        parser.error("--soft-sign-channels must be positive")
    if not math.isfinite(args.soft_sign_beta) or args.soft_sign_beta <= 0.0:
        parser.error("--soft-sign-beta must be finite and positive")
    if (args.selection_objective == "soft-sign"
            and args.field_structure != "global"):
        parser.error(
            "soft-sign selection requires --field-structure global"
        )
    if (args.selection_objective == "decorrelated-pl"
            and args.field_structure != "global"):
        parser.error(
            "decorrelated-pl selection requires "
            "--field-structure global"
        )
    if (args.selection_objective == "protected-anchored-pl"
            and args.field_structure != "anchored"):
        parser.error(
            "protected-anchored-pl selection requires "
            "--field-structure anchored"
        )
    if args.proposal_branches < 2:
        parser.error("--proposal-branches must be at least two")
    if not 1 <= args.proposal_prefix_tokens < args.max_new_tokens:
        parser.error(
            "--proposal-prefix-tokens must lie in [1, --max-new-tokens)"
        )
    schedule = None
    if args.candidate_schedule:
        try:
            schedule = _parse_candidate_schedule(args.candidate_schedule)
        except ValueError:
            parser.error("--candidate-schedule must be comma-separated integers")
        if len(schedule) != args.sentences:
            parser.error(
                f"--candidate-schedule has {len(schedule)} entries but "
                f"--sentences is {args.sentences}; they must match"
            )
        if min(schedule) < 2:
            parser.error("every --candidate-schedule entry must be at least two")
        if args.consensus_neighbors > min(schedule):
            parser.error("--consensus-neighbors cannot exceed the smallest "
                         "candidate-schedule entry")
    elif args.consensus_neighbors > args.semfield_candidates:
        parser.error("--consensus-neighbors cannot exceed --candidates")
    proposal_counts = schedule or [args.semfield_candidates]
    if args.proposal_mode == "hot-first-token":
        if args.proposal_pool_size < max(proposal_counts):
            parser.error("--proposal-pool-size must cover every candidate count")
        if args.proposal_top_k < args.proposal_pool_size:
            parser.error("--proposal-top-k must cover --proposal-pool-size")
        if (not math.isfinite(args.proposal_temperature)
                or args.proposal_temperature <= 0.0):
            parser.error("--proposal-temperature must be finite and positive")
        if (not math.isfinite(args.proposal_first_token_logprob_gap)
                or args.proposal_first_token_logprob_gap < 0.0):
            parser.error(
                "--proposal-first-token-logprob-gap must be finite and non-negative"
            )
    if args.proposal_mode == "branch-stratified":
        if any(value % args.proposal_branches for value in proposal_counts):
            parser.error(
                "every candidate count must be divisible by "
                "--proposal-branches in branch-stratified mode"
            )
    if args.output_dim is not None and args.output_dim <= 0:
        parser.error("--output-dim must be positive")
    if args.embed_batch_size is not None and args.embed_batch_size <= 0:
        parser.error("--embed-batch-size must be positive")
    if args.max_embed_length <= 0:
        parser.error("--max-embed-length must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    # Keep validation errors attached to the real parser/help text.
    args = parse_args(argv)
    real_parser = argparse.ArgumentParser(description=__doc__)
    _validate_args(args, real_parser)

    from semfield.embedders import (  # imported lazily after CLI validation
        EMBEDDERS,
        embed,
        load_embedder,
        resolve_devices,
    )

    if args.input:
        manifest_path = Path(args.input).expanduser().resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        available = manifest.get("documents", [])
        if args.document_ids:
            requested = list(dict.fromkeys(int(value) for value in args.document_ids))
            by_id = {int(row.get("document_id", index)): row
                     for index, row in enumerate(available)}
            missing = [value for value in requested if value not in by_id]
            if missing:
                raise ValueError(f"shared input is missing document IDs {missing}")
            selected = [by_id[value] for value in requested]
        else:
            selected = available[:args.documents]
        if not selected or any(
            not isinstance(row, dict) or not str(row.get("prompt", "")).strip()
            for row in selected
        ):
            raise ValueError("shared input needs non-empty document prompts")
        ids = [int(row.get("document_id", index))
               for index, row in enumerate(selected)]
        if not args.document_ids and ids != list(range(len(ids))):
            raise ValueError("shared input document IDs must be contiguous from zero")
        # Leading whitespace and trailing spaces/tabs are accidental and safe to
        # drop, but a trailing NEWLINE is load-bearing: a chat-templated prompt
        # ends with "<|im_start|>assistant\n", and stripping that newline leaves
        # the model generating from a position the template never produces. On
        # the Qwen3-instruct/Dolly run a plain .strip() damaged all 100 prompts
        # and only 45% of first sentences came back starting with a capital
        # letter -- the rest were fragments, one of them beginning with a stray
        # period. Preserve the manifest's prompt as written.
        prompts = [str(row["prompt"]).lstrip().rstrip(" \t") for row in selected]
        references = [str(row.get("reference", "")).strip() for row in selected]
        source = {
            "type": "shared-manifest",
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
            "documents": len(selected),
        }
    else:
        prompts, references, source = build_prompts(
            args.corpus_dir, args.documents, args.prompt_sentences
        )
    devices = resolve_devices(args.device)
    # A ``+``-separated embedder name enables a multi-view pilot, e.g.
    # ``mpnet+gte-base``.  Each view remains in its own coordinate block; we
    # only concatenate normalized blocks (scaled by 1/sqrt(M)).  This is not a
    # learned transform and keeps the generated text/sampling protocol fixed.
    embedder_names = [part.strip() for part in str(args.embedder).split("+")
                      if part.strip()]
    if not embedder_names:
        raise ValueError("--embedder must name at least one encoder")
    encoders = [
        load_embedder(
            name,
            output_dim=args.output_dim,
            device=devices[0],
            max_seq_length=args.max_embed_length,
            precision=args.precision,
        )
        for name in embedder_names
    ]
    # Keep the first encoder for the legacy metadata fields; the complete list
    # is recorded below so a detector can reconstruct the input dimension.
    encoder = encoders[0]

    def embedder(items: Sequence[str]) -> np.ndarray:
        blocks = []
        for name, view_encoder in zip(embedder_names, encoders):
            block = embed(
                view_encoder,
                list(items),
                batch_size=args.embed_batch_size,
                normalize=True,
                devices=devices,
                model_name=name,
                cache_dir=args.cache_dir or None,
                output_dim=args.output_dim,
                precision=args.precision,
                show_progress_bar=False,
            )
            # Re-normalize after cache/load so every view contributes equally.
            block = np.asarray(block, dtype=np.float32)
            block /= np.maximum(np.linalg.norm(block, axis=1, keepdims=True),
                                1e-12)
            blocks.append(block / math.sqrt(len(embedder_names)))
        return np.concatenate(blocks, axis=1)

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
    probe = embedder(["A sentence used only to determine embedding width."])
    feature_map = build_feature_map(
        args.kernel,
        input_dim=int(probe.shape[1]),
        feature_map=args.feature_map,
        residual_weight=args.residual_weight,
        whitener_path=args.whitener_path,
        cone_equalizer_path=args.cone_equalizer_path,
        mean_axis_shrink_path=args.mean_axis_shrink_path,
        projection_path=args.projection_path,
        projection_key=args.projection_key,
        router_path=args.router_path,
        n_features=args.feature_dim,
        length_scale=args.length_scale,
        alpha=args.kernel_alpha,
        seed=args.feature_seed,
    )
    key = experiment_key(args.key_seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder_metadata = _encoder_metadata(
        encoder,
        name=args.embedder,
        precision=args.precision,
        output_dim=args.output_dim,
        input_dim=probe.shape[1],
    )
    encoder_metadata["model_id"] = "+".join(
        EMBEDDERS.get(name, {}).get("id", name) for name in embedder_names
    )
    encoder_metadata["model_ids"] = [
        EMBEDDERS.get(name, {}).get("id", name) for name in embedder_names
    ]

    for variant in args.variants:
        documents, feature_matrices, summary = generate_variant(
            model,
            tokenizer,
            prompts,
            variant,
            feature_map,
            embedder,
            n_sentences=args.sentences,
            key=key,
            document_ids=ids,
            channel=args.field_channel,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            max_context_tokens=args.max_context_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_rows=args.max_rows,
            max_retry_rounds=args.max_retry_rounds,
            semfield_candidates=args.semfield_candidates,
            field_structure=args.field_structure,
            field_rho=args.field_rho,
            field_blocks=args.field_blocks,
            candidate_schedule=(
                _parse_candidate_schedule(args.candidate_schedule)
                if args.candidate_schedule else None),
            consensus_neighbors=args.consensus_neighbors,
            selection_objective=args.selection_objective,
            moment_lambda=args.moment_lambda,
            correlation_epsilon=args.correlation_epsilon,
            protected_global_epsilon=args.protected_global_epsilon,
            soft_sign_channels=args.soft_sign_channels,
            soft_sign_beta=args.soft_sign_beta,
            proposal_mode=args.proposal_mode,
            proposal_branches=args.proposal_branches,
            proposal_prefix_tokens=args.proposal_prefix_tokens,
            proposal_pool_size=args.proposal_pool_size,
            proposal_temperature=args.proposal_temperature,
            proposal_top_k=args.proposal_top_k,
            proposal_first_token_logprob_gap=(
                args.proposal_first_token_logprob_gap
            ),
            relational_harmonics=args.relational_harmonics,
            sampling_seed_offset=args.sampling_seed_offset,
            trajectory_paths=args.trajectory_paths,
            trajectory_candidates=args.trajectory_candidates,
            capture_candidate_texts=(
                args.candidate_pools_output is not None
                and variant == "semfield"
            ),
        )
        if args.document_ids:
            for local_index, document in enumerate(documents):
                document["document_id"] = int(ids[local_index])
        stem = f"{args.generator}_{variant}"
        sidecar_path = out_dir / f"{stem}.features.npz"
        sidecar = _write_feature_sidecar(
            sidecar_path, documents, feature_matrices, feature_map.n_features
        )
        output_path = out_dir / f"{stem}.json"
        payload = {
            "schema_version": SCHEMA_VERSION,
            "experiment": "exp19_semfield_generate",
            "config": vars(args),
            "source": source,
            "encoder": encoder_metadata,
            "feature_map": feature_map.metadata(),
            "field": {
                "type": (
                    "aer-relative-block-gaussian-v1"
                    if args.field_structure == "aer"
                    else
                    "trajectory-pl-keyed-gaussian-v1"
                    if args.trajectory_paths > 1
                    else
                    "relational-continuous-keyed-fourier-v1"
                    if args.selection_objective == "relational-continuous"
                    else
                    "innovation-span-keyed-gaussian-v1"
                    if args.selection_objective == "innovation-pl"
                    else
                    "soft-sign-semantic-field-v1"
                    if args.selection_objective == "soft-sign"
                    else
                    "sentence-decorrelated-keyed-gaussian-v1"
                    if args.selection_objective == "decorrelated-pl"
                    else
                    "anchored-content-position-keyed-gaussian-v1"
                    if args.field_structure == "anchored"
                    else "content-local-keyed-gaussian-v1"
                ),
                "structure": args.field_structure,
                "rho": float(args.field_rho),
                "blocks": (None if args.field_blocks is None
                           else int(args.field_blocks))
                if args.field_structure != "aer" else int(args.field_blocks or 4),
                "aggregation": (
                    "pairwise-relational-fourier"
                    if args.selection_objective == "relational-continuous"
                    else
                    "span-energy"
                    if args.selection_objective == "innovation-pl"
                    else
                    "sentence-decorrelated"
                    if args.selection_objective == "decorrelated-pl"
                    else "uniform"
                ),
                "test_tail": (
                    "control-calibrated-e-value"
                    if args.field_structure == "aer"
                    else
                    "two-sided-absolute"
                    if args.selection_objective in {
                        "polarity-locked", "correlation-aware-pl",
                        "protected-anchored-pl", "soft-sign",
                        "decorrelated-pl",
                    }
                    else "one-sided-positive"
                ),
                "channel": args.field_channel,
                "relational_harmonics": (
                    int(args.relational_harmonics)
                    if args.selection_objective == "relational-continuous" else None
                ),
                "soft_sign_channels": (
                    int(args.soft_sign_channels)
                    if args.selection_objective == "soft-sign" else None
                ),
                "soft_sign_beta": (
                    float(args.soft_sign_beta)
                    if args.selection_objective == "soft-sign" else None
                ),
                "global_channel": int(args.field_channel),
                "trajectory": ({
                    "enabled": bool(args.trajectory_paths > 1),
                    "paths": int(args.trajectory_paths),
                    "candidates_per_path": int(args.trajectory_candidates),
                    "root_candidates": int(
                        args.trajectory_paths * args.trajectory_candidates
                    ),
                    "selection": (
                        "best child per path, then best final path"
                    ),
                    "budget": (
                        "paths*candidates_per_path candidates per round; "
                        "root pool is the same size"
                    ),
                } if args.trajectory_paths > 1 else None),
                "local_channel": (
                    int(args.field_channel + 1)
                    if args.field_structure in {"anchored", "aer"} else None
                ),
                "aer_lambda_grid": ([0.0, 0.5, 1.0, 2.0, 3.0]
                    if args.field_structure == "aer" else None),
                "key_derivation_domain": KEY_DOMAIN,
                "fusion": (
                    "max-absolute-global-anchored-v1"
                    if args.selection_objective == "protected-anchored-pl"
                    else None
                ),
                "field_scope": "shared key-derived field",
                "experimental_key_seed_exposed": True,
                "moment": {
                    "enabled": args.selection_objective == "moment-hybrid",
                    "type": "centered-diagonal-scatter-v1",
                    "channel": int(args.field_channel + 1),
                    "lambda": float(args.moment_lambda),
                    "description": (
                        "independent keyed Gaussian score of the centered "
                        "per-feature sentence scatter"
                    ),
                },
                "correlation_guard": {
                    "enabled": args.selection_objective == "correlation-aware-pl",
                    "epsilon": float(args.correlation_epsilon),
                    "objective": "prospective-design-effect",
                    "description": (
                        "within the predeclared PL score margin, choose the "
                        "candidate with the smallest prospective design effect"
                    ),
                },
                "protected_global_guard": {
                    "enabled": (
                        args.selection_objective == "protected-anchored-pl"
                    ),
                    "epsilon": float(args.protected_global_epsilon),
                    "primary": "prospective-global-PL-z",
                    "secondary": "prospective-anchored-z",
                    "description": (
                        "maximize anchored evidence only among candidates "
                        "within epsilon of the best global PL candidate"
                    ),
                },
            },
            "key_seed": args.key_seed,
            "detector_requirements": {
                "document_context": (
                    "no document-side metadata is required; all documents "
                    "use the same pre-registered keyed field"
                ),
                "feature_map": "must match the generation fingerprint exactly",
                "preregistration": (
                    "feature bandwidth, dimension, seed, channel, and any "
                    "window family are fixed before inspecting the live score"
                ),
                "blind_text_only_detection": True,
            },
            "summary": summary,
            "prompts": prompts,
            "references": references,
            "documents": documents,
            "feature_sidecar": sidecar,
        }
        if args.candidate_pools_output is not None and variant == "semfield":
            capture_path = Path(args.candidate_pools_output).expanduser().resolve()
            capture_rows = []
            for document in documents:
                capture_rows.append({
                    "document_id": int(document["document_id"]),
                    "sentences": list(document.get("sentences", [])),
                    "candidate_texts": document.get("candidate_texts", []),
                    "selected_indices": list(document.get("selected_indices", [])),
                })
            capture_payload = {
                "format": "semfield-candidate-pools/v1",
                "source_generation": str(output_path),
                "generator": args.generator,
                "documents": len(capture_rows),
                "sentences": int(args.sentences),
                "candidates": int(args.semfield_candidates),
                "polarity_locked": bool(args.polarity_locked),
                "selection_objective": args.selection_objective,
                "rows": capture_rows,
            }
            capture_path.parent.mkdir(parents=True, exist_ok=True)
            capture_tmp = capture_path.with_suffix(capture_path.suffix + ".tmp")
            capture_tmp.write_text(
                json.dumps(capture_payload, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            capture_tmp.replace(capture_path)
            print(f"wrote {capture_path}")
            # Keep the main generation JSON compatible with the historical
            # schema; the potentially large candidate text payload lives only
            # in the explicitly requested audit sidecar.
            for document in documents:
                document.pop("candidate_texts", None)
            payload["documents"] = documents
        temporary = output_path.with_suffix(".tmp.json")
        temporary.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_path)
        print(f"wrote {output_path}")
        print(f"wrote {sidecar_path}")

    if getattr(model, "is_vllm_generator", False):
        model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
