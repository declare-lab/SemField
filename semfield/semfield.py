"""SemField: a calibrated, sentence-local semantic random field.

SemClock obtains an iid rank null by assigning a fresh keyed table to every
state/occurrence clock tick.  That makes its calibration simple, but an edit
to the path can change many later clock addresses.  SemField takes the other
side of the trade-off: the keyed score is a function of sentence meaning
alone, so every surviving sentence keeps exactly the same contribution after
insertion, deletion, cropping, or reordering.  The dependence created by
semantically similar sentences is retained in the null variance instead of
being declared away.

Let ``phi(e)`` be a public unit-norm random-feature map of a normalized
sentence embedding and let the shared secret key provision
``w ~ N(0, I)``.  A sentence receives score

    g(s) = w.T @ phi(e(s)).

For fixed unwatermarked text ``x`` with feature rows ``phi_t``,

    S(x) = sum_t g(s_t) = w.T @ v_x,
    V(x) = ||v_x||^2 = sum_{t,u} phi_t.T @ phi_u,

where ``v_x = sum_t phi_t``.  Hence ``S / sqrt(V)`` is exactly standard
normal in the ideal Gaussian-key model, conditional on the complete text.
The cross terms in ``V`` are the smooth analogue of the squared occupancy
terms exposed by the calibration experiments.

The implementation uses deterministic random Fourier features and an
HMAC-seeded NumPy Gaussian stream.  HMAC provides domain separation and an
unpredictable seed; PCG64 is an experimental deterministic expander, not a
claim of a production cryptographic stream cipher.  Publication experiments
therefore also use control-key randomization.  The finite-sample rank theorem
is exact when the live key and all control keys are iid draws from one key
distribution, independent of the fixed null text.  The reproducible benchmark
derives indexed keys with BLAKE2b instead; treating those concrete fixed keys
as exchangeable is an explicit ideal-hash/PRF assumption, not an unconditional
randomization theorem.  Under the iid-key theorem the rank p-value remains
valid for arbitrary statistics, including a maximum over a fixed window
family.
"""

from __future__ import annotations

import hashlib
import hmac
import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "BENCHMARK_KEY_DOMAIN",
    "FieldSelection",
    "FieldStatistic",
    "KeyedSemanticField",
    "InnovationSemanticField",
    "SoftSignSemanticField",
    "AnchoredSemanticField",
    "AERSemanticField",
    "benchmark_ensemble_key",
    "keyed_gaussian",
    "anchored_field_scores",
    "anchored_field_statistic",
    "protected_anchored_field_scores",
    "protected_anchored_field_statistic",
    "aggregate_features",
    "decorrelated_aggregate",
    "decorrelated_field_scores",
    "decorrelated_field_statistic",
    "prospective_decorrelated_scores",
    "field_statistic",
    "two_sided_field_statistic",
    "innovation_span_scores",
    "innovation_span_zscores",
    "innovation_span_statistic",
    "multiscale_window_features",
    "randomization_pvalue",
]


_FIELD_DOMAIN = bytes.fromhex("706163742d73656d6669656c642d676175737369616e2d763100")
# Fixed compatibility marker for the original shared-field stream.  It is
# constant process-level domain separation, not document-provided data.
_SHARED_FIELD_MARKER = b"b" + (0).to_bytes(8, "big")
BENCHMARK_KEY_DOMAIN = bytes.fromhex("706163742d73656d6669656c642d62656e63686d61726b2d6b65792d656e73656d626c652d7632").decode("ascii")
_NORMAL = NormalDist()


def _key_bytes(value: bytes | bytearray | memoryview | str) -> bytes:
    if isinstance(value, str):
        output = value.encode("utf-8")
    elif isinstance(value, (bytes, bytearray, memoryview)):
        output = bytes(value)
    else:
        raise TypeError("key must be bytes-like or str")
    if not output:
        raise ValueError("key must not be empty")
    return output


def block_positions(n_rows: int, blocks: int | None) -> list[int]:
    """Relative-position block index for each of ``n_rows`` sentences.

    Absolute indexing breaks whenever a rewrite changes the sentence count:
    one inserted sentence shifts every later row onto the wrong keyed weight.
    Rewrites do change counts -- dipper splits 12 sentences into 13.07 on
    average, llm-rewrite merges them to 11.50 -- but they preserve ORDER, so a
    block index computed from relative position survives what an absolute one
    cannot.

    ``blocks=None`` (or >= n_rows) reproduces absolute indexing exactly.
    """

    n = int(n_rows)
    if n <= 0:
        return []
    if blocks is None:
        return list(range(n))
    b = int(blocks)
    if b <= 0:
        raise ValueError("blocks must be positive")
    if b >= n:
        return list(range(n))
    return [min(b - 1, (i * b) // n) for i in range(n)]


def local_position_variance(phi, position_values) -> float:
    """Exact variance of ``sum_t u_{p(t)}^T phi_t`` for shared positions.

    Rows that share a position share one keyed weight, so their contributions
    are perfectly correlated.  The variance is therefore the summed energy of
    each position group's aggregate, which collapses to ``sum(||phi_t||^2)``
    when every position is distinct.
    """

    values = np.asarray(phi, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        return 0.0
    groups: dict[int, np.ndarray] = {}
    for row_index, position in enumerate(position_values):
        key_position = int(position)
        current = groups.get(key_position)
        if current is None:
            groups[key_position] = values[row_index].copy()
        else:
            current += values[row_index]
    return float(sum(float(np.dot(v, v)) for v in groups.values()))


def keyed_gaussian(
    key: bytes | bytearray | memoryview | str,
    dimension: int,
    *,
    channel: int = 0,
) -> np.ndarray:
    """Return a deterministic standard-normal field vector.

    HMAC-SHA256 maps ``(key, channel, dimension)`` to a 256-bit seed.
    NumPy PCG64 expands that seed before ``standard_normal`` is applied.  This
    is fast enough to evaluate thousands of control keys while keeping the
    experimental key streams domain separated.  Concrete cryptographic
    deployment should replace PCG64 with a reviewed AES/ChaCha counter stream;
    the control-key validity theorem is independent of that engineering swap.
    """

    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if channel < 0:
        raise ValueError("channel must be non-negative")
    context = (
        _FIELD_DOMAIN
        + _SHARED_FIELD_MARKER
        + int(channel).to_bytes(8, "big")
        + int(dimension).to_bytes(8, "big")
    )
    digest = hmac.new(_key_bytes(key), context, hashlib.sha256).digest()
    # SeedSequence consumes uint32 words without relying on host endianness.
    seed_words = np.frombuffer(digest, dtype=">u4").astype(np.uint32)
    generator = np.random.Generator(np.random.PCG64(seed_words))
    return generator.standard_normal(int(dimension), dtype=np.float64)


@dataclass(frozen=True)
class FieldSelection:
    index: int
    score: float
    candidate_scores: tuple[float, ...]
    candidate_kernel: float | None
    tied: bool


@dataclass(frozen=True)
class FieldStatistic:
    score_sum: float
    null_variance: float
    z_score: float
    analytic_pvalue: float
    n_sentences: int
    mean_pair_kernel: float | None
    design_effect: float | None


def _feature_matrix(features) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix")
    if values.shape[1] <= 0 or np.any(~np.isfinite(values)):
        raise ValueError("features must be finite and non-empty")
    return values


def _aggregation_weights(
    phi: np.ndarray,
    *,
    aggregation: str,
) -> np.ndarray:
    n = int(phi.shape[0])
    if aggregation == "uniform":
        return np.ones(n, dtype=np.float64)
    raise ValueError(f"unknown SemField aggregation {aggregation!r}")


def aggregate_features(
    features, *, aggregation: str = "uniform",
) -> np.ndarray:
    """Return a public, key-independent document feature aggregate."""

    phi = _feature_matrix(features)
    if phi.shape[0] == 0:
        return np.zeros(phi.shape[1], dtype=np.float64)
    sentence_weights = _aggregation_weights(
        phi, aggregation=aggregation
    )
    return sentence_weights @ phi


def _decorrelated_components(features) -> tuple[np.ndarray, np.ndarray]:
    """Return sentence weights and their public decorrelated aggregate.

    If ``G = phi @ phi.T`` is the sentence Gram matrix, ``G^-1/2``
    suppresses directions repeated by several similar sentences.  This is a
    small, per-document calculation; it is not a fitted embedding whitener.
    The numerical cutoff is the standard machine-precision rank cutoff and is
    therefore not a tuned watermark hyperparameter.
    """

    phi = _feature_matrix(features)
    n = int(phi.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(
            phi.shape[1], dtype=np.float64
        )
    gram = phi @ phi.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    largest = max(0.0, float(eigenvalues[-1]))
    cutoff = largest * max(gram.shape) * np.finfo(np.float64).eps
    inverse_sqrt = np.zeros_like(eigenvalues)
    keep = eigenvalues > cutoff
    inverse_sqrt[keep] = 1.0 / np.sqrt(eigenvalues[keep])
    coefficients = eigenvectors @ (
        inverse_sqrt * (eigenvectors.T @ np.ones(n, dtype=np.float64))
    )
    aggregate = coefficients @ phi
    return coefficients, aggregate


def decorrelated_aggregate(features) -> np.ndarray:
    """Aggregate sentences without repeatedly counting correlated content."""

    return _decorrelated_components(features)[1]


def decorrelated_field_scores(features, weights) -> np.ndarray:
    """Exact normalized field scores for one or many keyed field vectors."""

    aggregate = decorrelated_aggregate(features)
    norm = float(np.linalg.norm(aggregate))
    field_weights = np.asarray(weights, dtype=np.float64)
    if field_weights.ndim == 1:
        if field_weights.size != aggregate.size:
            raise ValueError("weights do not match the feature dimension")
        return np.asarray(
            [float(np.dot(field_weights, aggregate) / norm) if norm > 1e-15 else 0.0],
            dtype=np.float64,
        )
    if field_weights.ndim != 2 or field_weights.shape[1] != aggregate.size:
        raise ValueError("weights do not match the feature dimension")
    if norm <= 1e-15:
        return np.zeros(field_weights.shape[0], dtype=np.float64)
    return field_weights @ (aggregate / norm)


def prospective_decorrelated_scores(
    selected_features, candidate_features, weights,
) -> np.ndarray:
    """Score each candidate with the detector it would produce if selected."""

    candidates = _feature_matrix(candidate_features)
    field_weights = np.asarray(weights, dtype=np.float64).ravel()
    if field_weights.size != candidates.shape[1]:
        raise ValueError("weights do not match the feature dimension")
    selected = np.asarray(selected_features, dtype=np.float64)
    if selected.size == 0:
        selected = np.zeros((0, candidates.shape[1]), dtype=np.float64)
    if selected.ndim != 2 or selected.shape[1] != candidates.shape[1]:
        raise ValueError("selected features do not match candidates")
    return np.asarray([
        decorrelated_field_scores(
            np.vstack((selected, candidate)), field_weights
        )[0]
        for candidate in candidates
    ], dtype=np.float64)


def decorrelated_field_statistic(
    features, weights, *, two_sided: bool = False,
) -> FieldStatistic:
    """Sentence-decorrelated SemField statistic with an exact key null.

    The aggregate depends only on the observed text.  Conditional on that
    text, projecting it onto ``w ~ N(0, I)`` and dividing by its norm is still
    exactly standard normal, just as in ordinary SemField.
    """

    phi = _feature_matrix(features)
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.size != phi.shape[1] or np.any(~np.isfinite(w)):
        raise ValueError("weights do not match the feature dimension")
    n = int(phi.shape[0])
    if n == 0:
        return FieldStatistic(0.0, 0.0, 0.0, 1.0, 0, None, None)
    coefficients, aggregate = _decorrelated_components(phi)
    variance = float(np.dot(aggregate, aggregate))
    score = float(np.dot(w, aggregate))
    if variance <= 1e-15:
        z_score, pvalue = 0.0, 1.0
    else:
        z_score = score / math.sqrt(variance)
        if two_sided:
            score = abs(score)
            z_score = abs(z_score)
            pvalue = 2.0 * _NORMAL.cdf(-z_score)
        else:
            pvalue = _NORMAL.cdf(-z_score)

    raw_variance = float(np.square(phi.sum(axis=0)).sum())
    raw_diagonal = float(np.square(phi).sum())
    mean_pair = (
        (raw_variance - raw_diagonal) / (n * (n - 1)) if n > 1 else None
    )
    independent_variance = float(np.dot(
        np.square(coefficients), np.square(phi).sum(axis=1)
    ))
    return FieldStatistic(
        score_sum=score,
        null_variance=variance,
        z_score=float(z_score),
        analytic_pvalue=float(pvalue),
        n_sentences=n,
        mean_pair_kernel=(None if mean_pair is None else float(mean_pair)),
        design_effect=float(variance / max(independent_variance, 1e-15)),
    )


def field_statistic(
    features, weights, *, aggregation: str = "uniform",
) -> FieldStatistic:
    """Whole-document SemField statistic for precomputed feature rows."""

    phi = _feature_matrix(features)
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.size != phi.shape[1] or np.any(~np.isfinite(w)):
        raise ValueError("weights do not match the feature dimension")
    n = int(phi.shape[0])
    if n == 0:
        return FieldStatistic(0.0, 0.0, 0.0, 1.0, 0, None, None)
    sentence_weights = _aggregation_weights(
        phi, aggregation=aggregation
    )
    aggregate = sentence_weights @ phi
    variance = float(np.dot(aggregate, aggregate))
    score = float(np.dot(w, aggregate))
    if variance <= 1e-15:
        z_score, pvalue = 0.0, 1.0
    else:
        z_score = score / math.sqrt(variance)
        pvalue = _NORMAL.cdf(-z_score)

    row_variances = np.square(phi).sum(axis=1)
    independent_variance = float(np.dot(
        np.square(sentence_weights), row_variances
    ))
    raw_aggregate = phi.sum(axis=0)
    raw_variance = float(np.dot(raw_aggregate, raw_aggregate))
    raw_diagonal = float(row_variances.sum())
    if n > 1:
        mean_pair = (raw_variance - raw_diagonal) / (n * (n - 1))
    else:
        mean_pair = None
    return FieldStatistic(
        score_sum=score,
        null_variance=variance,
        z_score=float(z_score),
        analytic_pvalue=float(pvalue),
        n_sentences=n,
        mean_pair_kernel=(None if mean_pair is None else float(mean_pair)),
        design_effect=float(variance / max(independent_variance, 1e-15)),
    )


def two_sided_field_statistic(
    features, weights, *, aggregation: str = "uniform",
) -> FieldStatistic:
    """Absolute SemField statistic with its exact two-sided Gaussian p-value."""

    result = field_statistic(
        features, weights, aggregation=aggregation,
    )
    if result.null_variance <= 1e-15:
        return result
    absolute_z = abs(result.z_score)
    return FieldStatistic(
        score_sum=abs(result.score_sum),
        null_variance=result.null_variance,
        z_score=absolute_z,
        analytic_pvalue=float(2.0 * _NORMAL.cdf(-absolute_z)),
        n_sentences=result.n_sentences,
        mean_pair_kernel=result.mean_pair_kernel,
        design_effect=result.design_effect,
    )


class KeyedSemanticField:
    """Keyed sentence scorer and two-candidate tournament selector."""

    def __init__(self, feature_map,
                 key: bytes | bytearray | memoryview | str,
                 *, channel: int = 0):
        self.feature_map = feature_map
        self.key = _key_bytes(key)
        self.channel = int(channel)
        self.weights = keyed_gaussian(
            self.key, feature_map.n_features, channel=self.channel
        )

    def score_features(self, features) -> np.ndarray:
        phi = _feature_matrix(features)
        if phi.shape[1] != self.feature_map.n_features:
            raise ValueError("features do not match this field")
        return phi @ self.weights

    def score_embeddings(self, embeddings) -> np.ndarray:
        return self.score_features(self.feature_map.transform(embeddings))

    def select_features(
        self, features, rng: np.random.Generator | None = None
    ) -> FieldSelection:
        phi = _feature_matrix(features)
        if phi.shape[0] < 2:
            raise ValueError("SemField selection requires at least two candidates")
        scores = self.score_features(phi)
        largest = float(scores.max())
        best = np.flatnonzero(scores == largest)
        tied = best.size > 1
        if tied:
            rng = np.random.default_rng() if rng is None else rng
            index = int(rng.choice(best))
        else:
            index = int(best[0])
        kernel = (float(np.dot(phi[0], phi[1]))
                  if phi.shape[0] == 2 else None)
        return FieldSelection(
            index=index,
            score=float(scores[index]),
            candidate_scores=tuple(float(value) for value in scores),
            candidate_kernel=kernel,
            tied=bool(tied),
        )

    def select_embeddings(
        self, embeddings, rng: np.random.Generator | None = None
    ) -> FieldSelection:
        return self.select_features(self.feature_map.transform(embeddings), rng)

    def detect_features(self, features) -> FieldStatistic:
        return field_statistic(features, self.weights)

    def detect_embeddings(self, embeddings) -> FieldStatistic:
        return self.detect_features(self.feature_map.transform(embeddings))


def _innovation_row_basis(features: np.ndarray) -> np.ndarray:
    """Orthonormal basis for the observed sentence-feature row span."""

    phi = _feature_matrix(features)
    if phi.shape[0] == 0:
        return np.empty((phi.shape[1], 0), dtype=np.float64)
    _u, singular, vt = np.linalg.svd(phi, full_matrices=False)
    if singular.size == 0:
        return np.empty((phi.shape[1], 0), dtype=np.float64)
    cutoff = float(singular[0]) * max(phi.shape) * np.finfo(np.float64).eps
    rank = int(np.count_nonzero(singular > cutoff))
    return vt[:rank].T


def innovation_span_scores(features, weights) -> np.ndarray:
    """Energy of one or more keyed fields in the sentence-feature span.

    If ``U`` is an orthonormal basis of the observed row span, the score is
    ``||U.T w||^2``.  The candidate selector uses the one-row version of the
    same statistic (absolute keyed projection); controls use this exact span
    energy too, so the maximization is calibrated rather than free evidence.
    """

    phi = _feature_matrix(features)
    field = np.asarray(weights, dtype=np.float64)
    if field.ndim == 1:
        field = field[None, :]
    if field.ndim != 2 or field.shape[1] != phi.shape[1]:
        raise ValueError("weights do not match the feature dimension")
    basis = _innovation_row_basis(phi)
    if basis.shape[1] == 0:
        return np.zeros(field.shape[0], dtype=np.float64)
    projections = field @ basis
    return np.square(projections).sum(axis=1)


def innovation_span_zscores(features, weights) -> np.ndarray:
    """Standardize span energy under the conditional Gaussian null."""

    phi = _feature_matrix(features)
    rank = int(_innovation_row_basis(phi).shape[1])
    energies = innovation_span_scores(phi, weights)
    if rank <= 0:
        return np.zeros_like(energies)
    return (energies - float(rank)) / math.sqrt(2.0 * float(rank))


def innovation_span_statistic(
    features, weights, *, two_sided: bool = False,
) -> FieldStatistic:
    """Whole-document statistic for the full-span innovation carrier."""

    del two_sided  # span energy is intrinsically non-negative
    phi = _feature_matrix(features)
    rank = int(_innovation_row_basis(phi).shape[1])
    energy = float(innovation_span_scores(phi, weights)[0])
    if rank <= 0:
        return FieldStatistic(0.0, 0.0, 0.0, 1.0, 0, None, None)
    variance = float(2.0 * rank)
    z_score = (energy - rank) / math.sqrt(variance)
    diagonal = float(np.square(phi).sum())
    aggregate_energy = float(np.square(phi.sum(axis=0)).sum())
    n = int(phi.shape[0])
    mean_pair = (
        (aggregate_energy - diagonal) / float(n * (n - 1))
        if n > 1 else None
    )
    return FieldStatistic(
        score_sum=energy,
        null_variance=variance,
        z_score=float(z_score),
        analytic_pvalue=float(_NORMAL.cdf(-z_score)),
        n_sentences=n,
        mean_pair_kernel=mean_pair,
        design_effect=1.0,
    )


class InnovationSemanticField:
    """Continuous full-span innovation field.

    Candidate selection scores a residual direction by ``|w.T phi|``.  At
    detection, all observed residual rows are combined through their span
    energy.  The key, embedding representation, and sentence candidates stay
    continuous; no binary signature or per-document fitted map is introduced.
    """

    structure = "innovation-span-keyed-gaussian-v1"

    def __init__(self, feature_map, key, *, channel: int = 0):
        self.feature_map = feature_map
        self.key = _key_bytes(key)
        self.channel = int(channel)
        self.weights = keyed_gaussian(
            self.key, feature_map.n_features, channel=self.channel
        )

    def score_features(self, features) -> np.ndarray:
        phi = _feature_matrix(features)
        if phi.shape[1] != self.feature_map.n_features:
            raise ValueError("features do not match this field")
        return phi @ self.weights

    def score_embeddings(self, embeddings) -> np.ndarray:
        return self.score_features(self.feature_map.transform(embeddings))

    def detect_features(self, features, *, two_sided: bool = False) -> FieldStatistic:
        return innovation_span_statistic(
            features, self.weights, two_sided=two_sided
        )


class SoftSignSemanticField:
    """A bounded, multi-coordinate SemField scorer.

    Each sentence is evaluated against a small set of independent keyed
    directions.  The target polarity is position-specific and the tanh keeps
    one unusually large projection from dominating the document statistic.
    Candidate generation remains ordinary iid sampling; this class changes
    only the key-dependent selection statistic.
    """

    _DIRECTION_OFFSET = 10_000
    _TARGET_OFFSET = 20_000

    def __init__(self, dimension: int,
                 key: bytes | bytearray | memoryview | str,
                 *, channels: int = 8, beta: float = 4.0,
                 channel: int = 0):
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if channels <= 0:
            raise ValueError("soft-sign channels must be positive")
        if not math.isfinite(beta) or beta <= 0.0:
            raise ValueError("soft-sign beta must be finite and positive")
        self.dimension = int(dimension)
        self.key = _key_bytes(key)
        self.channels = int(channels)
        self.beta = float(beta)
        self.channel = int(channel)
        directions = np.asarray([
            keyed_gaussian(
                self.key, self.dimension,
                channel=self.channel + self._DIRECTION_OFFSET + index,
            )
            for index in range(self.channels)
        ], dtype=np.float64)
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        self.directions = directions / np.maximum(norms, 1e-15)
        # Feature rows are unit-normalized, so 1/sqrt(d) is the fixed null
        # scale of a unit Gaussian direction. It is not estimated from the
        # watermarked candidates or from an attack.
        self.projection_scale = 1.0 / math.sqrt(self.dimension)

    def target_signs(self, position: int) -> np.ndarray:
        if position < 0:
            raise ValueError("position must be non-negative")
        values = keyed_gaussian(
            self.key, self.channels,
            channel=self.channel + self._TARGET_OFFSET + int(position),
        )
        return np.where(values >= 0.0, 1.0, -1.0)

    def contributions(self, features, position: int) -> np.ndarray:
        phi = _feature_matrix(features)
        if phi.shape[1] != self.dimension:
            raise ValueError("features do not match this soft-sign field")
        projections = (phi @ self.directions.T) / self.projection_scale
        return np.tanh(self.beta * projections) * self.target_signs(position)

    def score_candidates(
        self, features, position: int, *, score_sum: float = 0.0,
        null_variance: float = 0.0,
    ) -> np.ndarray:
        """Return each candidate's prospective normalized document score."""

        values = self.contributions(features, position)
        numerators = float(score_sum) + values.sum(axis=1) / self.channels
        variances = (
            float(null_variance)
            + np.square(values).sum(axis=1) / (self.channels ** 2)
        )
        return np.divide(
            numerators,
            np.sqrt(variances),
            out=np.full(values.shape[0], -np.inf, dtype=np.float64),
            where=variances > 1e-15,
        )

    def detect_features(
        self, features, *, two_sided: bool = False
    ) -> FieldStatistic:
        phi = _feature_matrix(features)
        if phi.shape[1] != self.dimension:
            raise ValueError("features do not match this soft-sign field")
        if phi.shape[0] == 0:
            return FieldStatistic(0.0, 0.0, 0.0, 1.0, 0, None, None)
        contributions = np.asarray([
            self.contributions(phi[position:position + 1], position)[0]
            for position in range(phi.shape[0])
        ], dtype=np.float64)
        # The scale is the conditional sum-of-squares under independent target
        # polarities. Control-key calibration remains the authoritative test.
        score = float(contributions.sum() / self.channels)
        variance = float(np.square(contributions).sum() / (self.channels ** 2))
        if variance <= 1e-15:
            z_score, pvalue = 0.0, 1.0
        else:
            z_score = score / math.sqrt(variance)
            if two_sided:
                score = abs(score)
                z_score = abs(z_score)
                pvalue = 2.0 * _NORMAL.cdf(-z_score)
            else:
                pvalue = _NORMAL.cdf(-z_score)
        raw_variance = float(np.square(phi.sum(axis=0)).sum())
        row_energy = float(np.square(phi).sum())
        mean_pair = (
            (raw_variance - row_energy) / (phi.shape[0] * (phi.shape[0] - 1))
            if phi.shape[0] > 1 else None
        )
        return FieldStatistic(
            score_sum=score,
            null_variance=variance,
            z_score=float(z_score),
            analytic_pvalue=float(pvalue),
            n_sentences=int(phi.shape[0]),
            mean_pair_kernel=(None if mean_pair is None else float(mean_pair)),
            design_effect=None,
        )


def _anchored_components(
    features: np.ndarray,
    key: bytes,
    *,
    rho: float,
    channel: int,
    positions: Sequence[int] | None = None,
) -> tuple[np.ndarray, float]:
    """Return the keyed score vector and its exact conditional variance.

    The global component uses one Gaussian field for the document aggregate.
    The local component uses an independently derived field for each sentence
    position.  For fixed feature rows, the two components have conditional
    variance ``rho * ||sum(phi)||^2 + (1-rho) * sum(||phi_t||^2)``.
    ``positions`` is useful for a scan over a contiguous portion of a text.
    """

    phi = _feature_matrix(features)
    if not math.isfinite(rho) or not 0.0 <= float(rho) <= 1.0:
        raise ValueError("rho must lie in [0, 1]")
    if channel < 0:
        raise ValueError("channel must be non-negative")
    n, dimension = phi.shape
    if positions is None:
        position_values = list(range(n))
    else:
        position_values = [int(value) for value in positions]
        if len(position_values) != n:
            raise ValueError("positions must align with feature rows")
        if any(value < 0 for value in position_values):
            raise ValueError("positions must be non-negative")
    aggregate = phi.sum(axis=0)
    global_weight = keyed_gaussian(
        key, dimension, channel=int(channel)
    )
    local_scores = np.zeros(n, dtype=np.float64)
    if n:
        for row_index, position in enumerate(position_values):
            local_weight = keyed_gaussian(
                key, dimension, channel=int(channel) + 1 + position
            )
            local_scores[row_index] = float(np.dot(local_weight, phi[row_index]))
    global_score = float(np.dot(global_weight, aggregate))
    # Rows sharing a position share one keyed weight, so their local terms are
    # perfectly correlated and the exact local variance is the energy of each
    # position's summed rows, not the row-wise energy.  With distinct
    # positions every group is a single row and this reduces to
    # ``sum(||phi_t||^2)`` exactly, so existing callers are unaffected.
    local_variance = float(local_position_variance(phi, position_values))
    variance = float(
        float(rho) * np.dot(aggregate, aggregate)
        + (1.0 - float(rho)) * local_variance
    )
    score = math.sqrt(float(rho)) * global_score
    if n:
        score += math.sqrt(1.0 - float(rho)) * float(local_scores.sum())
    return np.asarray([score], dtype=np.float64), variance


def anchored_field_scores(
    features,
    keys: Sequence[bytes | bytearray | memoryview | str],
    *,
    rho: float,
    channel: int = 0,
    positions: Sequence[int] | None = None,
) -> np.ndarray:
    """Return normalized anchored-field scores for several keys.

    This is the vectorized detector-facing form.  Keys are independent draws
    from the same keyed-Gaussian construction; no binary code or Hamming
    distance is involved.
    """

    phi = _feature_matrix(features)
    values: list[float] = []
    variance: float | None = None
    for key in keys:
        result, current_variance = _anchored_components(
            phi, _key_bytes(key), rho=rho, channel=channel,
            positions=positions,
        )
        if variance is None:
            variance = current_variance
        elif not math.isclose(
            variance, current_variance, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise RuntimeError("anchored variance changed across keys")
        values.append(float(result[0]))
    if variance is None or variance <= 1e-15:
        return np.zeros(len(values), dtype=np.float64)
    return np.asarray(values, dtype=np.float64) / math.sqrt(variance)


def anchored_field_statistic(
    features,
    key: bytes | bytearray | memoryview | str,
    *,
    rho: float,
    channel: int = 0,
    positions: Sequence[int] | None = None,
    two_sided: bool = False,
) -> FieldStatistic:
    """Whole-document statistic for the anchored correlated field."""

    phi = _feature_matrix(features)
    if phi.shape[0] == 0:
        return FieldStatistic(0.0, 0.0, 0.0, 1.0, 0, None, None)
    raw_scores, mixed_variance = _anchored_components(
        phi, _key_bytes(key), rho=rho, channel=channel,
        positions=positions,
    )
    raw_score = float(raw_scores[0])
    score = (
        raw_score / math.sqrt(mixed_variance)
        if mixed_variance > 1e-15 else 0.0
    )
    if two_sided:
        score = abs(score)
        pvalue = float(2.0 * _NORMAL.cdf(-score))
    else:
        pvalue = float(_NORMAL.cdf(-score))
    raw_variance = float(np.square(phi.sum(axis=0)).sum())
    diagonal = float(np.square(phi).sum())
    n = int(phi.shape[0])
    mean_pair = (
        (raw_variance - diagonal) / float(n * (n - 1))
        if n > 1 else None
    )
    # The denominator is the exact mixed variance, so the design effect is
    # reported relative to the independent sentence-row energy.
    return FieldStatistic(
        score_sum=raw_score,
        null_variance=mixed_variance,
        z_score=score,
        analytic_pvalue=pvalue,
        n_sentences=n,
        mean_pair_kernel=(None if mean_pair is None else float(mean_pair)),
        design_effect=float(mixed_variance / max(diagonal, 1e-15)),
    )


def protected_anchored_field_scores(
    features,
    keys: Sequence[bytes | bytearray | memoryview | str],
    *,
    rho: float,
    channel: int = 0,
    positions: Sequence[int] | None = None,
) -> np.ndarray:
    """Return the preregistered maximum of global and anchored evidence.

    The global PL statistic is invariant to sentence order.  The anchored
    statistic adds position-local evidence that is useful when it survives.
    The complete maximum is calibrated for every control key (and against
    untouched human text in the comparison table), so the extra look is part
    of the null rather than free evidence.
    """

    phi = _feature_matrix(features)
    anchored = np.abs(anchored_field_scores(
        phi, keys, rho=rho, channel=channel, positions=positions,
    ))
    aggregate = phi.sum(axis=0)
    norm = float(np.linalg.norm(aggregate))
    if norm <= 1e-15:
        global_scores = np.zeros(len(keys), dtype=np.float64)
    else:
        unit = aggregate / norm
        global_scores = np.asarray([
            abs(float(np.dot(
                keyed_gaussian(key, phi.shape[1], channel=channel),
                unit,
            )))
            for key in keys
        ], dtype=np.float64)
    return np.maximum(global_scores, anchored)


def protected_anchored_field_statistic(
    features,
    key: bytes | bytearray | memoryview | str,
    *,
    rho: float,
    channel: int = 0,
    positions: Sequence[int] | None = None,
) -> FieldStatistic:
    """Two-channel statistic for protected Anchored-PL.

    The score is ``max(|Z_global|, |Z_anchored|)``.  Its analytic p-value is
    the conservative Bonferroni bound for two two-sided Gaussian tests.
    Benchmark operating points use the empirical human null, while the
    randomization test applies the same maximum to every control key.
    """

    phi = _feature_matrix(features)
    if phi.shape[0] == 0:
        return FieldStatistic(0.0, 1.0, 0.0, 1.0, 0, None, None)
    fused = float(protected_anchored_field_scores(
        phi, [key], rho=rho, channel=channel,
        positions=positions,
    )[0])
    anchored = anchored_field_statistic(
        phi, key, rho=rho, channel=channel,
        positions=positions, two_sided=True,
    )
    return FieldStatistic(
        score_sum=fused,
        null_variance=1.0,
        z_score=fused,
        analytic_pvalue=float(min(1.0, 4.0 * _NORMAL.cdf(-fused))),
        n_sentences=anchored.n_sentences,
        mean_pair_kernel=anchored.mean_pair_kernel,
        design_effect=anchored.design_effect,
    )


class AnchoredSemanticField:
    """Correlated global/local continuous semantic field.

    ``rho=1`` is the existing global SemField field.  As ``rho`` decreases,
    the local position-indexed component contributes more evidence.  The
    local fields are continuous Gaussian vectors, not AliMark-style binary
    signatures, and are used only through the same semantic feature rows and
    prospective candidate tournament.
    """

    structure = "anchored-content-position-keyed-gaussian-v1"

    def __init__(
        self,
        feature_map,
        key: bytes | bytearray | memoryview | str,
        *,
        rho: float = 0.5,
        channel: int = 0,
        blocks: int | None = None,
    ):
        if not math.isfinite(rho) or not 0.0 <= float(rho) <= 1.0:
            raise ValueError("rho must lie in [0, 1]")
        if channel < 0:
            raise ValueError("channel must be non-negative")
        if blocks is not None and int(blocks) <= 0:
            raise ValueError("blocks must be positive")
        self.feature_map = feature_map
        self.key = _key_bytes(key)
        self.rho = float(rho)
        self.channel = int(channel)
        self.blocks = None if blocks is None else int(blocks)
        self.weights = keyed_gaussian(
            self.key, feature_map.n_features,
            channel=self.channel,
        )
        self.global_weights = self.weights
        self._local_cache: dict[int, np.ndarray] = {}

    def local_weights(self, position: int) -> np.ndarray:
        position = int(position)
        if position < 0:
            raise ValueError("position must be non-negative")
        cached = self._local_cache.get(position)
        if cached is None:
            cached = keyed_gaussian(
                self.key, self.feature_map.n_features,
                channel=self.channel + 1 + position,
            )
            self._local_cache[position] = cached
        return cached

    def score_features(self, features) -> np.ndarray:
        """Score rows with the global component (for matched null diagnostics)."""

        phi = _feature_matrix(features)
        if phi.shape[1] != self.feature_map.n_features:
            raise ValueError("features do not match this field")
        return phi @ self.weights

    def score_embeddings(self, embeddings) -> np.ndarray:
        return self.score_features(self.feature_map.transform(embeddings))

    def prospective_scores(
        self,
        features,
        aggregate,
        local_score: float,
        local_variance: float,
        *,
        position: int,
        block_sum=None,
    ) -> np.ndarray:
        """Score candidates using the prospective mixed self-normalized field.

        ``block_sum`` is the running sum of rows already placed at
        ``position``.  Adding a row there moves that group's energy from
        ``||s||^2`` to ``||s + phi||^2``, so the local variance grows by
        ``2 s.phi + ||phi||^2``.  With one row per position ``s`` is zero and
        this is the plain ``||phi||^2`` increment.
        """

        phi = _feature_matrix(features)
        current = np.asarray(aggregate, dtype=np.float64).ravel()
        if current.size != phi.shape[1]:
            raise ValueError("aggregate does not match feature width")
        local = self.local_weights(position)
        prospective_global = current[None, :] + phi
        local_scores = float(local_score) + phi @ local
        if block_sum is None:
            local_increment = np.square(phi).sum(axis=1)
        else:
            block = np.asarray(block_sum, dtype=np.float64).ravel()
            if block.size != phi.shape[1]:
                raise ValueError("block_sum does not match feature width")
            local_increment = (
                2.0 * (phi @ block) + np.square(phi).sum(axis=1)
            )
        variance = (
            self.rho * np.square(prospective_global).sum(axis=1)
            + (1.0 - self.rho) * (
                float(local_variance) + local_increment
            )
        )
        numerator = (
            math.sqrt(self.rho) * (prospective_global @ self.weights)
            + math.sqrt(1.0 - self.rho) * local_scores
        )
        return np.divide(
            numerator,
            np.sqrt(variance),
            out=np.full(phi.shape[0], -np.inf, dtype=np.float64),
            where=variance > 1e-15,
        )

    def detect_features(self, features, *, two_sided: bool = False) -> FieldStatistic:
        phi = _feature_matrix(features)
        return anchored_field_statistic(
            phi, self.key, rho=self.rho,
            channel=self.channel, two_sided=two_sided,
            positions=block_positions(phi.shape[0], self.blocks),
        )

    def detect_embeddings(self, embeddings, *, two_sided: bool = False) -> FieldStatistic:
        return self.detect_features(
            self.feature_map.transform(embeddings), two_sided=two_sided
        )


class AERSemanticField:
    """Blockwise analog SemField with erasure-safe e-value decoding.

    Each relative-position block has its own keyed Gaussian direction.  The
    class deliberately exposes prospective candidate scores so generation can
    use the same objective that detection evaluates.  It does not use signs,
    bits, regions, Hamming distance, or a learned transform.
    """

    structure = "aer-relative-block-gaussian-v1"

    def __init__(self, feature_map, key, *, blocks: int = 4,
                 channel: int = 0, lambda_grid=(0.0, .5, 1.0, 2.0, 3.0)):
        if int(blocks) <= 0:
            raise ValueError("blocks must be positive")
        grid = np.asarray(lambda_grid, dtype=np.float64).ravel()
        if grid.size == 0 or np.any(~np.isfinite(grid)) or np.any(grid < 0):
            raise ValueError("lambda_grid must be finite and non-negative")
        self.feature_map, self.key = feature_map, _key_bytes(key)
        self.blocks, self.channel, self.lambda_grid = int(blocks), int(channel), grid
        self._weights = {}

    def position(self, row_index: int, n_rows: int) -> int:
        return block_positions(int(n_rows), self.blocks)[min(int(row_index), max(0, int(n_rows)-1))]

    def weights(self, block: int) -> np.ndarray:
        block = int(block)
        if block not in self._weights:
            self._weights[block] = keyed_gaussian(
                self.key, self.feature_map.n_features,
                channel=self.channel + 1 + block)
        return self._weights[block]

    # Shared name used by the existing anchored-generation bookkeeping.
    local_weights = weights

    def prospective_scores(self, features, block: int, block_sum=None) -> np.ndarray:
        phi = _feature_matrix(features)
        if block_sum is None:
            aggregate = phi
        else:
            base = np.asarray(block_sum, dtype=np.float64).ravel()
            if base.size != phi.shape[1]:
                raise ValueError("block_sum does not match feature width")
            aggregate = phi + base[None, :]
        norms = np.linalg.norm(aggregate, axis=1)
        return np.divide(
            aggregate @ self.weights(block), norms,
            out=np.full(phi.shape[0], -np.inf, dtype=np.float64),
            where=norms > 1e-15,
        )

    def score_features(self, features) -> np.ndarray:
        """Compatibility scorer used while generating the null condition."""
        return self.prospective_scores(features, 0)

    def e_value(self, z: float) -> float:
        lam = self.lambda_grid
        return float(np.mean(np.exp(lam * float(z) - .5 * lam * lam)))

    def detect_features(self, features) -> FieldStatistic:
        phi = _feature_matrix(features)
        n = phi.shape[0]
        if n == 0:
            return FieldStatistic(0.0, 1.0, 0.0, 1.0, 0, None, None)
        positions = block_positions(n, self.blocks)
        block_z, block_e = [], []
        for block in range(min(self.blocks, n)):
            rows = phi[np.asarray(positions) == block]
            if not len(rows):
                continue
            aggregate = rows.sum(axis=0)
            denom = float(np.linalg.norm(aggregate))
            z = float(np.dot(self.weights(block), aggregate) / denom) if denom > 1e-15 else 0.0
            block_z.append(z); block_e.append(self.e_value(z))
        e_value = float(np.mean(block_e)) if block_e else 1.0
        # log(e) is a monotone document score for ranking.  The control-key
        # evaluator, rather than a Gaussian p-value, supplies the operating
        # threshold for this nonlinear statistic.
        score = math.log(max(e_value, 1e-300))
        return FieldStatistic(score, 1.0, score, 1.0, n, None, None)


def multiscale_window_features(
    features,
    *,
    min_length: int = 8,
    max_length: int | None = None,
    sizes: Sequence[int] | None = None,
    stride_fraction: float = 0.5,
) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    """Return unit aggregate features for a preregistered multiscale scan.

    Default sizes double from ``min_length`` and include the largest allowed
    window.  Each scale advances by half a window.  This is O(T log T) rather
    than an all-interval O(T^2) family and makes its multiplicity explicit.
    """

    phi = _feature_matrix(features)
    n = int(phi.shape[0])
    if min_length <= 0:
        raise ValueError("min_length must be positive")
    if not 0.0 < stride_fraction <= 1.0:
        raise ValueError("stride_fraction must lie in (0,1]")
    if n == 0:
        return np.zeros((0, phi.shape[1]), dtype=np.float64), ()
    maximum = n if max_length is None else min(n, int(max_length))
    minimum = min(int(min_length), maximum)
    if sizes is None:
        chosen = []
        width = minimum
        while width <= maximum:
            chosen.append(width)
            width *= 2
        if maximum not in chosen:
            chosen.append(maximum)
    else:
        chosen = sorted({int(value) for value in sizes
                         if minimum <= int(value) <= maximum})
        if maximum not in chosen:
            chosen.append(maximum)
    prefix = np.vstack([
        np.zeros((1, phi.shape[1]), dtype=np.float64),
        np.cumsum(phi, axis=0),
    ])
    vectors: list[np.ndarray] = []
    bounds: list[tuple[int, int]] = []
    for width in chosen:
        stride = max(1, int(round(width * stride_fraction)))
        starts = list(range(0, n - width + 1, stride))
        final_start = n - width
        if final_start not in starts:
            starts.append(final_start)
        for start in starts:
            stop = start + width
            aggregate = prefix[stop] - prefix[start]
            norm = float(np.linalg.norm(aggregate))
            if norm <= 1e-15:
                continue
            vectors.append(aggregate / norm)
            bounds.append((start, stop))
    if not vectors:
        return np.zeros((0, phi.shape[1]), dtype=np.float64), ()
    return np.asarray(vectors, dtype=np.float64), tuple(bounds)


def randomization_pvalue(live: float, controls: Iterable[float]) -> tuple[float, int, int]:
    """Conservative exchangeable rank p-value with ties counted against live."""

    values = np.asarray(list(controls), dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("at least one control statistic is required")
    if not math.isfinite(live) or np.any(~np.isfinite(values)):
        raise ValueError("live and control statistics must be finite")
    exceedances = int(np.count_nonzero(values >= float(live)))
    return (float((1 + exceedances) / (values.size + 1)),
            exceedances, int(values.size))


def benchmark_ensemble_key(ensemble_seed: int, index: int) -> bytes:
    """Return one reproducible key from the public benchmark ensemble.

    Index zero is reserved for the live benchmark key; indices ``1..R`` are
    its control keys.  The symmetric derivation makes the construction
    auditable, but its outputs are deterministic.  Exact exchangeable-rank
    validity still requires iid random keys; for these benchmark keys it rests
    on the stated ideal-hash/PRF model for BLAKE2b.
    """

    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise TypeError("benchmark key index must be an integer")
    if int(index) < 0:
        raise ValueError("benchmark key index must be non-negative")
    material = (
        BENCHMARK_KEY_DOMAIN.encode("ascii")
        + b"\x00"
        + str(int(ensemble_seed)).encode("ascii")
        + b"\x00"
        + int(index).to_bytes(8, "big")
    )
    return hashlib.blake2b(material, digest_size=32).digest()
