"""Scheme-specific null laws for discrete semantic watermarks.

This module keeps three designs separate.  They have the same marginal green
rate, but they do *not* have the same variance or exact null distribution.

``static_iid``
    Each semantic cell receives an independent Bernoulli mark.

``global_fixed_size``
    One uniformly random subset of exactly ``G`` of ``M`` cells is marked.

``semstamp_row``
    For every previous-sentence cell ``a``, exactly ``G`` of the ``K`` possible
    next cells are marked.  Rows are independent in the ideal-key model.  This
    is the finite-population structure induced by SemStamp's
    previous-signature-conditioned masks.

Every calculation is conditional on the observed cell path.  Consequently it
makes no stochastic assumption about the sentences or their cell labels.  Its
randomness is the setup key/mask table, which must be sampled independently of
null text.  The row-stratified formulas additionally assume independent,
uniform fixed-size row masks; use a control-key randomization test instead if a
concrete implementation does not satisfy that idealization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import comb
from typing import Iterable

import numpy as np
from scipy.stats import norm

__all__ = [
    "NullMoments",
    "static_iid_moments",
    "static_iid_pmf",
    "global_fixed_size_moments",
    "global_fixed_size_pmf",
    "semstamp_transition_counts",
    "semstamp_row_moments",
    "semstamp_row_pmf",
    "semstamp_row_pvalue",
    "upper_tail_pvalue",
    "minimum_upper_tail_pvalue",
    "naive_z_score",
    "naive_z_pvalue",
    "simulate_semstamp_row_scores",
]


@dataclass(frozen=True)
class NullMoments:
    """First two null moments and their comparison with a sentence z-test."""

    mean: float
    variance: float
    naive_variance: float
    design_effect: float
    n_trials: int
    success_probability: float

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def _integer_vector(values: Iterable[int], name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if raw.size == 0:
        return np.zeros(0, dtype=np.int64)
    try:
        numeric = raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integers") from exc
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} must be finite")
    if not np.all(numeric == np.floor(numeric)):
        raise ValueError(f"{name} must contain integers")
    if np.any(numeric < 0):
        raise ValueError(f"{name} must be non-negative")
    return numeric.astype(np.int64)


def _integer_matrix(values, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    try:
        numeric = raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integers") from exc
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} must be finite")
    if not np.all(numeric == np.floor(numeric)):
        raise ValueError(f"{name} must contain integers")
    if np.any(numeric < 0):
        raise ValueError(f"{name} must be non-negative")
    return numeric.astype(np.int64)


def _probability(value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    return value


def _green_count(total_cells: int, n_green: int) -> int:
    total_cells = int(total_cells)
    n_green = int(n_green)
    if total_cells <= 0:
        raise ValueError("the cell space must be non-empty")
    if not 0 <= n_green <= total_cells:
        raise ValueError("n_green must lie between zero and the cell count")
    return n_green


def _make_moments(mean: float, variance: float, n_trials: int,
                  probability: float) -> NullMoments:
    naive = float(n_trials * probability * (1.0 - probability))
    # Roundoff can produce a negative number of order machine precision in a
    # perfectly balanced fixed-size design.
    variance = max(0.0, float(variance))
    effect = float(variance / naive) if naive > 0.0 else float("nan")
    return NullMoments(
        mean=float(mean),
        variance=variance,
        naive_variance=naive,
        design_effect=effect,
        n_trials=int(n_trials),
        success_probability=float(probability),
    )


# ---------------------------------------------------------------------------
# Independent static cell marks
# ---------------------------------------------------------------------------


def static_iid_moments(multiplicities: Iterable[int],
                       probability: float) -> NullMoments:
    """Moments for ``S = sum_b n_b B_b``, ``B_b iid Bernoulli(p)``.

    ``multiplicities`` may contain zero-count cells; they have no effect.
    """

    n = _integer_vector(multiplicities, "multiplicities")
    p = _probability(probability)
    total = int(n.sum())
    variance = p * (1.0 - p) * float(np.square(n.astype(float)).sum())
    return _make_moments(p * total, variance, total, p)


def static_iid_pmf(multiplicities: Iterable[int],
                   probability: float) -> np.ndarray:
    """Exact conditional PMF for independent static Bernoulli cell marks."""

    n = _integer_vector(multiplicities, "multiplicities")
    p = _probability(probability)
    total = int(n.sum())
    pmf = np.zeros(total + 1, dtype=float)
    pmf[0] = 1.0
    for weight in n:
        if weight == 0:
            continue
        new = pmf * (1.0 - p)
        new[weight:] += p * pmf[: total + 1 - weight]
        pmf = new
    return pmf


# ---------------------------------------------------------------------------
# One global fixed-size subset
# ---------------------------------------------------------------------------


def global_fixed_size_moments(multiplicities: Iterable[int],
                              n_cells_total: int,
                              n_green: int) -> NullMoments:
    """Moments when exactly ``n_green`` of all cells are selected uniformly.

    Missing entries in ``multiplicities`` are treated as zero-count cells, so
    callers may pass either a full length-``n_cells_total`` vector or only the
    occupied-cell counts.
    """

    n = _integer_vector(multiplicities, "multiplicities")
    M = int(n_cells_total)
    G = _green_count(M, n_green)
    if n.size > M:
        raise ValueError("multiplicities describes more cells than the space")
    occupied = int(np.count_nonzero(n))
    if occupied > M:
        raise ValueError("more occupied cells than cells in the space")

    p = G / M
    total = int(n.sum())
    if M == 1:
        variance = 0.0
    else:
        sum_squares = float(np.square(n.astype(float)).sum())
        variance = p * (1.0 - p) * (
            M * sum_squares - total * total
        ) / (M - 1)
    return _make_moments(p * total, variance, total, p)


def global_fixed_size_pmf(multiplicities: Iterable[int],
                          n_cells_total: int,
                          n_green: int) -> np.ndarray:
    """Exact weighted-subset PMF for one uniform fixed-size marked set.

    The implementation counts subsets of occupied cells with each possible
    weight and analytically fills the remaining selected positions with empty
    cells.  Python integer combination counts avoid numerical overflow in the
    dynamic program; only the final normalized PMF is floating point.
    """

    n = _integer_vector(multiplicities, "multiplicities")
    M = int(n_cells_total)
    G = _green_count(M, n_green)
    if n.size > M:
        raise ValueError("multiplicities describes more cells than the space")
    positive = [int(x) for x in n if x > 0]
    occupied = len(positive)
    if occupied > M:
        raise ValueError("more occupied cells than cells in the space")

    total = sum(positive)
    max_occupied_picks = min(G, occupied)
    ways: list[dict[int, int]] = [dict() for _ in range(max_occupied_picks + 1)]
    ways[0][0] = 1
    processed = 0
    for weight in positive:
        upper = min(max_occupied_picks, processed + 1)
        for picked in range(upper, 0, -1):
            for subtotal, count in list(ways[picked - 1].items()):
                score = subtotal + weight
                ways[picked][score] = ways[picked].get(score, 0) + count
        processed += 1

    denominator = comb(M, G)
    empty = M - occupied
    pmf = np.zeros(total + 1, dtype=float)
    for picked, scores in enumerate(ways):
        empty_needed = G - picked
        if not 0 <= empty_needed <= empty:
            continue
        completions = comb(empty, empty_needed)
        for score, count in scores.items():
            pmf[score] += (count * completions) / denominator

    # Keep tails stable after repeated convolutions in the row-stratified law.
    mass = float(pmf.sum())
    if not np.isclose(mass, 1.0, atol=1e-12):
        raise RuntimeError(f"fixed-size PMF lost probability mass: {mass}")
    pmf /= mass
    return pmf


# ---------------------------------------------------------------------------
# Previous-state-stratified SemStamp masks
# ---------------------------------------------------------------------------


def semstamp_transition_counts(cell_path: Iterable[int],
                               n_cells: int) -> np.ndarray:
    """Return ``n[a,b]`` for adjacent transitions in a semantic cell path."""

    path = _integer_vector(cell_path, "cell_path")
    K = int(n_cells)
    if K <= 0:
        raise ValueError("n_cells must be positive")
    if path.size and np.any(path >= K):
        raise ValueError("cell_path contains an index outside the cell space")
    counts = np.zeros((K, K), dtype=np.int64)
    if path.size >= 2:
        np.add.at(counts, (path[:-1], path[1:]), 1)
    return counts


def semstamp_row_moments(transition_counts, n_valid: int) -> NullMoments:
    """Moments for independently keyed, fixed-size SemStamp transition rows.

    For each predecessor cell ``a``, a uniformly random subset of exactly
    ``n_valid`` of the ``K`` next cells is accepted.  Different predecessor
    rows are assumed independent.
    """

    n = _integer_matrix(transition_counts, "transition_counts")
    K = int(n.shape[0])
    G = _green_count(K, n_valid)
    p = G / K
    total = int(n.sum())
    if K == 1:
        variance = 0.0
    else:
        row_totals = n.sum(axis=1).astype(float)
        sum_squares = np.square(n.astype(float)).sum(axis=1)
        variance = p * (1.0 - p) * float(
            np.sum(K * sum_squares - np.square(row_totals))
        ) / (K - 1)
    return _make_moments(p * total, variance, total, p)


def semstamp_row_pmf(transition_counts, n_valid: int) -> np.ndarray:
    """Exact PMF for independent fixed-size masks within predecessor rows."""

    n = _integer_matrix(transition_counts, "transition_counts")
    K = int(n.shape[0])
    G = _green_count(K, n_valid)
    pmf = np.array([1.0])
    for row in n:
        if row.sum() == 0:
            continue
        row_pmf = global_fixed_size_pmf(row, K, G)
        pmf = np.convolve(pmf, row_pmf)
    mass = float(pmf.sum())
    if mass == 0.0:
        raise RuntimeError("row-stratified PMF has zero mass")
    pmf /= mass
    return pmf


# ---------------------------------------------------------------------------
# Tail probabilities and the published sentence z approximation
# ---------------------------------------------------------------------------


def upper_tail_pvalue(observed_score: float, pmf: Iterable[float]) -> float:
    """Return ``P(S >= observed_score)`` for an integer-supported PMF."""

    distribution = np.asarray(pmf, dtype=float)
    if distribution.ndim != 1 or distribution.size == 0:
        raise ValueError("pmf must be a non-empty one-dimensional array")
    if np.any(distribution < -1e-15) or not np.all(np.isfinite(distribution)):
        raise ValueError("pmf must contain finite non-negative probabilities")
    threshold = int(np.ceil(float(observed_score) - 1e-12))
    if threshold <= 0:
        return 1.0
    if threshold >= distribution.size:
        return 0.0
    return float(distribution[threshold:].sum())


def minimum_upper_tail_pvalue(pmf: Iterable[float]) -> float:
    """Smallest non-zero deterministic upper-tail p-value of a PMF."""

    distribution = np.asarray(pmf, dtype=float)
    if distribution.ndim != 1 or distribution.size == 0:
        raise ValueError("pmf must be a non-empty one-dimensional array")
    support = np.flatnonzero(distribution > 0.0)
    if support.size == 0:
        raise ValueError("pmf has no positive mass")
    return float(distribution[int(support[-1]):].sum())


def semstamp_row_pvalue(observed_score: float, transition_counts,
                        n_valid: int) -> float:
    """Exact one-sided p-value under the ideal row-stratified SemStamp null."""

    return upper_tail_pvalue(
        observed_score,
        semstamp_row_pmf(transition_counts, n_valid),
    )


def naive_z_score(observed_score: float, n_trials: int,
                  probability: float) -> float:
    """Sentence-binomial z-score used by the published SemStamp detector."""

    p = _probability(probability)
    n_trials = int(n_trials)
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    variance = n_trials * p * (1.0 - p)
    if variance <= 0.0:
        raise ValueError("the z approximation requires probability in (0, 1)")
    return float((float(observed_score) - p * n_trials) / np.sqrt(variance))


def naive_z_pvalue(observed_score: float, n_trials: int,
                   probability: float) -> float:
    """One-sided normal-tail p-value from the sentence-binomial denominator."""

    return float(norm.sf(naive_z_score(observed_score, n_trials, probability)))


# ---------------------------------------------------------------------------
# Ideal-key simulation, used to audit formulas and experiment code
# ---------------------------------------------------------------------------


def simulate_semstamp_row_scores(
    transition_counts,
    n_valid: int,
    n_keys: int = 100_000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw scores from independent uniform fixed-size masks in every row."""

    n = _integer_matrix(transition_counts, "transition_counts")
    K = int(n.shape[0])
    G = _green_count(K, n_valid)
    n_keys = int(n_keys)
    if n_keys <= 0:
        raise ValueError("n_keys must be positive")
    rng = np.random.default_rng() if rng is None else rng
    scores = np.zeros(n_keys, dtype=np.int64)
    if G == 0:
        return scores
    if G == K:
        scores.fill(int(n.sum()))
        return scores

    for row in n:
        if row.sum() == 0:
            continue
        priorities = rng.random((n_keys, K))
        selected = np.argpartition(priorities, G - 1, axis=1)[:, :G]
        scores += row[selected].sum(axis=1)
    return scores
