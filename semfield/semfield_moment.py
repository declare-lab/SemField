"""Small, training-free second-moment channel for SemField.

The first SemField channel uses the document sum ``sum(phi_t)``.  This module
adds a key-independent, order-invariant centered scatter vector
``sum((phi_t-mean(phi))**2)`` and scores it with an independent keyed Gaussian
field.  It is deliberately diagonal rather than a dense covariance matrix so
that it costs O(d) memory and time and can be used at the same scale as the
existing feature rows.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

from semfield.semfield import FieldStatistic, field_statistic

_NORMAL = NormalDist()


def centered_scatter(features: np.ndarray) -> np.ndarray:
    """Return the diagonal centered scatter of sentence feature rows."""

    phi = np.asarray(features, dtype=np.float64)
    if phi.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix")
    if phi.shape[1] <= 0 or np.any(~np.isfinite(phi)):
        raise ValueError("features must be finite and non-empty")
    if phi.shape[0] < 2:
        return np.zeros(phi.shape[1], dtype=np.float64)
    centered = phi - phi.mean(axis=0, keepdims=True)
    return np.square(centered).sum(axis=0)


def prospective_scatter(
    mean: np.ndarray, scatter: np.ndarray, count: int, row: np.ndarray
) -> np.ndarray:
    """Update centered scatter without retaining all previous rows."""

    value = np.asarray(row, dtype=np.float64).ravel()
    old_mean = np.asarray(mean, dtype=np.float64).ravel()
    old_scatter = np.asarray(scatter, dtype=np.float64).ravel()
    if old_mean.size != value.size or old_scatter.size != value.size:
        raise ValueError("moment state and row widths differ")
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return np.zeros_like(value)
    delta = value - old_mean
    # C_{n+1}=C_n+n/(n+1)(x-mean_n)^2.
    return old_scatter + (float(count) / float(count + 1)) * np.square(delta)


def fused_components(
    features: np.ndarray,
    first_weights: np.ndarray,
    moment_weights: np.ndarray,
    moment_lambda: float,
) -> tuple[float, float, float]:
    """Return (first-channel z, moment-channel z, fused z)."""

    phi = np.asarray(features, dtype=np.float64)
    if phi.ndim != 2 or phi.shape[1] <= 0 or np.any(~np.isfinite(phi)):
        raise ValueError("features must be a finite two-dimensional matrix")
    w1 = np.asarray(first_weights, dtype=np.float64).ravel()
    w2 = np.asarray(moment_weights, dtype=np.float64).ravel()
    if w1.size != phi.shape[1] or w2.size != phi.shape[1]:
        raise ValueError("moment weights do not match feature width")
    aggregate = phi.sum(axis=0)
    aggregate_norm = float(np.linalg.norm(aggregate))
    z1 = 0.0 if aggregate_norm <= 1e-15 else float(np.dot(w1, aggregate) / aggregate_norm)
    scatter = centered_scatter(phi)
    scatter_norm = float(np.linalg.norm(scatter))
    z2 = 0.0 if scatter_norm <= 1e-15 else float(np.dot(w2, scatter) / scatter_norm)
    lam = float(moment_lambda)
    if not math.isfinite(lam) or lam < 0.0:
        raise ValueError("moment_lambda must be finite and non-negative")
    fused = (z1 + lam * z2) / math.sqrt(1.0 + lam * lam)
    return z1, z2, float(fused)


def moment_field_statistic(
    features: np.ndarray,
    first_weights: np.ndarray,
    moment_weights: np.ndarray,
    *,
    moment_lambda: float,
    two_sided: bool = False,
) -> FieldStatistic:
    """Represent the fused z score using the existing detector schema."""

    phi = np.asarray(features, dtype=np.float64)
    if phi.ndim != 2 or phi.shape[0] == 0:
        return FieldStatistic(0.0, 0.0, 0.0, 1.0, 0, None, None)
    z1, z2, fused = fused_components(
        phi, first_weights, moment_weights, moment_lambda
    )
    if two_sided:
        score = abs(fused)
        pvalue = 2.0 * _NORMAL.cdf(-score)
    else:
        score = fused
        pvalue = _NORMAL.cdf(-score)
    first = field_statistic(phi, first_weights)
    # Keep the established fields for backwards-compatible reports.  The
    # payload records both component scores and the fusion parameters.
    return FieldStatistic(
        score_sum=float(score),
        null_variance=1.0,
        z_score=float(score),
        analytic_pvalue=float(pvalue),
        n_sentences=int(phi.shape[0]),
        mean_pair_kernel=first.mean_pair_kernel,
        design_effect=first.design_effect,
    )


def prospective_fused_scores(
    aggregate: np.ndarray,
    mean: np.ndarray,
    scatter: np.ndarray,
    count: int,
    candidates: np.ndarray,
    first_weights: np.ndarray,
    moment_weights: np.ndarray,
    moment_lambda: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score each candidate using the state after appending that candidate."""

    values = np.asarray(candidates, dtype=np.float64)
    current = np.asarray(aggregate, dtype=np.float64).ravel()
    mean_value = np.asarray(mean, dtype=np.float64).ravel()
    scatter_value = np.asarray(scatter, dtype=np.float64).ravel()
    w1 = np.asarray(first_weights, dtype=np.float64).ravel()
    w2 = np.asarray(moment_weights, dtype=np.float64).ravel()
    next_aggregate = values + current
    norms = np.linalg.norm(next_aggregate, axis=1)
    z1 = np.divide(
        next_aggregate @ w1, norms,
        out=np.zeros(values.shape[0], dtype=np.float64),
        where=norms > 1e-15,
    )
    if count == 0:
        z2 = np.zeros(values.shape[0], dtype=np.float64)
    else:
        delta = values - mean_value
        next_scatter = scatter_value[None, :] + (
            float(count) / float(count + 1)
        ) * np.square(delta)
        scatter_norms = np.linalg.norm(next_scatter, axis=1)
        z2 = np.divide(
            next_scatter @ w2, scatter_norms,
            out=np.zeros(values.shape[0], dtype=np.float64),
            where=scatter_norms > 1e-15,
        )
    lam = float(moment_lambda)
    fused = (z1 + lam * z2) / math.sqrt(1.0 + lam * lam)
    return z1, z2, fused
