"""Public feature maps used by SemField.

The original linear map is stateless: every sentence is normalized on its
own.  The innovation maps below are deliberately stateful.  They remove the
part of a sentence embedding already explained by the preceding document
context before applying the keyed field.  The state is only document text,
never the key, so the detector can reproduce it exactly.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

__all__ = [
    "LinearSphericalFeatures",
    "PolynomialRandomFeatures",
    "RBFRandomFourierFeatures",
    "ProjectedLinearSphericalFeatures",
    "ContentRoutedLinearSphericalFeatures",
    "ResidualLinearSphericalFeatures",
    "SpanResidualLinearSphericalFeatures",
    "WhitenedLinearSphericalFeatures",
    "ConeEqualizedLinearSphericalFeatures",
    "MeanAxisShrunkLinearSphericalFeatures",
    "build_feature_map",
    "feature_map_from_metadata",
]


def _unit_rows(embeddings, input_dim: int) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != input_dim:
        raise ValueError(f"embeddings must have shape (n, {input_dim})")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError("embeddings must be finite and non-zero")
    return values / norms


class LinearSphericalFeatures:
    """Exact normalized feature map for the cosine/linear kernel."""

    kind = "linear-spherical-v1"

    def __init__(self, input_dim: int):
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.input_dim = int(input_dim)
        self.n_features = int(input_dim)

    def transform(self, embeddings) -> np.ndarray:
        return _unit_rows(embeddings, self.input_dim)

    def metadata(self) -> dict[str, int | bool | str]:
        return {
            "type": self.kind,
            "kernel": "linear",
            "input_dim": self.input_dim,
            "n_features": self.n_features,
            "row_normalized": True,
            "exact": True,
        }

    @classmethod
    def from_metadata(cls, metadata: dict) -> "LinearSphericalFeatures":
        if metadata.get("type") != cls.kind:
            raise ValueError(f"unsupported feature type {metadata.get('type')!r}")
        instance = cls(int(metadata["input_dim"]))
        if int(metadata.get("n_features", instance.n_features)) != instance.n_features:
            raise ValueError("linear feature dimension must equal input dimension")
        return instance


class PolynomialRandomFeatures:
    """Fixed-width random Maclaurin features for ``(x^T y)^degree``.

    Each output coordinate is a product of independent Rademacher
    projections.  Consequently the unnormalized feature inner product is an
    unbiased estimate of the degree-m polynomial kernel.  Rows are
    normalized because SemField operates on spherical features.
    """

    kind = "polynomial-random-features-v1"

    def __init__(self, input_dim: int, n_features: int, degree: int, seed: int = 0):
        if input_dim <= 0 or n_features <= 0:
            raise ValueError("input_dim and n_features must be positive")
        if degree not in {2, 3}:
            raise ValueError("polynomial degree must be 2 or 3")
        self.input_dim = int(input_dim)
        self.n_features = int(n_features)
        self.degree = int(degree)
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        self.weights = rng.choice(
            np.array([-1.0, 1.0], dtype=np.float64),
            size=(self.degree, self.n_features, self.input_dim),
        )

    def transform(self, embeddings) -> np.ndarray:
        values = _unit_rows(embeddings, self.input_dim)
        features = np.ones((values.shape[0], self.n_features), dtype=np.float64)
        for weights in self.weights:
            features *= values @ weights.T
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-12):
            raise ValueError("polynomial random features must be finite and non-zero")
        return features / norms

    def metadata(self) -> dict[str, int | bool | str]:
        return {
            "type": self.kind,
            "kernel": f"polynomial-degree-{self.degree}",
            "input_dim": self.input_dim,
            "n_features": self.n_features,
            "degree": self.degree,
            "seed": self.seed,
            "row_normalized": True,
            "exact": False,
            "approximation": "random-maclaurin",
        }

    @classmethod
    def from_metadata(cls, metadata: dict) -> "PolynomialRandomFeatures":
        if metadata.get("type") != cls.kind:
            raise ValueError(f"unsupported feature type {metadata.get('type')!r}")
        instance = cls(
            int(metadata["input_dim"]), int(metadata["n_features"]),
            int(metadata["degree"]), int(metadata.get("seed", 0)),
        )
        return instance


class RBFRandomFourierFeatures:
    """Fixed-width random Fourier features for an RBF kernel."""

    kind = "rbf-random-fourier-features-v1"

    def __init__(self, input_dim: int, n_features: int,
                 length_scale: float = 0.5, seed: int = 0):
        if input_dim <= 0 or n_features <= 0:
            raise ValueError("input_dim and n_features must be positive")
        if not math.isfinite(float(length_scale)) or float(length_scale) <= 0:
            raise ValueError("RBF length_scale must be finite and positive")
        self.input_dim = int(input_dim)
        self.n_features = int(n_features)
        self.length_scale = float(length_scale)
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        self.omega = rng.normal(
            0.0, 1.0 / self.length_scale,
            size=(self.n_features, self.input_dim),
        )
        self.phase = rng.uniform(0.0, 2.0 * math.pi, size=self.n_features)

    def transform(self, embeddings) -> np.ndarray:
        values = _unit_rows(embeddings, self.input_dim)
        features = math.sqrt(2.0 / self.n_features) * np.cos(
            values @ self.omega.T + self.phase[None, :]
        )
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-12):
            raise ValueError("RBF random features must be finite and non-zero")
        return features / norms

    def metadata(self) -> dict[str, int | bool | str | float]:
        return {
            "type": self.kind,
            "kernel": "rbf",
            "input_dim": self.input_dim,
            "n_features": self.n_features,
            "length_scale": self.length_scale,
            "seed": self.seed,
            "row_normalized": True,
            "exact": False,
            "approximation": "random-fourier",
        }

    @classmethod
    def from_metadata(cls, metadata: dict) -> "RBFRandomFourierFeatures":
        if metadata.get("type") != cls.kind:
            raise ValueError(f"unsupported feature type {metadata.get('type')!r}")
        return cls(
            int(metadata["input_dim"]), int(metadata["n_features"]),
            float(metadata.get("length_scale", 0.5)),
            int(metadata.get("seed", 0)),
        )


class ProjectedLinearSphericalFeatures:
    """Public, dimension-reducing linear map for candidate PCA audits.

    ``basis`` is a frozen ``(input_dim, output_dim)`` orthonormal matrix.  The
    projection is independent of the secret key and is therefore safe to use
    identically during generation and detection.  Reducing ``n_features`` is
    important: a rank-k matrix with a 768-dimensional keyed field would not
    realize the k-dimensional candidate-variance calculation.
    """

    kind = "projected-linear-spherical-v1"

    def __init__(
        self, input_dim: int, projection_path: str, projection_key: str = "basis_k64"
    ):
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        path = str(projection_path)
        with np.load(path, allow_pickle=False) as payload:
            if projection_key not in payload.files:
                raise ValueError(
                    f"projection file {path!r} has no array {projection_key!r}"
                )
            basis = np.asarray(payload[projection_key], dtype=np.float64)
        if basis.ndim != 2 or basis.shape[0] != int(input_dim) or basis.shape[1] <= 0:
            raise ValueError(
                "projection basis must have shape "
                f"({input_dim}, k), found {basis.shape}"
            )
        if np.any(~np.isfinite(basis)):
            raise ValueError("projection basis must be finite")
        gram = basis.T @ basis
        if not np.allclose(gram, np.eye(basis.shape[1]), atol=2e-4, rtol=2e-4):
            raise ValueError("projection basis must be approximately orthonormal")
        self.input_dim = int(input_dim)
        self.n_features = int(basis.shape[1])
        self.projection_path = path
        self.projection_key = str(projection_key)
        self.basis = np.ascontiguousarray(basis)
        self.projection_sha256 = hashlib.sha256(
            open(path, "rb").read()
        ).hexdigest()

    def transform(self, embeddings) -> np.ndarray:
        values = _unit_rows(embeddings, self.input_dim)
        projected = values @ self.basis
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
            raise ValueError("projected embeddings must be finite and non-zero")
        return projected / norms

    def metadata(self) -> dict[str, int | bool | str]:
        return {
            "type": self.kind,
            "kernel": "linear",
            "input_dim": self.input_dim,
            "n_features": self.n_features,
            "row_normalized": True,
            "exact": True,
            "projection_path": self.projection_path,
            "projection_key": self.projection_key,
            "projection_sha256": self.projection_sha256,
            "projection": "frozen-clean-candidate-pca",
        }

    @classmethod
    def from_metadata(cls, metadata: dict) -> "ProjectedLinearSphericalFeatures":
        if metadata.get("type") != cls.kind:
            raise ValueError(f"unsupported feature type {metadata.get('type')!r}")
        instance = cls(
            int(metadata["input_dim"]),
            str(metadata["projection_path"]),
            str(metadata.get("projection_key", "basis_k64")),
        )
        expected = metadata.get("projection_sha256")
        if expected is not None and expected != instance.projection_sha256:
            raise ValueError(
                "projection basis does not match the generation fingerprint; "
                "detection must use the identical public basis"
            )
        if int(metadata.get("n_features", instance.n_features)) != instance.n_features:
            raise ValueError("projected feature dimension does not match metadata")
        return instance


class ContentRoutedLinearSphericalFeatures:
    """A smooth, content-routed continuous SemField feature map.

    A public, frozen bank of normalized anchor vectors ``a_j`` defines a
    smooth route ``r(x)`` for each normalized sentence embedding ``x``.  The
    feature is the tensor product ``r(x) ⊗ x``.  Consequently, its exact
    kernel is

        <phi(x), phi(y)> = <x, y> <r(x), r(y)>.

    The first factor retains the ordinary semantic geometry.  The second
    factor makes distinct semantic regions use increasingly independent
    components of the same continuous Gaussian field.  There are no hard
    regions, keyed partitions, bits, or learned weights: anchors and the
    routing scale are public and fixed before watermark generation.
    """

    kind = "content-routed-linear-spherical-v1"

    def __init__(self, input_dim: int, router_path: str):
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        path = str(router_path)
        payload = np.load(path)
        anchors = np.asarray(payload["anchors"], dtype=np.float64)
        if anchors.ndim != 2 or anchors.shape[1] != int(input_dim):
            raise ValueError(
                f"router anchors must have shape (r, {input_dim}), "
                f"found {anchors.shape}"
            )
        if anchors.shape[0] < 2:
            raise ValueError("content router needs at least two anchors")
        anchors = _unit_rows(anchors, int(input_dim))
        routing_rule = (
            str(np.asarray(payload["routing_rule"]).reshape(-1)[0])
            if "routing_rule" in payload else "cosine-distance-softmax-v1"
        )
        if routing_rule not in {
            "cosine-distance-softmax-v1", "row-zscore-softmax-v1",
        }:
            raise ValueError(f"unsupported content routing rule {routing_rule!r}")
        temperature = None
        if routing_rule == "cosine-distance-softmax-v1":
            if "temperature" not in payload:
                raise ValueError("content router artifact needs a temperature")
            temperature = float(np.asarray(payload["temperature"]).reshape(-1)[0])
            if not math.isfinite(temperature) or temperature <= 0.0:
                raise ValueError("content router temperature must be finite and positive")

        self.input_dim = int(input_dim)
        self.anchors = np.ascontiguousarray(anchors)
        self.routes = int(anchors.shape[0])
        self.temperature = temperature
        self.routing_rule = routing_rule
        self.n_features = self.input_dim * self.routes
        self.router_path = path
        with open(path, "rb") as handle:
            self.router_sha256 = hashlib.sha256(handle.read()).hexdigest()
        self.router_source = (
            str(np.asarray(payload["source"]).reshape(-1)[0])
            if "source" in payload else None
        )

    def route(self, embeddings) -> np.ndarray:
        """Return L2-normalized, smooth public routing weights."""

        values = _unit_rows(embeddings, self.input_dim)
        similarities = values @ self.anchors.T
        if self.routing_rule == "row-zscore-softmax-v1":
            # Relative anchor affinity is dimensionless.  This avoids a
            # tunable global bandwidth and prevents an anisotropic encoder
            # from making every route almost uniform.
            centered = similarities - similarities.mean(axis=1, keepdims=True)
            scale = np.sqrt(np.mean(np.square(centered), axis=1, keepdims=True))
            logits = centered / np.maximum(scale, 1e-12)
        else:
            # Cosine distance is 1 - x.T a.  The subtraction of the row
            # maximum is only numerical stabilization; it leaves softmax
            # unchanged.
            logits = (similarities - 1.0) / float(self.temperature)
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        weights /= weights.sum(axis=1, keepdims=True)
        norms = np.linalg.norm(weights, axis=1, keepdims=True)
        if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
            raise ValueError("content routing weights must be finite and non-zero")
        return weights / norms

    def transform(self, embeddings) -> np.ndarray:
        values = _unit_rows(embeddings, self.input_dim)
        route = self.route(values)
        # Both factors have unit norm, so every output row has unit norm.  The
        # explicit reshape is an exact finite feature map, not an RFF proxy.
        return np.ascontiguousarray(
            (route[:, :, None] * values[:, None, :]).reshape(
                values.shape[0], self.n_features
            )
        )

    def metadata(self) -> dict[str, int | bool | str | float | None]:
        return {
            "type": self.kind,
            "kernel": "content-routed-linear",
            "input_dim": self.input_dim,
            "n_features": self.n_features,
            "row_normalized": True,
            "exact": True,
            "routes": self.routes,
            "router_temperature": self.temperature,
            "routing_rule": self.routing_rule,
            "router_path": self.router_path,
            "router_sha256": self.router_sha256,
            "router_source": self.router_source,
        }

    @classmethod
    def from_metadata(cls, metadata: dict) -> "ContentRoutedLinearSphericalFeatures":
        if metadata.get("type") != cls.kind:
            raise ValueError(f"unsupported feature type {metadata.get('type')!r}")
        path = metadata.get("router_path")
        if not path:
            raise ValueError("content-routed metadata needs router_path")
        instance = cls(int(metadata["input_dim"]), str(path))
        expected = metadata.get("router_sha256")
        if expected is not None and expected != instance.router_sha256:
            raise ValueError(
                "content router does not match the generation fingerprint; "
                "detection must use the identical public router"
            )
        if int(metadata.get("n_features", instance.n_features)) != instance.n_features:
            raise ValueError("content-routed feature dimension does not match metadata")
        return instance


class ResidualLinearSphericalFeatures:
    """Sequential semantic-innovation feature map.

    For sentence embedding ``x_t`` and the preceding normalized embeddings,
    let ``u`` be their normalized sum.  The innovation is the component of
    ``x_t`` orthogonal to ``u``.  With ``blend_weight`` set, the output keeps
    a second, absolute channel so that a rewrite which changes discourse
    transitions cannot erase all of the signal.  Both operations are
    deterministic and key-independent.
    """

    residual_kind = "residual-linear-spherical-v1"
    hybrid_kind = "hybrid-residual-linear-spherical-v1"
    stateful = True

    def __init__(self, input_dim: int, *, blend_weight: float | None = None):
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if blend_weight is not None and not 0.0 < float(blend_weight) < 1.0:
            raise ValueError("blend_weight must lie strictly between zero and one")
        self.input_dim = int(input_dim)
        self.blend_weight = (
            None if blend_weight is None else float(blend_weight)
        )
        self.n_features = (
            self.input_dim
            if self.blend_weight is None
            else 2 * self.input_dim
        )
        self.kind = (
            self.residual_kind
            if self.blend_weight is None
            else self.hybrid_kind
        )

    def _innovation(
        self, rows: np.ndarray, history: np.ndarray | None
    ) -> np.ndarray:
        values = _unit_rows(rows, self.input_dim)
        if history is None:
            history_values = np.zeros((0, self.input_dim), dtype=np.float64)
        else:
            history_values = np.asarray(history, dtype=np.float64)
            if history_values.size == 0:
                history_values = np.zeros((0, self.input_dim), dtype=np.float64)
            if history_values.ndim != 2 or history_values.shape[1] != self.input_dim:
                raise ValueError(
                    f"history must have shape (n, {self.input_dim})"
                )
        if history_values.shape[0] == 0:
            return values

        direction = history_values.sum(axis=0)
        direction_norm = float(np.linalg.norm(direction))
        if not math.isfinite(direction_norm) or direction_norm <= 1e-15:
            return values
        direction /= direction_norm
        residual = values - (values @ direction[:, None]) * direction[None, :]
        residual_norms = np.linalg.norm(residual, axis=1, keepdims=True)
        # A sentence parallel to the running topic has no stable residual.
        # Keeping its normalized absolute embedding is deterministic and
        # avoids introducing NaNs into a generation or detector sidecar.
        fallback = (~np.isfinite(residual_norms[:, 0])) | (
            residual_norms[:, 0] <= 1e-12
        )
        safe_norms = np.where(fallback[:, None], 1.0, residual_norms)
        residual = residual / safe_norms
        if np.any(fallback):
            residual[fallback] = values[fallback]
        return residual

    def transform_candidates(self, embeddings, history=None) -> np.ndarray:
        """Transform candidates using already selected raw embeddings."""

        values = _unit_rows(embeddings, self.input_dim)
        innovation = self._innovation(values, history)
        if self.blend_weight is None:
            return innovation
        weight = float(self.blend_weight)
        return np.concatenate(
            (
                math.sqrt(1.0 - weight) * values,
                math.sqrt(weight) * innovation,
            ),
            axis=1,
        )

    def transform_sequence(self, embeddings) -> np.ndarray:
        """Transform one document in order, using prior rows as context."""

        values = _unit_rows(embeddings, self.input_dim)
        output: list[np.ndarray] = []
        history: list[np.ndarray] = []
        for row in values:
            output.append(self.transform_candidates(row[None, :], history)[0])
            history.append(row.copy())
        if not output:
            return np.zeros((0, self.n_features), dtype=np.float64)
        return np.asarray(output, dtype=np.float64)

    def transform(self, embeddings) -> np.ndarray:
        """Transform an embedding sequence as one document.

        Stateful maps should use ``transform_candidates`` during generation;
        this method is provided for detector and generic feature-map callers.
        """

        return self.transform_sequence(embeddings)

    def metadata(self) -> dict[str, int | bool | str | float | None]:
        return {
            "type": self.kind,
            "kernel": "linear",
            "input_dim": self.input_dim,
            "n_features": self.n_features,
            "row_normalized": True,
            "stateful": True,
            "innovation": "running-sum-orthogonal-residual-v1",
            "blend_weight": self.blend_weight,
            "exact": True,
        }

    @classmethod
    def from_metadata(cls, metadata: dict) -> "ResidualLinearSphericalFeatures":
        kind = metadata.get("type")
        if kind not in {cls.residual_kind, cls.hybrid_kind}:
            raise ValueError(f"unsupported feature type {kind!r}")
        blend = metadata.get("blend_weight")
        if kind == cls.residual_kind:
            blend = None
        elif blend is None:
            raise ValueError("hybrid residual metadata needs blend_weight")
        instance = cls(int(metadata["input_dim"]), blend_weight=blend)
        if int(metadata.get("n_features", instance.n_features)) != instance.n_features:
            raise ValueError("residual feature dimension does not match metadata")
        return instance


class SpanResidualLinearSphericalFeatures:
    """Residual feature map that removes the complete prior sentence span.

    Unlike the older running-sum residual, this projects each candidate out of
    the full row span of the already emitted normalized embeddings.  A small
    residual falls back to the original normalized embedding, keeping the map
    finite and making the edge case explicit to generation and detection.
    """

    kind = "span-residual-linear-spherical-v1"
    stateful = True

    def __init__(self, input_dim: int):
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.input_dim = int(input_dim)
        self.n_features = int(input_dim)

    @staticmethod
    def _basis(history: np.ndarray, input_dim: int) -> np.ndarray:
        if history.size == 0:
            return np.empty((input_dim, 0), dtype=np.float64)
        values = _unit_rows(history, input_dim)
        _u, singular, vt = np.linalg.svd(values, full_matrices=False)
        if singular.size == 0:
            return np.empty((input_dim, 0), dtype=np.float64)
        cutoff = float(singular[0]) * max(values.shape) * np.finfo(np.float64).eps
        rank = int(np.count_nonzero(singular > cutoff))
        return vt[:rank].T

    def transform_candidates(self, embeddings, history=None) -> np.ndarray:
        values = _unit_rows(embeddings, self.input_dim)
        if history is None:
            history_values = np.zeros((0, self.input_dim), dtype=np.float64)
        else:
            history_values = np.asarray(history, dtype=np.float64)
            if history_values.size == 0:
                history_values = np.zeros((0, self.input_dim), dtype=np.float64)
            if history_values.ndim != 2 or history_values.shape[1] != self.input_dim:
                raise ValueError(f"history must have shape (n, {self.input_dim})")
        basis = self._basis(history_values, self.input_dim)
        if basis.shape[1] == 0:
            return values
        residual = values - (values @ basis) @ basis.T
        norms = np.linalg.norm(residual, axis=1)
        fallback = (~np.isfinite(norms)) | (norms <= 1e-12)
        safe = np.where(fallback, 1.0, norms)[:, None]
        output = residual / safe
        if np.any(fallback):
            output[fallback] = values[fallback]
        return output

    def transform_sequence(self, embeddings) -> np.ndarray:
        values = _unit_rows(embeddings, self.input_dim)
        output = []
        history = []
        for row in values:
            output.append(self.transform_candidates(row[None, :], history)[0])
            history.append(row.copy())
        if not output:
            return np.zeros((0, self.n_features), dtype=np.float64)
        return np.asarray(output, dtype=np.float64)

    def transform(self, embeddings) -> np.ndarray:
        return self.transform_sequence(embeddings)

    def metadata(self) -> dict[str, int | bool | str]:
        return {
            "type": self.kind,
            "kernel": "linear",
            "input_dim": self.input_dim,
            "n_features": self.n_features,
            "row_normalized": True,
            "stateful": True,
            "innovation": "full-prior-row-span-residual-v1",
            "fallback": "normalized-absolute-embedding-if-residual-is-degenerate",
            "exact": True,
        }

    @classmethod
    def from_metadata(cls, metadata: dict) -> "SpanResidualLinearSphericalFeatures":
        if metadata.get("type") != cls.kind:
            raise ValueError(f"unsupported feature type {metadata.get('type')!r}")
        instance = cls(int(metadata["input_dim"]))
        if int(metadata.get("n_features", instance.n_features)) != instance.n_features:
            raise ValueError("span residual feature dimension does not match metadata")
        return instance


class WhitenedLinearSphericalFeatures:
    """Linear map applied in whitened embedding coordinates.

    A keyed Gaussian direction spends its budget uniformly over every
    embedding dimension, but sentence candidates only differ in a small
    fraction of them, so most of the budget lands where candidates are
    identical.  Whitening by the corpus second moment ``M = E[phi phi^T]``
    equalises the directions before the field is applied, which widens the
    candidate spread a keyed direction can see.

    ``M`` is estimated once from unwatermarked text, never depends on a key,
    and is fixed before any watermarked document is scored, so it is a public
    feature map exactly like the residual map above.  The conditional null and
    the control-key calibration are therefore unchanged.
    """

    kind = "whitened-linear-spherical-v1"

    def __init__(self, input_dim: int, whitener_path: str):
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        payload = np.load(whitener_path)
        whitener = np.asarray(payload["whitener"], dtype=np.float64)
        if whitener.shape != (int(input_dim), int(input_dim)):
            raise ValueError(
                f"whitener must have shape ({input_dim}, {input_dim}), "
                f"found {whitener.shape}"
            )
        if np.any(~np.isfinite(whitener)):
            raise ValueError("whitener must be finite")
        self.input_dim = int(input_dim)
        self.n_features = int(input_dim)
        self.whitener_path = str(whitener_path)
        self.whitener = np.ascontiguousarray(whitener)
        self.shrinkage = float(payload["shrinkage"]) if "shrinkage" in payload else None
        self.whitener_sha256 = hashlib.sha256(
            open(whitener_path, "rb").read()
        ).hexdigest()

    def transform(self, embeddings) -> np.ndarray:
        values = _unit_rows(embeddings, self.input_dim)
        whitened = values @ self.whitener.T
        norms = np.linalg.norm(whitened, axis=1, keepdims=True)
        if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
            raise ValueError("whitened embeddings must be finite and non-zero")
        return whitened / norms

    def metadata(self) -> dict[str, int | bool | str | float | None]:
        return {
            "type": self.kind,
            "kernel": "linear",
            "input_dim": self.input_dim,
            "n_features": self.n_features,
            "row_normalized": True,
            "exact": True,
            "whitener_path": self.whitener_path,
            "whitener_sha256": self.whitener_sha256,
            "whitener_shrinkage": self.shrinkage,
        }

    @classmethod
    def from_metadata(cls, metadata: dict) -> "WhitenedLinearSphericalFeatures":
        if metadata.get("type") != cls.kind:
            raise ValueError(f"unsupported feature type {metadata.get('type')!r}")
        instance = cls(int(metadata["input_dim"]), str(metadata["whitener_path"]))
        expected = metadata.get("whitener_sha256")
        if expected is not None and expected != instance.whitener_sha256:
            raise ValueError(
                "whitening matrix does not match the generation fingerprint; "
                "detection must use the identical public matrix"
            )
        return instance


class ConeEqualizedLinearSphericalFeatures:
    """Public one-axis cone equalizer for normalized sentence embeddings.

    The transform is fitted once from an independent public corpus.  If ``u``
    is the leading eigenvector of ``E[x x^T]`` and ``s`` is the ratio that
    equalizes that axis with the average orthogonal variance, the stored map
    is ``P = I + (s-1) u u^T``.  It is key-independent and is applied before
    the ordinary SemField random-feature map, identically at generation and
    detection.  Rows are renormalized after applying ``P`` because SemField
    operates on a sphere.
    """

    kind = "cone-equalized-linear-spherical-v1"

    def __init__(self, input_dim: int, equalizer_path: str):
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        with np.load(equalizer_path, allow_pickle=False) as payload:
            if "transform" not in payload:
                raise ValueError("cone equalizer must contain a 'transform' array")
            transform = np.asarray(payload["transform"], dtype=np.float64)
            self.lambda_1 = float(payload["lambda_1"]) if "lambda_1" in payload else None
            self.lambda_perp = float(payload["lambda_perp"]) if "lambda_perp" in payload else None
            self.scale = float(payload["scale"]) if "scale" in payload else None
        if transform.shape != (int(input_dim), int(input_dim)):
            raise ValueError(
                f"cone equalizer must have shape ({input_dim}, {input_dim}), "
                f"found {transform.shape}"
            )
        if np.any(~np.isfinite(transform)):
            raise ValueError("cone equalizer must be finite")
        self.input_dim = int(input_dim)
        self.n_features = int(input_dim)
        self.equalizer_path = str(equalizer_path)
        self.transform_matrix = np.ascontiguousarray(transform)
        self.equalizer_sha256 = hashlib.sha256(open(equalizer_path, "rb").read()).hexdigest()

    def transform(self, embeddings) -> np.ndarray:
        values = _unit_rows(embeddings, self.input_dim)
        equalized = values @ self.transform_matrix.T
        norms = np.linalg.norm(equalized, axis=1, keepdims=True)
        if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
            raise ValueError("cone-equalized embeddings must be finite and non-zero")
        return equalized / norms

    def metadata(self) -> dict[str, int | bool | str | float | None]:
        return {
            "type": self.kind,
            "kernel": "linear",
            "input_dim": self.input_dim,
            "n_features": self.n_features,
            "row_normalized": True,
            "exact": True,
            "equalizer_path": self.equalizer_path,
            "equalizer_sha256": self.equalizer_sha256,
            "equalizer_scale": self.scale,
            "fit_lambda_1": self.lambda_1,
            "fit_lambda_perp": self.lambda_perp,
        }

    @classmethod
    def from_metadata(cls, metadata: dict) -> "ConeEqualizedLinearSphericalFeatures":
        if metadata.get("type") != cls.kind:
            raise ValueError(f"unsupported feature type {metadata.get('type')!r}")
        instance = cls(int(metadata["input_dim"]), str(metadata["equalizer_path"]))
        expected = metadata.get("equalizer_sha256")
        if expected is not None and expected != instance.equalizer_sha256:
            raise ValueError(
                "cone equalizer does not match the generation fingerprint; "
                "generation and detection must use the identical public map"
            )
        return instance


class MeanAxisShrunkLinearSphericalFeatures:
    """Shrink the corpus mean direction before applying the keyed field.

    The unit mean axis is estimated from independent, unwatermarked text and
    stored in a frozen transform.  For shrink factor ``s``, the map applies
    ``P_s = I + (s - 1) u u^T`` and then renormalizes each row.  ``s=1`` is
    the identity; smaller positive values attenuate only the shared mean axis.
    """

    kind = "mean-axis-shrunk-linear-spherical-v1"

    def __init__(self, input_dim: int, transform_path: str):
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        with np.load(transform_path, allow_pickle=False) as payload:
            if "transform" not in payload or "axis" not in payload:
                raise ValueError("mean-axis transform must contain transform and axis")
            transform = np.asarray(payload["transform"], dtype=np.float64)
            axis = np.asarray(payload["axis"], dtype=np.float64)
            self.scale = float(payload["scale"])
            self.mean_norm = float(payload["mean_norm"])
            self.fit_sentences = int(payload["n_sentences"])
            self.fit_documents = int(payload["n_documents"])
            self.fit_source = str(payload["source"])
            self.fit_source_sha256 = str(payload["source_sha256"])
            self.fit_embedder = str(payload["embedder"])
            self.fit_split_counts_json = str(payload["split_counts"])
            self.axis = np.asarray(axis, dtype=np.float64)
        if transform.shape != (int(input_dim), int(input_dim)):
            raise ValueError(
                f"mean-axis transform must have shape ({input_dim}, {input_dim}), "
                f"found {transform.shape}"
            )
        if axis.shape != (int(input_dim),):
            raise ValueError(f"mean axis must have shape ({input_dim},)")
        if np.any(~np.isfinite(transform)) or np.any(~np.isfinite(axis)):
            raise ValueError("mean-axis transform and axis must be finite")
        if not 0.0 < self.scale <= 1.0:
            raise ValueError("mean-axis shrink scale must be in (0, 1]")
        if not math.isfinite(self.mean_norm) or self.mean_norm <= 0.0:
            raise ValueError("fitted mean norm must be finite and positive")
        axis_norm = float(np.linalg.norm(axis))
        if not np.isclose(axis_norm, 1.0, atol=1e-6):
            raise ValueError("mean axis must have unit norm")
        expected = np.eye(int(input_dim), dtype=np.float64) + (
            self.scale - 1.0
        ) * np.outer(axis, axis)
        if not np.allclose(transform, expected, atol=2e-8, rtol=2e-8):
            raise ValueError("saved transform does not match its axis and scale")
        self.input_dim = int(input_dim)
        self.n_features = int(input_dim)
        self.transform_path = str(transform_path)
        self.transform_matrix = np.ascontiguousarray(transform)
        with open(transform_path, "rb") as handle:
            self.transform_sha256 = hashlib.sha256(handle.read()).hexdigest()

    def transform(self, embeddings) -> np.ndarray:
        values = _unit_rows(embeddings, self.input_dim)
        mapped = values @ self.transform_matrix.T
        norms = np.linalg.norm(mapped, axis=1, keepdims=True)
        if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
            raise ValueError("mean-axis-shrunk embeddings must be finite and non-zero")
        return mapped / norms

    def metadata(self) -> dict[str, int | bool | str | float]:
        return {
            "type": self.kind,
            "kernel": "linear",
            "input_dim": self.input_dim,
            "n_features": self.n_features,
            "row_normalized": True,
            "exact": True,
            "transform_path": self.transform_path,
            "transform_sha256": self.transform_sha256,
            "mean_axis_shrink_scale": self.scale,
            "fit_mean_norm": self.mean_norm,
            "fit_sentences": self.fit_sentences,
            "fit_documents": self.fit_documents,
            "fit_split_counts_json": self.fit_split_counts_json,
            "fit_embedder": self.fit_embedder,
            "fit_source": self.fit_source,
            "fit_source_sha256": self.fit_source_sha256,
        }

    @classmethod
    def from_metadata(cls, metadata: dict) -> "MeanAxisShrunkLinearSphericalFeatures":
        if metadata.get("type") != cls.kind:
            raise ValueError(f"unsupported feature type {metadata.get('type')!r}")
        instance = cls(int(metadata["input_dim"]), str(metadata["transform_path"]))
        expected = metadata.get("transform_sha256")
        if expected is not None and expected != instance.transform_sha256:
            raise ValueError(
                "mean-axis transform does not match the generation fingerprint; "
                "generation and detection must use the identical public map"
            )
        if float(metadata.get("mean_axis_shrink_scale", instance.scale)) != instance.scale:
            raise ValueError("mean-axis shrink scale does not match the saved transform")
        return instance


def build_feature_map(
    kernel: str,
    *,
    input_dim: int,
    feature_map: str = "linear",
    residual_weight: float = 0.5,
    whitener_path: str | None = None,
    cone_equalizer_path: str | None = None,
    mean_axis_shrink_path: str | None = None,
    projection_path: str | None = None,
    projection_key: str = "basis_k64",
    router_path: str | None = None,
    n_features: int = 2048,
    length_scale: float = 0.5,
    seed: int = 0,
    **_ignored,
):
    if str(kernel).lower() not in {"linear", "cosine"}:
        raise ValueError("SemField supports only the linear kernel")
    mode = str(feature_map).lower().replace("_", "-")
    if mode in {"linear", "lf", "linear-spherical"}:
        return LinearSphericalFeatures(input_dim)
    if mode in {"poly2", "polynomial2", "polynomial-degree-2"}:
        return PolynomialRandomFeatures(input_dim, int(n_features), 2, int(seed))
    if mode in {"poly3", "polynomial3", "polynomial-degree-3"}:
        return PolynomialRandomFeatures(input_dim, int(n_features), 3, int(seed))
    if mode in {"rbf", "rbf-rff", "random-fourier", "random-fourier-rbf"}:
        return RBFRandomFourierFeatures(
            input_dim, int(n_features), float(length_scale), int(seed)
        )
    if mode in {"content-routed", "content-route", "atlas", "atlas-linear"}:
        if not router_path:
            raise ValueError("the content-routed feature map needs --router-path")
        return ContentRoutedLinearSphericalFeatures(input_dim, router_path)
    if mode in {"residual", "innovation", "residual-linear"}:
        return ResidualLinearSphericalFeatures(input_dim)
    if mode in {"span-residual", "full-residual", "span-innovation", "innovation-span"}:
        return SpanResidualLinearSphericalFeatures(input_dim)
    if mode in {"hybrid-residual", "residual-hybrid", "hybrid-innovation"}:
        return ResidualLinearSphericalFeatures(
            input_dim, blend_weight=float(residual_weight)
        )
    if mode in {"whitened", "whitened-linear", "whitened-linear-spherical"}:
        if not whitener_path:
            raise ValueError("the whitened feature map needs --whitener-path")
        return WhitenedLinearSphericalFeatures(input_dim, whitener_path)
    if mode in {
        "cone-equalized", "cone-equalizer", "cone-axis-equalized",
        "axis-equalized",
    }:
        path = cone_equalizer_path or whitener_path
        if not path:
            raise ValueError("the cone-equalized feature map needs --cone-equalizer-path")
        return ConeEqualizedLinearSphericalFeatures(input_dim, path)
    if mode in {"mean-axis-shrunk", "mean-shrunk", "mean-axis-shrink"}:
        if not mean_axis_shrink_path:
            raise ValueError("the mean-axis-shrunk feature map needs --mean-axis-shrink-path")
        return MeanAxisShrunkLinearSphericalFeatures(input_dim, mean_axis_shrink_path)
    if mode in {
        "projected", "projected-linear", "projected-linear-spherical",
        "candidate-pca", "candidate-projection",
    }:
        if not projection_path:
            raise ValueError("the projected feature map needs --projection-path")
        return ProjectedLinearSphericalFeatures(
            input_dim, projection_path, projection_key
        )
    raise ValueError(f"unsupported feature map {feature_map!r}")


def feature_map_from_metadata(metadata: dict):
    kind = metadata.get("type")
    if kind == LinearSphericalFeatures.kind:
        return LinearSphericalFeatures.from_metadata(metadata)
    if kind == PolynomialRandomFeatures.kind:
        return PolynomialRandomFeatures.from_metadata(metadata)
    if kind == RBFRandomFourierFeatures.kind:
        return RBFRandomFourierFeatures.from_metadata(metadata)
    if kind == ContentRoutedLinearSphericalFeatures.kind:
        return ContentRoutedLinearSphericalFeatures.from_metadata(metadata)
    if kind in {
        ResidualLinearSphericalFeatures.residual_kind,
        ResidualLinearSphericalFeatures.hybrid_kind,
    }:
        return ResidualLinearSphericalFeatures.from_metadata(metadata)
    if kind == SpanResidualLinearSphericalFeatures.kind:
        return SpanResidualLinearSphericalFeatures.from_metadata(metadata)
    if kind == WhitenedLinearSphericalFeatures.kind:
        return WhitenedLinearSphericalFeatures.from_metadata(metadata)
    if kind == ConeEqualizedLinearSphericalFeatures.kind:
        return ConeEqualizedLinearSphericalFeatures.from_metadata(metadata)
    if kind == MeanAxisShrunkLinearSphericalFeatures.kind:
        return MeanAxisShrunkLinearSphericalFeatures.from_metadata(metadata)
    if kind == ProjectedLinearSphericalFeatures.kind:
        return ProjectedLinearSphericalFeatures.from_metadata(metadata)
    raise ValueError(f"unsupported SemField feature type {kind!r}")
