"""Cell maps: sentence -> discrete semantic cell.

Every sentence-level semantic watermark is built from a map that sends a
sentence to one of a finite set of cells, chosen so that a paraphrase usually
lands in the same cell. The published schemes differ only in this map:

  LSHCells      the sign pattern of d random hyperplanes applied to a sentence
                embedding. This is SemStamp, which uses d = 3, giving 8 cells.

  KMeansCells   the index of the nearest of K centroids fitted to a corpus.
                This is k-SemStamp.

Both are provided so that experiments can measure the design effect of the
published configurations rather than a stand-in for them.

Schemes that derive the green set from the previous sentence's cell, which is
inherited from the token-level red/green design, effectively use the pair
(previous cell, current cell) as the unit. transition_cells builds that pair
so the design effect is measured on the correct unit.
"""

from __future__ import annotations

import numpy as np

__all__ = ["LSHCells", "KMeansCells", "transition_cells"]


class LSHCells:
    """SemStamp's cell map: signs of d random hyperplanes.

    The signature of an embedding v is the bit vector [1{n_i . v > 0}] over d
    random Gaussian directions n_i, read as an integer in [0, 2^d).

    d = 3 is the released SemStamp configuration, giving 8 signatures.
    """

    def __init__(self, dim: int, d: int = 3, seed: int = 0):
        self.dim, self.d = dim, d
        rng = np.random.default_rng(seed)
        self.normals = rng.normal(size=(d, dim))

    @property
    def n_cells(self) -> int:
        return 2**self.d

    def __call__(self, embeddings) -> np.ndarray:
        v = np.atleast_2d(np.asarray(embeddings, dtype=float))
        bits = (v @ self.normals.T) > 0                 # (n_sentences, d)
        powers = 1 << np.arange(self.d)
        return (bits * powers).sum(axis=1).astype(int)

    def margins(self, embeddings) -> np.ndarray:
        """Distance to the nearest hyperplane, per sentence.

        SemStamp rejects candidate sentences whose embedding sits close to a
        hyperplane, because a paraphrase can push them across it. Reported so
        that experiments can separate cell instability from calibration error.
        """
        v = np.atleast_2d(np.asarray(embeddings, dtype=float))
        unit = self.normals / np.linalg.norm(self.normals, axis=1, keepdims=True)
        vn = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        return np.abs(vn @ unit.T).min(axis=1)


class KMeansCells:
    """k-SemStamp's cell map: index of the nearest of K centroids.

    Centroids are fitted on a random subsample when the corpus is larger than
    fit_sample. This matches k-SemStamp, where the centroids come from a sample
    of a domain corpus rather than from the documents under test, and it is the
    one place in the pipeline whose cost is superlinear in nuisance: ten
    restarts of Lloyd's algorithm over 646000 book sentences take minutes to
    place eight centroids that 100000 sentences already fix. Set fit_sample to
    0 to fit on everything. Assignment always covers every sentence.

    The arithmetic stays in float64. float32 halves the memory but, measured on
    this hardware, makes the fit slower rather than faster, and it perturbs the
    centroids enough to move rho in the third digit -- a cost with no matching
    benefit.
    """

    def __init__(self, n_clusters: int = 8, seed: int = 0,
                 fit_sample: int = 100_000):
        self.n_clusters, self.seed = n_clusters, seed
        self.fit_sample = fit_sample
        self.centroids: np.ndarray | None = None

    @property
    def n_cells(self) -> int:
        return self.n_clusters

    def fit(self, embeddings) -> "KMeansCells":
        from sklearn.cluster import KMeans

        v = np.asarray(embeddings, dtype=float)
        if self.fit_sample and v.shape[0] > self.fit_sample:
            rng = np.random.default_rng(self.seed)
            keep = rng.choice(v.shape[0], self.fit_sample, replace=False)
            keep.sort()                      # sequential reads beat a gather
            v = v[keep]
        km = KMeans(n_clusters=self.n_clusters, random_state=self.seed, n_init=10)
        km.fit(v)
        self.centroids = km.cluster_centers_
        return self

    def __call__(self, embeddings) -> np.ndarray:
        if self.centroids is None:
            raise RuntimeError("call fit() before using KMeansCells")
        v = np.atleast_2d(np.asarray(embeddings, dtype=float))
        # cosine distance, matching k-SemStamp
        vn = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        cn = self.centroids / np.maximum(
            np.linalg.norm(self.centroids, axis=1, keepdims=True), 1e-12
        )
        return np.argmax(vn @ cn.T, axis=1).astype(int)


def transition_cells(cell_ids, n_cells: int) -> np.ndarray:
    """Pair each sentence's cell with the previous sentence's cell.

    Schemes that seed the green set from the previous sentence use the pair as
    the effective unit, so there are n_cells^2 of them. The first sentence has
    no predecessor and is dropped.
    """
    ids = np.asarray(cell_ids, dtype=int)
    if ids.size < 2:
        return np.zeros(0, dtype=int)
    return (ids[:-1] * n_cells + ids[1:]).astype(int)
