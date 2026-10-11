"""Faithful row-conditioned SemStamp structure under an ideal keyed PRF.

SemStamp does not use one global green mask.  The previous sentence's semantic
cell selects a row, and that row contains a fixed-size subset of valid next
cells.  This module implements that structure directly:

* every predecessor row has exactly ``n_valid`` outgoing cells;
* rows are independent uniform subsets in the ideal-PRF model;
* generation accepts the first completed sentence whose transition is valid
  and, optionally, whose LSH margin clears a public threshold; and
* detection uses the exact row-stratified null from
  :mod:`semfield.semantic_calibration`.

The keyed HMAC/Fisher--Yates construction is an ideal-PRF baseline.  It is not
bit-for-bit compatible with the released SemStamp repository's RNG seeding.
That distinction is exposed in metadata rather than hidden behind the scheme
name.  The semantic transition structure and fixed-size row law are faithful;
the concrete key-to-mask mapping is deliberately implementation independent.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import struct
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .semantic_calibration import (
    NullMoments,
    semstamp_row_moments,
    semstamp_row_pmf,
    semstamp_row_pvalue,
    semstamp_transition_counts,
)

__all__ = [
    "IDEAL_PRF_MODE",
    "RELEASED_RNG_COMPATIBLE",
    "RowDecision",
    "RowSelection",
    "RowDetectionResult",
    "IdealRowSemStamp",
]


IDEAL_PRF_MODE = "ideal-prf-independent-fixed-size-rows-v1"
RELEASED_RNG_COMPATIBLE = False
_ROW_DOMAIN = bytes.fromhex("706163742d696465616c2d726f772d73656d7374616d702d763100")
_TWO64 = 1 << 64


@dataclass(frozen=True)
class RowDecision:
    """Acceptance decision for one completed-sentence candidate."""

    previous_cell: int
    current_cell: int
    green: bool
    margin: float | None
    margin_ok: bool
    accepted: bool


@dataclass(frozen=True)
class RowSelection:
    """First accepted candidate in a sequential precomputed candidate list."""

    index: int | None
    attempts: int
    satisfied: bool
    decision: RowDecision | None


@dataclass(frozen=True)
class RowDetectionResult:
    """Observed score and exact ideal-row conditional null."""

    score: int
    n_transitions: int
    pvalue: float
    moments: NullMoments
    transition_counts: np.ndarray
    pmf: np.ndarray
    initial_cell: int | None
    mask_mode: str = IDEAL_PRF_MODE
    released_rng_compatible: bool = RELEASED_RNG_COMPATIBLE


def _key_bytes(key: bytes | bytearray | memoryview | str) -> bytes:
    if isinstance(key, str):
        value = key.encode("utf-8")
    elif isinstance(key, (bytes, bytearray, memoryview)):
        value = bytes(key)
    else:
        raise TypeError("key must be bytes-like or str")
    if not value:
        raise ValueError("key must not be empty")
    return value


class _HMACStream:
    """Domain-separated deterministic pseudorandom 64-bit words."""

    def __init__(self, key: bytes, context: bytes):
        self.key = key
        self.context = context
        self.counter = 0
        self.buffer = b""

    def uint64(self) -> int:
        if len(self.buffer) < 8:
            block = hmac.new(
                self.key,
                self.context + self.counter.to_bytes(8, "big"),
                hashlib.sha256,
            ).digest()
            self.counter += 1
            self.buffer += block
        value = int.from_bytes(self.buffer[:8], "big")
        self.buffer = self.buffer[8:]
        return value

    def randbelow(self, upper: int) -> int:
        if upper <= 0 or upper > _TWO64:
            raise ValueError("upper must lie in [1, 2**64]")
        limit = _TWO64 - (_TWO64 % upper)
        while True:
            value = self.uint64()
            if value < limit:
                return value % upper


class IdealRowSemStamp:
    """Independent fixed-size outgoing masks keyed by predecessor cell.

    Parameters
    ----------
    n_cells:
        Number of semantic cells ``K``.
    n_valid:
        Exact number ``G`` of accepted destinations in each predecessor row.
    key:
        Secret setup key.  HMAC is treated as an ideal PRF for the statistical
        claims.
    min_margin:
        Optional public LSH stability threshold.  A candidate is accepted only
        when it is in the row mask and has margin at least this value.  Set to
        zero to disable the margin condition.
    mask_mode:
        Only :data:`IDEAL_PRF_MODE` is implemented.  The released repository's
        concrete RNG stream is intentionally not claimed as bit compatible.
    """

    def __init__(
        self,
        n_cells: int,
        n_valid: int,
        key: bytes | bytearray | memoryview | str,
        *,
        min_margin: float = 0.0,
        mask_mode: str = IDEAL_PRF_MODE,
    ):
        self.n_cells = int(n_cells)
        self.n_valid = int(n_valid)
        if self.n_cells <= 0:
            raise ValueError("n_cells must be positive")
        if not 0 < self.n_valid < self.n_cells:
            raise ValueError("n_valid must lie strictly between 0 and n_cells")
        self.key = _key_bytes(key)
        self.min_margin = float(min_margin)
        if not math.isfinite(self.min_margin) or self.min_margin < 0.0:
            raise ValueError("min_margin must be finite and non-negative")
        if mask_mode != IDEAL_PRF_MODE:
            raise NotImplementedError(
                "only the ideal-PRF row baseline is implemented; it is not "
                "bit-for-bit compatible with the released SemStamp RNG"
            )
        self.mask_mode = mask_mode
        self._rows = self._build_rows()

    @property
    def gamma(self) -> float:
        return self.n_valid / self.n_cells

    @property
    def released_rng_compatible(self) -> bool:
        return RELEASED_RNG_COMPATIBLE

    def metadata(self) -> dict[str, object]:
        return {
            "scheme": "semstamp-row",
            "mask_mode": self.mask_mode,
            "n_cells": self.n_cells,
            "n_valid": self.n_valid,
            "gamma": self.gamma,
            "min_margin": self.min_margin,
            "released_rng_compatible": self.released_rng_compatible,
            "compatibility_note": (
                "faithful ideal fixed-size row law; key-to-mask mapping is "
                "not the released SemStamp RNG stream"
            ),
        }

    def _row_permutation(self, previous_cell: int) -> np.ndarray:
        previous = self._validate_cell(previous_cell, "previous_cell")
        context = _ROW_DOMAIN + struct.pack(">II", self.n_cells, previous)
        stream = _HMACStream(self.key, context)
        order = list(range(self.n_cells))
        for i in range(self.n_cells - 1, 0, -1):
            j = stream.randbelow(i + 1)
            order[i], order[j] = order[j], order[i]
        return np.asarray(order, dtype=np.int64)

    def _build_rows(self) -> np.ndarray:
        rows = np.zeros((self.n_cells, self.n_cells), dtype=bool)
        if self.n_valid == 0:
            return rows
        if self.n_valid == self.n_cells:
            rows.fill(True)
            return rows
        for previous in range(self.n_cells):
            order = self._row_permutation(previous)
            rows[previous, order[:self.n_valid]] = True
        return rows

    def _validate_cell(self, cell: int, name: str) -> int:
        value = int(cell)
        if not 0 <= value < self.n_cells:
            raise ValueError(f"{name} is outside [0, n_cells)")
        return value

    def mask_matrix(self) -> np.ndarray:
        """Copy of the ``K x K`` boolean outgoing-mask table."""
        return self._rows.copy()

    def green_mask(self, previous_cell: int) -> np.ndarray:
        """Copy of the fixed-size valid-destination mask for one row."""
        previous = self._validate_cell(previous_cell, "previous_cell")
        return self._rows[previous].copy()

    def green_cells(self, previous_cell: int) -> np.ndarray:
        return np.flatnonzero(self.green_mask(previous_cell))

    def decide(
        self,
        previous_cell: int,
        current_cell: int,
        *,
        margin: float | None = None,
    ) -> RowDecision:
        """Evaluate one completed candidate without mutating scheme state."""
        previous = self._validate_cell(previous_cell, "previous_cell")
        current = self._validate_cell(current_cell, "current_cell")
        if margin is None:
            margin_ok = self.min_margin == 0.0
            margin_value = None
        else:
            margin_value = float(margin)
            if not math.isfinite(margin_value):
                raise ValueError("margin must be finite")
            margin_ok = margin_value >= self.min_margin
        green = bool(self._rows[previous, current])
        return RowDecision(
            previous_cell=previous,
            current_cell=current,
            green=green,
            margin=margin_value,
            margin_ok=bool(margin_ok),
            accepted=bool(green and margin_ok),
        )

    def select_first(
        self,
        candidate_cells: Sequence[int] | np.ndarray,
        previous_cell: int,
        *,
        margins: Sequence[float] | np.ndarray | None = None,
    ) -> RowSelection:
        """Select the first accepted candidate in sequential proposal order."""
        cells = np.asarray(candidate_cells, dtype=np.int64).ravel()
        if margins is None:
            margin_values: list[float | None] = [None] * cells.size
        else:
            raw_margins = np.asarray(margins, dtype=float).ravel()
            if raw_margins.size != cells.size:
                raise ValueError("margins must match candidate_cells")
            margin_values = raw_margins.tolist()
        last: RowDecision | None = None
        for index, (cell, margin) in enumerate(zip(cells, margin_values)):
            last = self.decide(previous_cell, int(cell), margin=margin)
            if last.accepted:
                return RowSelection(index, index + 1, True, last)
        return RowSelection(None, int(cells.size), False, last)

    def transition_counts(
        self,
        cell_ids: Iterable[int],
        *,
        initial_cell: int | None = None,
    ) -> np.ndarray:
        """Transition table, optionally including prompt-to-first transition."""
        cells = [self._validate_cell(cell, "cell_id") for cell in cell_ids]
        if initial_cell is not None:
            cells.insert(0, self._validate_cell(initial_cell, "initial_cell"))
        return semstamp_transition_counts(cells, self.n_cells)

    def score(
        self,
        cell_ids: Iterable[int],
        *,
        initial_cell: int | None = None,
    ) -> int:
        counts = self.transition_counts(cell_ids, initial_cell=initial_cell)
        return int(counts[self._rows].sum())

    def detect(
        self,
        cell_ids: Iterable[int],
        *,
        initial_cell: int | None = None,
    ) -> RowDetectionResult:
        """Exact one-sided detector under independent uniform row masks."""
        counts = self.transition_counts(cell_ids, initial_cell=initial_cell)
        score = int(counts[self._rows].sum())
        pmf = semstamp_row_pmf(counts, self.n_valid)
        moments = semstamp_row_moments(counts, self.n_valid)
        return RowDetectionResult(
            score=score,
            n_transitions=int(counts.sum()),
            pvalue=semstamp_row_pvalue(score, counts, self.n_valid),
            moments=moments,
            transition_counts=counts,
            pmf=pmf,
            initial_cell=(None if initial_cell is None else int(initial_cell)),
        )
