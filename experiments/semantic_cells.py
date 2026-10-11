"""Reconstruct public semantic cell maps used by legacy baselines."""
from __future__ import annotations

import math
from typing import Sequence


def _n_cells(payload: dict) -> int:
    value = payload.get("cell_map", {}).get("n_cells")
    if value is None:
        bits = payload.get("cell_map", {}).get("bits")
        if bits is not None:
            value = 2 ** int(bits)
    if value is None or int(value) < 2:
        raise ValueError("input has no valid cell_map.n_cells")
    return int(value)


class TextCellEncoder:
    """Lazily reconstruct an LSH cell map from generation metadata."""

    def __init__(self, payload: dict, embedder_name: str | None,
                 device: str | None):
        self.payload = payload
        self.embedder_name = (
            embedder_name or payload.get("config", {}).get("embedder")
        )
        self.device = device
        self._embed = None
        self._cell_map = None

    def __call__(self, sentences: Sequence[str]) -> tuple[int, ...]:
        if not sentences:
            return ()
        if not self.embedder_name:
            raise ValueError("sentence inputs require --embedder")
        if self._embed is None:
            from semfield.cells import LSHCells
            from semfield.embedders import embed, load_embedder, resolve_devices

            devices = resolve_devices(self.device)
            encoder = load_embedder(self.embedder_name, device=devices[0])

            def encode(items):
                return embed(
                    encoder, list(items), devices=devices,
                    model_name=self.embedder_name,
                )

            probe = encode(["Embedding-width probe."])
            metadata = self.payload.get("cell_map", {})
            bits = int(metadata.get(
                "bits", round(math.log2(_n_cells(self.payload)))
            ))
            self._cell_map = LSHCells(
                dim=probe.shape[1], d=bits,
                seed=int(metadata.get("seed", 0)),
            )
            self._embed = encode
        return tuple(
            int(value) for value in self._cell_map(self._embed(sentences))
        )
