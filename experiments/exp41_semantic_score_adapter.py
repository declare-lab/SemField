"""Convert native semantic robustness results to the common score schema."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def score_for(experiment: str, row: dict, statistic: str) -> float | None:
    if experiment == "exp12_semclock_robustness":
        value = row["z_score"] if statistic == "whole" else row["scan_statistic"]
    elif experiment == "exp20_semfield_robustness":
        if statistic == "whole":
            value = row["analytic"]["z_score"]
        elif statistic == "control-whole":
            value = row["control_whole"]["statistic"]
        else:
            value = row["control_scan"]["statistic"]
    elif experiment == "exp17_semstamp_robustness":
        value = row["z_score_exact_null"]
    else:
        raise ValueError(f"unsupported experiment {experiment!r}")
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("native result contains a non-finite score")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--attack", default="paraphrase")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument(
        "--statistic", choices=["whole", "control-whole", "scan"],
        default="whole",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    source = Path(args.input).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    experiment = str(payload.get("experiment", ""))
    candidates = [
        condition for condition in payload.get("conditions", [])
        if condition.get("attack") == args.attack
        and math.isclose(float(condition.get("strength", -1.0)), args.strength)
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"expected one {args.attack}@{args.strength:g} condition; "
            f"found {len(candidates)}"
        )
    documents = candidates[0].get("documents", {})
    watermarked = {
        int(row["document_id"]): row
        for row in documents.get("watermarked", [])
    }
    null = {
        int(row["document_id"]): row for row in documents.get("null", [])
    }
    paired = []
    for document_id in sorted(set(watermarked) & set(null)):
        null_score = score_for(experiment, null[document_id], args.statistic)
        watermarked_score = score_for(
            experiment, watermarked[document_id], args.statistic
        )
        if null_score is not None and watermarked_score is not None:
            paired.append((document_id, null_score, watermarked_score))
    if not paired:
        raise ValueError("condition has no paired watermarked/null documents")

    bundle = {
        "schema_version": 1,
        "experiment": "exp41_semantic_score_adapter",
        "source_experiment": experiment,
        "condition": {"attack": args.attack, "strength": args.strength},
        "statistic": args.statistic,
        "source": str(source),
        "scores": {
            "document_ids": [row[0] for row in paired],
            "null": [row[1] for row in paired],
            "watermarked": [row[2] for row in paired],
        },
    }
    destination = Path(args.out).expanduser().resolve()
    atomic_json(destination, bundle)
    print(f"wrote {destination} ({len(paired)} paired documents)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
