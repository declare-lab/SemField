"""Convert official SAMark/SwordStamp detector arrays to exp37 bundles."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


CONDITIONS = ("clean", "pegasus", "parrot", "dipper", "llm-rewrite")
METHODS = ("samark", "swordstamp", "k-swordstamp")


def ids_for(dataset_dir: Path, n: int) -> list[int]:
    try:
        from datasets import load_from_disk
        values = load_from_disk(str(dataset_dir))["document_id"]
        return [int(x) for x in values[:n]]
    except Exception:
        return list(range(n))


def arrays(base: Path, method: str, condition: str):
    if condition == "clean":
        # SAMark's native clean detector writes under ``detect``; SwordStamp
        # keeps its clean arrays directly in the method directory.
        root = base / method / "detect" if method == "samark" else base / method
    else:
        # The SAMark attack wrapper adds a per-condition ``detect`` layer,
        # whereas SwordStamp/k-SwordStamp write arrays directly in the
        # condition directory.
        root = (base / method / "attack-eval" / condition / "detect"
                if method == "samark" else
                base / method / "attack-eval" / condition)
    if method == "samark":
        null_name, positive_name = "human_z_scores.npy", (
            "z_scores_text.npy" if condition == "clean" else "z_scores_para.npy")
    else:
        null_name, positive_name = "human_z_scores.npy", (
            "z_scores.npy" if condition == "clean" else "para_z_scores.npy")
    null = np.asarray(np.load(root / null_name), dtype=float).reshape(-1)
    positive = np.asarray(np.load(root / positive_name), dtype=float).reshape(-1)
    # Native detectors keep human scores in source-document order.  The
    # attack dataset carries IDs, which matters for the documented Qwen
    # 499-document Parrot/k-SwordStamp exclusion.
    if condition != "clean":
        ids = ids_for(root, len(positive))
        if len(ids) != len(positive):
            ids = list(range(min(len(null), len(positive))))
        valid = [i for i in ids if 0 <= i < len(null)]
        n = min(len(valid), len(positive))
        ids = valid[:n]
        null = null[ids]
        positive = positive[:n]
    else:
        n = min(len(null), len(positive))
        ids = list(range(n))
        null = null[:n]
        positive = positive[:n]
    return ids, null, positive


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    base, out = Path(args.base).resolve(), Path(args.out).resolve()
    score_dir = out / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    result_args = []
    for method in METHODS:
        for condition in CONDITIONS:
            ids, null, positive = arrays(base, method, condition)
            if not len(ids) or len(ids) != len(null) or len(ids) != len(positive):
                raise RuntimeError(f"invalid arrays for {method}/{condition}")
            path = score_dir / f"{method}__{condition}.json"
            path.write_text(json.dumps({
                "schema_version": 1,
                "experiment": "exp102_external_unified_scores",
                "method": method,
                "condition": condition,
                "source": str(base),
                "scores": {
                    "document_ids": ids,
                    "null": [float(x) for x in null],
                    "watermarked": [float(x) for x in positive],
                },
            }, indent=2) + "\n")
            result_args += ["--result", f"{method}:{condition}={path}"]
    report = out / "report" / "main-table.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable,
               str(Path(__file__).with_name("exp37_full_rewrite_metrics.py")),
               *result_args, "--bootstrap", "1000", "--out", str(report)]
    subprocess.run(command, check=True)
    print(f"wrote external unified report {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
