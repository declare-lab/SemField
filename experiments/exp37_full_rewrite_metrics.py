"""Aggregate score bundles into AUROC and true TPR@1%/TPR@5% FPR."""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, roc_curve


def named_result(value: str) -> tuple[str, str, Path]:
    if "=" not in value or ":" not in value.split("=", 1)[0]:
        raise argparse.ArgumentTypeError("use METHOD:CONDITION=PATH")
    name, raw = value.split("=", 1)
    method, condition = name.split(":", 1)
    path = Path(raw).expanduser().resolve()
    if not method or not condition or not path.is_file():
        raise argparse.ArgumentTypeError(f"invalid result {value!r}")
    return method, condition, path


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try: os.unlink(tmp)
        except FileNotFoundError: pass
        raise


def metric(null: np.ndarray, positive: np.ndarray) -> dict:
    if not len(null) or not len(positive):
        raise ValueError("both classes must be nonempty")
    if not np.isfinite(null).all() or not np.isfinite(positive).all():
        raise ValueError("scores must be finite")
    labels = np.r_[np.zeros(len(null), dtype=int),
                   np.ones(len(positive), dtype=int)]
    scores = np.r_[null, positive]
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)

    def interpolated(target: float) -> float:
        return float(np.interp(target, fpr, tpr))

    def conservative(target: float) -> dict:
        valid = np.flatnonzero(fpr <= target + 1e-15)
        index = int(valid[np.argmax(tpr[valid])])
        threshold = thresholds[index]
        return {"tpr": float(tpr[index]), "fpr": float(fpr[index]),
                "threshold": None if math.isinf(threshold) else float(threshold)}

    return {
        "auroc": float(auc(fpr, tpr)),
        "tpr_at_fpr_interpolated": {
            "0.01": interpolated(0.01), "0.05": interpolated(0.05)},
        "tpr_at_fpr_empirical_conservative": {
            "0.01": conservative(0.01), "0.05": conservative(0.05)},
        # The AliMark release prints these as 1%/5%, but they are actually
        # evaluated at 0.1%/0.5%. Keep them only for transparent compatibility.
        "alimark_release_code_compat": {
            "actual_fpr_0.001": interpolated(0.001),
            "actual_fpr_0.005": interpolated(0.005)},
        "null_resolution": float(1.0 / len(null)),
    }


def bootstrap(null: np.ndarray, positive: np.ndarray, count: int,
              seed: int) -> dict:
    if count == 0:
        return {}
    rng = np.random.default_rng(seed)
    values = {"auroc": [], "tpr@1%": [], "tpr@5%": []}
    for _ in range(count):
        result = metric(null[rng.integers(len(null), size=len(null))],
                        positive[rng.integers(len(positive), size=len(positive))])
        values["auroc"].append(result["auroc"])
        values["tpr@1%"].append(result["tpr_at_fpr_interpolated"]["0.01"])
        values["tpr@5%"].append(result["tpr_at_fpr_interpolated"]["0.05"])
    return {name: [float(x) for x in np.quantile(data, [0.025, 0.975])]
            for name, data in values.items()}


def markdown(rows: list[dict]) -> str:
    lines = [
        "| Method | Condition | AUROC | TPR@1% FPR | TPR@5% FPR | null:wm |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        m = row["metrics"]
        lines.append(
            f"| {row['method']} | {row['condition']} | {100*m['auroc']:.1f} | "
            f"{100*m['tpr_at_fpr_interpolated']['0.01']:.1f} | "
            f"{100*m['tpr_at_fpr_interpolated']['0.05']:.1f} | "
            f"{row['n_null']}:{row['n_watermarked']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", required=True,
                        metavar="METHOD:CONDITION=PATH")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.bootstrap < 0: parser.error("--bootstrap cannot be negative")

    rows = []
    for index, (method, condition, path) in enumerate(
            named_result(value) for value in args.result):
        payload = json.loads(path.read_text(encoding="utf-8"))
        bundle = payload.get("scores", {})
        ids = [int(value) for value in bundle.get("document_ids", [])]
        null = np.asarray(bundle.get("null", []), dtype=float)
        positive = np.asarray(bundle.get("watermarked", []), dtype=float)
        if ids and (len(ids) != len(null) or len(ids) != len(positive)):
            raise ValueError(f"score/ID mismatch in {path}")
        # Each row's AUROC/TPR is computed only from its own null/positive
        # arrays, so methods do not need a matching document cohort. EXP is
        # legitimately 499/500 (see exp38_markllm_semantic_shared.py) while
        # every other method is 500/500.
        result = metric(null, positive)
        result["bootstrap_percentile_95"] = bootstrap(
            null, positive, args.bootstrap, args.seed + index)
        rows.append({"method": method, "condition": condition,
                     "source": str(path), "n_null": len(null),
                     "n_watermarked": len(positive), "metrics": result})

    output = {"schema_version": 1,
              "experiment": "exp37_full_rewrite_metrics",
              "protocol": {
                  "negative": "untouched human text",
                  "positive": "clean or completely rewritten watermarked text",
                  "reported_fpr_targets": [0.01, 0.05],
                  "partial_paraphrases_included": False,
                  "operating_point": "linear interpolation on sklearn ROC",
              },
              "rows": rows}
    out = Path(args.out).resolve()
    atomic_json(out, output)
    out.with_suffix(".md").write_text(markdown(rows), encoding="utf-8")
    print(markdown(rows), end="")
    print(f"wrote {out} and {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
