"""
TPR metrics for SAMark sentence-level detection.

Reads detection logs produced by samark_detect.py and computes TPR@FPR and AUC
using sentence-level z-scores. Saves per-attack summaries and a consolidated file.
"""

import os
import json
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score


def detect_score_direction(neg_scores, pos_scores):
    return "higher" if np.mean(pos_scores) > np.mean(neg_scores) else "lower"


def calculate_tpr_at_fpr(neg_scores, pos_scores, fpr_target=0.01, direction="auto"):
    neg = np.array(neg_scores)
    pos = np.array(pos_scores)
    if direction == "auto":
        direction = detect_score_direction(neg, pos)
    if direction == "higher":
        threshold = np.percentile(neg, 100 * (1 - fpr_target))
        tp = np.sum(pos >= threshold)
    else:
        threshold = np.percentile(neg, 100 * fpr_target)
        tp = np.sum(pos <= threshold)
    tpr = tp / len(pos) if len(pos) > 0 else 0.0
    return tpr, threshold, direction


def main():
    parser = argparse.ArgumentParser(description="SAMark sentence-level TPR metrics")
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--attack_name", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--detect_suffix", type=str, default="")
    args = parser.parse_args()

    root = args.root_dir
    attack_label = args.attack_name if args.attack_name else "no_attack"
    safe_attack_label = attack_label.replace("/", "_").replace("(", "").replace(")", "").replace(" ", "_")

    detect_folder = f"detect{args.detect_suffix}"
    nat_detect = f"{root}/{detect_folder}"
    atk_detect = f"{root}/attack/{args.attack_name}/{detect_folder}" if args.attack_name else None

    scores = {"nature": [], "watermark": [], "attack": []}
    decisions = {"nature": [], "watermark": [], "attack": []}

    for i in range(args.start, args.start + args.num_samples):
        nat_path = f"{nat_detect}/{i}.json"
        if not os.path.exists(nat_path):
            continue
        with open(nat_path, encoding="utf-8") as f:
            d = json.load(f)

        wm_log = d["watermark_log"]["sentence_level"]
        scores["watermark"].append(wm_log["z_score"])
        decisions["watermark"].append(bool(wm_log["is_watermarked"]))

        if d.get("origin_log"):
            orig = d["origin_log"]["sentence_level"]
            scores["nature"].append(orig["z_score"])
            decisions["nature"].append(bool(orig["is_watermarked"]))

        if atk_detect:
            atk_path = f"{atk_detect}/{i}.json"
            if os.path.exists(atk_path):
                with open(atk_path, encoding="utf-8") as f:
                    ad = json.load(f)
                atk = ad["watermark_log"]["sentence_level"]
                scores["attack"].append(atk["z_score"])
                decisions["attack"].append(bool(atk["is_watermarked"]))

    suffix_tag = args.detect_suffix if args.detect_suffix else ""
    out_path = f"{root}_sentence_scores{suffix_tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=4)

    n = len(scores["watermark"])
    if n == 0:
        print("No valid samples found.")
        return

    summary = {"num_samples": n, "attack_name": attack_label}

    nat = scores["nature"]
    wm = scores["watermark"]
    atk = scores["attack"]

    if nat:
        direction = detect_score_direction(nat, wm)
        for fpr in [0.01, 0.05]:
            tpr_wm, th, _ = calculate_tpr_at_fpr(nat, wm, fpr, direction)
            key = f"fpr_{int(fpr * 100)}"
            summary[key] = {"threshold": float(th), "tpr_watermark": float(tpr_wm)}
            if atk:
                tpr_atk, _, _ = calculate_tpr_at_fpr(nat, atk, fpr, direction)
                summary[key]["tpr_attack"] = float(tpr_atk)

        y_true_original = [0] * len(nat) + [1] * len(wm)
        y_scores_original = nat + wm
        if direction == "lower":
            y_scores_original = [-s for s in y_scores_original]
        summary["auc_original"] = float(roc_auc_score(y_true_original, y_scores_original))

        if atk:
            y_true_attack = [0] * len(nat) + [1] * len(atk)
            y_scores_attack = nat + atk
            if direction == "lower":
                y_scores_attack = [-s for s in y_scores_attack]
            summary["auc_attack"] = float(roc_auc_score(y_true_attack, y_scores_attack))

    if decisions["watermark"]:
        summary["decision_tpr_watermark"] = float(sum(decisions["watermark"]) / len(decisions["watermark"]))
    if decisions["nature"]:
        summary["decision_fpr_origin"] = float(sum(decisions["nature"]) / len(decisions["nature"]))
    if decisions["attack"]:
        summary["decision_tpr_attack"] = float(sum(decisions["attack"]) / len(decisions["attack"]))

    attack_results_dir = f"{root}/attack_results{suffix_tag}"
    os.makedirs(attack_results_dir, exist_ok=True)

    per_attack_path = f"{attack_results_dir}/{safe_attack_label}_summary.json"
    with open(per_attack_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    all_results_path = f"{attack_results_dir}/all.json"
    if os.path.exists(all_results_path):
        with open(all_results_path, encoding="utf-8") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    all_results[attack_label] = summary
    with open(all_results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    summary_path = f"{root}_sentence_summary{suffix_tag}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print(f"Saved: {per_attack_path}")
    print(f"Saved: {all_results_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
