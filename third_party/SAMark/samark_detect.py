import os
import json
import argparse

import torch
import numpy as np
from tqdm import tqdm
from scipy import stats
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F


def get_random_vectors(dim, num, seed=42):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((dim, num * 6))
    Q, _ = np.linalg.qr(A)
    return torch.tensor(Q.T[:num, :])


def get_text_embeddings(texts, embedder):
    return embedder.encode(texts, convert_to_tensor=True).cpu()


def get_cosine_similarities(A, B):
    B = B.to(dtype=A.dtype, device=A.device)
    return F.cosine_similarity(A[:, None, :], B[None, :, :], dim=-1)


def apply_transform(x, mode="none", k=30.0):
    if mode == "tanh":
        return torch.tanh(k * x)
    return x


def detect_sentence_level(sentences, embedder, msig, z_threshold=3.0, transform="none", transform_k=30.0):
    if not sentences or msig == 0:
        return {
            "is_watermarked": False,
            "z_score": 0.0,
            "p_value": 1.0,
            "aligned_channels": [],
            "guessed_flags": [],
        }

    all_cossims = []
    for sentence in sentences:
        emb = get_text_embeddings([sentence], embedder)
        pivot = get_random_vectors(emb.shape[1], msig, seed=42).to(emb.device)
        cossim = get_cosine_similarities(emb, pivot)
        all_cossims.append(cossim)

    all_cossims = torch.cat(all_cossims, dim=0)

    channel_sums = all_cossims.sum(dim=0)
    guessed_flags = torch.where(channel_sums >= 0, 1.0, -1.0)

    aligned = all_cossims * guessed_flags
    aligned = apply_transform(aligned, mode=transform, k=transform_k)

    per_sent_aligned = (aligned > 0).sum(dim=1).tolist()

    flat = aligned.view(-1)
    mean_s = flat.mean().item()
    std_s = flat.std().item()
    n = flat.numel()

    if std_s < 1e-12:
        return {
            "is_watermarked": False,
            "z_score": 0.0,
            "p_value": 1.0,
            "aligned_channels": per_sent_aligned,
            "guessed_flags": guessed_flags.tolist(),
        }

    z_score = mean_s * (n ** 0.5) / std_s
    p_value = float(stats.norm.sf(z_score))
    is_wm = z_score > z_threshold

    return {
        "is_watermarked": bool(is_wm),
        "z_score": float(z_score),
        "p_value": p_value,
        "mean_score": mean_s,
        "guessed_flags": guessed_flags.tolist(),
        "aligned_channels": per_sent_aligned,
        "transform": transform,
        "transform_k": transform_k,
    }


def parse_args():
    p = argparse.ArgumentParser(description="SAMark detection (sentence-level only)")
    p.add_argument("--embedder_path", type=str, required=True)
    p.add_argument("--log_dir", type=str, required=True)
    p.add_argument("--sub_dir", type=str, default=None)
    p.add_argument("--msig", type=int, default=2)
    p.add_argument("--sent_z_threshold", type=float, default=3.0)
    p.add_argument("--detect_origin", action="store_true")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=500)
    p.add_argument("--transform", type=str, default="none", choices=["none", "tanh"])
    p.add_argument("--transform_param", type=float, default=30.0)
    p.add_argument("--detect_subdir", type=str, default="detect")
    return p.parse_args()


def main():
    args = parse_args()

    data_dir = f"{args.log_dir}/{args.sub_dir}" if args.sub_dir else args.log_dir
    detect_dir = f"{data_dir}/{args.detect_subdir}"
    os.makedirs(detect_dir, exist_ok=True)

    all_done = all(os.path.exists(f"{detect_dir}/{i}.json") for i in range(args.start, args.end))
    if all_done:
        print(f"All samples [{args.start}, {args.end}) already detected in {detect_dir}, skipping.")
        return

    with open(f"{detect_dir}/config.json", "w") as f:
        json.dump(args.__dict__, f, ensure_ascii=False, indent=4)

    embedder = SentenceTransformer(args.embedder_path, device="cuda")

    wm_results = []
    origin_results = []

    for i in tqdm(range(args.start, args.end)):
        log_path = f"{data_dir}/{i}.json"
        if not os.path.exists(log_path):
            continue

        out_path = f"{detect_dir}/{i}.json"
        if os.path.exists(out_path):
            continue

        with open(log_path, "r", encoding="utf-8") as f:
            gen_log = json.load(f)

        generated_text = gen_log.get("generated_text", "")
        detect_sentences = sent_tokenize(generated_text)
        sentence_result = detect_sentence_level(
            detect_sentences,
            embedder,
            args.msig,
            z_threshold=args.sent_z_threshold,
            transform=args.transform,
            transform_k=args.transform_param,
        )

        watermark_log = {"sentence_level": sentence_result}
        wm_results.append(sentence_result["is_watermarked"])

        origin_log = None
        if args.detect_origin:
            orig_text = gen_log.get("original_text", "")
            if orig_text:
                orig_sents = sent_tokenize(orig_text)
                if len(orig_sents) > 12:
                    orig_sents = orig_sents[:12]
                orig_result = detect_sentence_level(
                    orig_sents,
                    embedder,
                    args.msig,
                    z_threshold=args.sent_z_threshold,
                    transform=args.transform,
                    transform_k=args.transform_param,
                )
                origin_log = {"sentence_level": orig_result}
                origin_results.append(orig_result["is_watermarked"])

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"watermark_log": watermark_log, "origin_log": origin_log}, f, ensure_ascii=False, indent=4)

    if wm_results:
        tpr = sum(wm_results) / len(wm_results)
        print(f"Watermarked TPR: {tpr:.4f} ({sum(wm_results)}/{len(wm_results)})")
    if origin_results:
        fpr = sum(origin_results) / len(origin_results)
        print(f"Origin FPR: {fpr:.4f} ({sum(origin_results)}/{len(origin_results)})")

    print("Done!")


if __name__ == "__main__":
    main()
