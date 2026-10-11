"""AliMark-compatible teacher-forced perplexity through vLLM."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def load_vllm_judge(name: str, precision: str, max_length: int,
                    gpu_memory_utilization: float = 0.8):
    """Load one vLLM judge and its tokenizer."""
    from vllm import LLM

    dtype = {"fp16": "float16", "bf16": "bfloat16",
             "fp32": "float32", "auto": "auto"}[precision]
    engine = LLM(
        model=name,
        dtype=dtype,
        max_model_len=int(max_length),
        gpu_memory_utilization=float(gpu_memory_utilization),
        enable_prefix_caching=True,
    )
    tokenizer = engine.get_tokenizer()
    resolved = ("bf16" if dtype == "bfloat16" else
                "fp16" if dtype == "float16" else dtype)
    print(f"judge: {name}, vLLM {resolved}, max_length={max_length}")
    return engine, tokenizer, resolved


def _logprob(entry, token_id: int) -> float:
    if entry is None:
        raise ValueError("vLLM returned no prompt logprob for a target token")
    value = entry.get(token_id)
    if value is None:
        value = entry.get(str(token_id))
    if value is None:
        raise ValueError(f"vLLM omitted target token {token_id}")
    value = float(getattr(value, "logprob", value))
    if not math.isfinite(value):
        raise ValueError("vLLM returned a non-finite prompt logprob")
    return value


def score_documents_vllm_alimark(documents: Sequence[dict], engine, tokenizer,
                                 max_length: int, bootstrap: int,
                                 seed: int, batch_size: int = 64) -> dict[str, object]:
    """Match AliMark/SemStamp's single-pass, 2048-token PPL protocol.

    Text is tokenized once, truncated, and scored with one causal loss pass;
    the first token is excluded.  vLLM requires ``max_tokens >= 1``, so one
    generated token is requested and discarded while prompt log-probabilities
    are retained.
    """
    from tqdm.auto import tqdm
    from vllm import SamplingParams

    if max_length < 2 or batch_size < 1:
        raise ValueError("max_length must be >=2 and batch_size positive")
    prepared: list[tuple[int, list[int]]] = []
    for position, document in enumerate(documents):
        text = str(document.get("text", ""))
        if not text.strip():
            continue
        text = text.replace("Paraphrase:", "").replace("paraphrase:", "")
        prefix = (str(getattr(tokenizer, "bos_token", "") or "")
                  if "GPT2Tokenizer" in type(tokenizer).__name__ else "")
        ids = list(tokenizer.encode(prefix + text, add_special_tokens=False))[:max_length]
        if len(ids) >= 2:
            prepared.append((int(document.get("document_id", position)), ids))

    params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1,
                            detokenize=False, skip_special_tokens=True)
    nlls: list[float] = []
    lengths: list[int] = []
    ppls: list[float] = []
    progress = tqdm(total=len(prepared), desc="vLLM AliMark PPL", unit="doc",
                    dynamic_ncols=True)
    for offset in range(0, len(prepared), batch_size):
        batch = prepared[offset:offset + batch_size]
        outputs = engine.generate([ids for _, ids in batch], params,
                                  use_tqdm=False)
        for (document_id, ids), output in zip(batch, outputs):
            nll = sum(-_logprob(output.prompt_logprobs[i], ids[i])
                      for i in range(1, len(ids)))
            count = len(ids) - 1
            nlls.append(float(nll))
            lengths.append(count)
            ppls.append(float(math.exp(nll / count)))
            progress.update(1)
    progress.close()

    total_nll = float(sum(nlls))
    total_tokens = int(sum(lengths))
    corpus_ppl = float(math.exp(total_nll / max(1, total_tokens)))
    document_avg = float(np.mean(ppls)) if ppls else float("nan")
    document_median = float(np.median(ppls)) if ppls else float("nan")
    ci_half = 0.0
    if len(ppls) >= 2:
        from scipy import stats
        ci_half = float(stats.t.ppf(.975, len(ppls) - 1)
                        * np.std(ppls, ddof=1) / math.sqrt(len(ppls)))
    rows = [{"document_id": doc_id, "nll": nll, "tokens": count,
             "mean_nll": nll / count, "perplexity": ppl}
            for (doc_id, _), nll, count, ppl in
            zip(prepared, nlls, lengths, ppls)]
    return {
        "perplexity": corpus_ppl,
        "perplexity_ci95": [document_avg - ci_half, document_avg + ci_half],
        "document_avg_perplexity": document_avg,
        "document_median_perplexity": document_median,
        "scored_documents": len(rows),
        "scored_tokens": total_tokens,
        "max_length": int(max_length),
        "backend": "vllm_prompt_logprobs",
        "protocol": "alimark_single_pass_truncate_2048",
        "per_document": rows,
    }
