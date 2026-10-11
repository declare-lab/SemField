#!/usr/bin/env python3
"""Persistent offline-vLLM worker for full-document rewrite attacks."""
from __future__ import annotations

import argparse
import json
import re
import sys

from vllm import LLM, SamplingParams
from vllm.v1.attention.backends.registry import AttentionBackendEnum

RESULT_PREFIX = "VLLM_RESULT "


def alimark_prompt(text: str) -> str:
    return (
        "Please rewrite the following text, avoiding the use of same words "
        "or phrases as the original text as much as possible. You are able "
        "to merge or split sentences but must preserve the original meaning. "
        "Return only the rewritten text, without commentary or analysis:\n"
        f"{text}"
    )


def clean_answer(value: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", value,
                  flags=re.IGNORECASE | re.DOTALL).strip()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    engine = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_input_tokens + args.max_new_tokens,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=True,
        trust_remote_code=True,
        attention_backend=AttentionBackendEnum.FLASHINFER,
    )
    tokenizer = engine.get_tokenizer()
    print("VLLM_READY", flush=True)
    for raw in sys.stdin:
        if not raw.strip():
            continue
        request = json.loads(raw)
        if request.get("command") == "close":
            break
        request_id = request["request_id"]
        prompts = []
        for text in request["texts"]:
            messages = [{
                "role": "user",
                "content": [{"type": "text", "text": alimark_prompt(text)}],
            }]
            token_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                enable_thinking=False,
            )["input_ids"]
            # Match the previous Transformers path: retain the rightmost input
            # tokens when a document exceeds the configured context allowance.
            token_ids = token_ids[-args.max_input_tokens:]
            prompts.append({"prompt_token_ids": token_ids})
        sampling = [SamplingParams(
            temperature=float(request["temperature"]),
            top_p=float(request["top_p"]), top_k=int(request["top_k"]),
            presence_penalty=float(request["presence_penalty"]),
            max_tokens=args.max_new_tokens, seed=int(seed),
        ) for seed in request["seeds"]]
        if len(sampling) != len(prompts):
            raise ValueError("one deterministic seed is required per prompt")
        try:
            outputs = engine.generate(prompts, sampling, use_tqdm=True)
            answers = [clean_answer(item.outputs[0].text) for item in outputs]
            response = {"request_id": request_id, "answers": answers}
        except BaseException as exc:
            response = {"request_id": request_id,
                        "error": f"{type(exc).__name__}: {exc}"}
        print(RESULT_PREFIX + json.dumps(response, ensure_ascii=False),
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
