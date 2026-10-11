#!/usr/bin/env python3
"""Persistent offline-vLLM worker for sentence-candidate generation."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

from vllm import LLM, SamplingParams
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

PREFIX = "VLLM_CANDIDATES "
SEED_MASK = (1 << 63) - 1


class _ArithmeticRequest:
    """Inverse-CDF decoder for one point in an arithmetic codebook."""

    def __init__(self, config: dict):
        self.code_value = float(config["code"])
        self.temperature = float(config["temperature"])
        self.top_p = float(config["top_p"])
        self.repetition_penalty = float(config["repetition_penalty"])
        self.code = None

    def __call__(self, prompt_ids: list[int], output_ids: list[int],
                 logits: torch.Tensor) -> torch.Tensor:
        scores = logits.float().clone()
        if self.repetition_penalty != 1.0:
            seen = torch.as_tensor(
                list(set(prompt_ids + output_ids)),
                dtype=torch.long, device=scores.device,
            )
            repeated = scores[seen]
            scores[seen] = torch.where(
                repeated > 0,
                repeated / self.repetition_penalty,
                repeated * self.repetition_penalty,
            )
        scores /= self.temperature
        probabilities = torch.softmax(scores, dim=-1)

        # Most instruct-model nuclei are tiny. topk avoids a full vocabulary
        # sort while the fallback preserves exact top-p behavior when needed.
        support = min(4096, probabilities.numel())
        kept_probabilities, kept_ids = torch.topk(probabilities, support)
        if self.top_p < 1.0:
            if kept_probabilities.sum() < self.top_p:
                kept_probabilities, kept_ids = torch.sort(
                    probabilities, descending=True
                )
            cumulative_before = (
                torch.cumsum(kept_probabilities, dim=0) - kept_probabilities
            )
            keep = cumulative_before < self.top_p
            kept_probabilities = kept_probabilities[keep]
            kept_ids = kept_ids[keep]
        kept_probabilities /= kept_probabilities.sum()

        # Arithmetic coding uses a fixed vocabulary order, not probability
        # rank. Each selected interval is rescaled back to [0, 1).
        order = torch.argsort(kept_ids)
        kept_ids = kept_ids[order]
        kept_probabilities = kept_probabilities[order]
        cumulative = torch.cumsum(kept_probabilities, dim=0)
        if self.code is None:
            self.code = torch.tensor(
                self.code_value, dtype=torch.float64, device=scores.device
            )
        cumulative = cumulative.to(torch.float64)
        selected_i = torch.searchsorted(cumulative, self.code, right=False)
        selected_i = selected_i.clamp_max(cumulative.numel() - 1)
        lower = torch.where(
            selected_i > 0,
            cumulative[(selected_i - 1).clamp_min(0)],
            cumulative.new_zeros(()),
        )
        mass = kept_probabilities[selected_i].to(torch.float64)
        self.code = ((self.code - lower) / mass).clamp_(1e-15, 1.0 - 1e-15)
        selected_id = kept_ids[selected_i]
        logits.fill_(-torch.inf)
        logits[selected_id] = 0.0
        return logits


class ArithmeticLogitsProcessor(AdapterLogitsProcessor):
    """Enable arithmetic decoding only for explicitly marked requests."""

    @classmethod
    def validate_params(cls, params: SamplingParams) -> None:
        config = (params.extra_args or {}).get("arithmetic_sampling")
        if config is not None and not 0.0 <= float(config["code"]) < 1.0:
            raise ValueError("arithmetic code must lie in [0, 1)")

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params: SamplingParams):
        config = (params.extra_args or {}).get("arithmetic_sampling")
        return _ArithmeticRequest(config) if config is not None else None


def _seed(seed: int, stage: int, branch: int = 0) -> int:
    """Small deterministic seed mixer that always fits vLLM's int64 state."""

    return int(
        int(seed)
        + 1_000_003 * int(stage)
        + 97_003 * int(branch)
    ) & SEED_MASK


def _sentence_stops() -> list[str]:
    return [
        ". ", "! ", "? ", ".\n", "!\n", "?\n",
        '." ', '!" ', '?" ', ".' ", "!' ", "?' ",
    ]


def _iid_groups(llm, prompts: list[dict], request: dict) -> list[list[str]]:
    stop_mode = str(request.get("stop_mode", "sentence"))
    stop = None if stop_mode == "native" else _sentence_stops()
    params = [SamplingParams(
        n=int(request["n_candidates"]),
        temperature=float(request["temperature"]),
        top_p=float(request["top_p"]),
        repetition_penalty=float(request.get("repetition_penalty", 1.0)),
        max_tokens=int(request["max_new_tokens"]),
        min_tokens=int(request.get("min_tokens", 0)),
        stop=stop,
        # Recovery requests deliberately run without sentence stop strings.
        # Instruct checkpoints can otherwise emit EOS immediately after a
        # selected prefix, leaving no text for ``first_sentence`` to recover.
        # Keeping EOS disabled only for this explicitly requested native mode
        # does not change the ordinary benchmark distribution.
        ignore_eos=(stop_mode == "native"),
        include_stop_str_in_output=(stop_mode == "sentence"),
        seed=int(seed),
    ) for seed in request["seeds"]]
    outputs = llm.generate(prompts, params, use_tqdm=True)
    return [[candidate.text for candidate in output.outputs]
            for output in outputs]


def _arithmetic_groups(llm, prompts: list[dict], request: dict) -> list[list[str]]:
    """Sample an evenly spaced, randomly shifted arithmetic codebook."""

    total = int(request["n_candidates"])
    stop_mode = str(request.get("stop_mode", "sentence"))
    stop = None if stop_mode == "native" else _sentence_stops()
    flat_prompts: list[dict] = []
    params: list[SamplingParams] = []
    owners: list[int] = []
    for document_i, (prompt, seed) in enumerate(zip(prompts, request["seeds"])):
        offset = float(np.random.default_rng(_seed(seed, 60)).random())
        for candidate_i in range(total):
            code = (offset + candidate_i / total) % 1.0
            flat_prompts.append(prompt)
            owners.append(document_i)
            params.append(SamplingParams(
                n=1,
                # ArithmeticLogitsProcessor applies these settings before
                # inverse-CDF selection; keep vLLM's sampler neutral afterward.
                temperature=1.0,
                top_p=1.0,
                repetition_penalty=1.0,
                max_tokens=int(request["max_new_tokens"]),
                min_tokens=int(request.get("min_tokens", 0)),
                stop=stop,
                ignore_eos=(stop_mode == "native"),
                include_stop_str_in_output=(stop_mode == "sentence"),
                seed=_seed(seed, 61, candidate_i),
                extra_args={"arithmetic_sampling": {
                    "code": code,
                    "temperature": float(request["temperature"]),
                    "top_p": float(request["top_p"]),
                    "repetition_penalty": float(
                        request.get("repetition_penalty", 1.0)
                    ),
                }},
            ))
    outputs = llm.generate(flat_prompts, params, use_tqdm=True)
    groups: list[list[str]] = [[] for _ in prompts]
    for owner, output in zip(owners, outputs):
        groups[owner].append(output.outputs[0].text)
    return groups


def _branch_tokens(output, branches: int, seed: int) -> list[int]:
    """Gumbel-top-k sample distinct first tokens from processed log-probs."""

    completion = output.outputs[0]
    if not completion.logprobs:
        raise RuntimeError("vLLM returned no first-token log probabilities")
    row = completion.logprobs[0]
    choices = [
        (int(token_id), float(value.logprob))
        for token_id, value in row.items()
        if math.isfinite(float(value.logprob))
    ]
    # A low-entropy instruct prompt can leave fewer than ``branches`` finite
    # tokens after the caller's top-p mask.  That is not a generation error:
    # use every available branch and allocate the full candidate budget across
    # them below.  The proposal remains key-independent and deterministic.
    branches = min(int(branches), len(choices))
    if branches < 1:
        raise RuntimeError("vLLM returned no finite first-token choices")
    token_ids = np.asarray([item[0] for item in choices], dtype=np.int64)
    log_probs = np.asarray([item[1] for item in choices], dtype=np.float64)
    rng = np.random.default_rng(_seed(seed, 11))
    uniforms = np.clip(rng.random(log_probs.size), 1e-15, 1.0 - 1e-15)
    gumbels = -np.log(-np.log(uniforms))
    selected = np.argpartition(log_probs + gumbels, -branches)[-branches:]
    selected = selected[np.argsort(
        (log_probs + gumbels)[selected], kind="stable"
    )[::-1]]
    return [int(value) for value in token_ids[selected]]


def _branch_stratified_groups(llm, tokenizer, prompts: list[dict],
                              request: dict) -> list[list[str]]:
    """Build key-independent prefix branches, then complete each branch."""

    if str(request.get("stop_mode", "sentence")) != "sentence":
        raise ValueError("branch-stratified proposals require sentence mode")
    total = int(request["n_candidates"])
    branches = int(request.get("proposal_branches", 8))
    prefix_tokens = int(request.get("proposal_prefix_tokens", 8))
    max_tokens = int(request["max_new_tokens"])
    if branches < 2 or total % branches:
        raise ValueError(
            "branch-stratified n_candidates must be divisible by branches"
        )
    if not 1 <= prefix_tokens < max_tokens:
        raise ValueError("invalid branch prefix token count")
    temperature = float(request["temperature"])
    top_p = float(request["top_p"])
    repetition_penalty = float(request.get("repetition_penalty", 1.0))

    # This one-token pass exposes the processed top-p distribution. Sampling
    # B tokens without replacement from it prevents all B branches from
    # starting along the same high-probability path. The proposal seeds do not
    # contain the watermark key.
    probe_params = [SamplingParams(
        n=1,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_tokens=1,
        min_tokens=1,
        logprobs=1024,
        seed=_seed(seed, 10),
    ) for seed in request["seeds"]]
    probe_outputs = llm.generate(prompts, probe_params, use_tqdm=True)

    branch_prompts: list[dict] = []
    # The final field is the number of completions assigned to this branch.
    branch_meta: list[tuple[int, list[int], int, int]] = []
    for document_i, (prompt, output, base_seed) in enumerate(zip(
        prompts, probe_outputs, request["seeds"]
    )):
        branch_tokens = _branch_tokens(output, branches, int(base_seed))
        effective_branches = len(branch_tokens)
        base_repetitions, remainder = divmod(total, effective_branches)
        if effective_branches != branches:
            print(
                f"branch-stratified: document {document_i} has "
                f"{effective_branches}/{branches} finite first-token branches",
                file=sys.stderr, flush=True,
            )
        for branch_i, token_id in enumerate(branch_tokens):
            first = [int(token_id)]
            branch_prompts.append({
                "prompt_token_ids": list(prompt["prompt_token_ids"]) + first
            })
            branch_meta.append((
                document_i, first, branch_i,
                base_repetitions + int(branch_i < remainder),
            ))

    remaining_prefix = prefix_tokens - 1
    if remaining_prefix:
        prefix_params = [SamplingParams(
            n=1,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=remaining_prefix,
            min_tokens=remaining_prefix,
            ignore_eos=True,
            seed=_seed(request["seeds"][document_i], 20, branch_i),
        ) for document_i, _, branch_i, _ in branch_meta]
        prefix_outputs = llm.generate(
            branch_prompts, prefix_params, use_tqdm=True
        )
        full_prefixes = [
            first + [int(value) for value in output.outputs[0].token_ids]
            for (_, first, _, _), output in zip(branch_meta, prefix_outputs)
        ]
    else:
        full_prefixes = [first for _, first, _, _ in branch_meta]

    completion_prompts = []
    completion_params = []
    for (document_i, _, branch_i, repetitions), prefix in zip(
        branch_meta, full_prefixes
    ):
        completion_prompts.append({
            "prompt_token_ids": (
                list(prompts[document_i]["prompt_token_ids"]) + prefix
            )
        })
        completion_params.append(SamplingParams(
            n=int(repetitions),
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens - len(prefix),
            min_tokens=max(0, int(request.get("min_tokens", 0)) - len(prefix)),
            stop=_sentence_stops(),
            include_stop_str_in_output=True,
            seed=_seed(request["seeds"][document_i], 30, branch_i),
        ))
    completion_outputs = llm.generate(
        completion_prompts, completion_params, use_tqdm=True
    )

    groups: list[list[str]] = [[] for _ in prompts]
    for (document_i, _, _, _), prefix, output in zip(
        branch_meta, full_prefixes, completion_outputs
    ):
        for completion in output.outputs:
            token_ids = prefix + [int(value) for value in completion.token_ids]
            groups[document_i].append(tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ))
    return groups


def _entropy_stratified_groups(llm, tokenizer, prompts: list[dict],
                               request: dict) -> list[list[str]]:
    """Branch at the earliest naturally diverse token, then decode normally."""

    if str(request.get("stop_mode", "sentence")) != "sentence":
        raise ValueError("entropy-stratified proposals require sentence mode")
    total = int(request["n_candidates"])
    wanted_branches = min(int(request.get("proposal_branches", 8)), total)
    probe_tokens = int(request.get("proposal_prefix_tokens", 8))
    max_tokens = int(request["max_new_tokens"])
    temperature = float(request["temperature"])
    top_p = float(request["top_p"])
    repetition_penalty = float(request.get("repetition_penalty", 1.0))
    logprob_gap = float(request.get("proposal_first_token_logprob_gap", 5.0))
    if wanted_branches < 2 or not 1 <= probe_tokens < max_tokens:
        raise ValueError("invalid entropy-stratified branch configuration")

    # This probe uses the benchmark sampler unchanged. Its per-token processed
    # distributions identify where the model itself offers real alternatives.
    probe_params = [SamplingParams(
        n=1,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_tokens=probe_tokens,
        min_tokens=1,
        stop=_sentence_stops(),
        include_stop_str_in_output=True,
        logprobs=1024,
        seed=_seed(seed, 50),
    ) for seed in request["seeds"]]
    probe_outputs = llm.generate(prompts, probe_params, use_tqdm=True)

    special_ids = set(int(value) for value in tokenizer.all_special_ids)
    branch_prompts: list[dict] = []
    # document, visible prefix through branch token, branch number, repetitions
    branch_meta: list[tuple[int, list[int], int, int]] = []
    selected_positions: list[int] = []
    selected_entropies: list[float] = []
    selected_supports: list[int] = []
    fallback_indices: list[int] = []
    for document_i, (prompt, output, base_seed) in enumerate(zip(
        prompts, probe_outputs, request["seeds"]
    )):
        completion = output.outputs[0]
        rows = completion.logprobs or []
        candidates: list[tuple[int, float, list[tuple[int, float]]]] = []
        for position, row in enumerate(rows):
            choices = [
                (int(token_id), float(value.logprob))
                for token_id, value in row.items()
                if int(token_id) not in special_ids
                if math.isfinite(float(value.logprob))
            ]
            if not choices:
                continue
            best = max(value for _token, value in choices)
            choices = [item for item in choices if item[1] >= best - logprob_gap]
            if len(choices) < 2:
                continue
            log_weights = np.asarray([item[1] for item in choices], dtype=np.float64)
            log_weights -= float(np.logaddexp.reduce(log_weights))
            probabilities = np.exp(log_weights)
            entropy = float(-np.dot(probabilities, log_weights))
            candidates.append((position, entropy, choices))
        if not candidates:
            # Some instruction contexts are genuinely deterministic throughout
            # the short probe. Branching cannot create diversity without
            # leaving the configured sampling distribution, so retain the
            # benchmark's ordinary IID proposal for this document.
            fallback_indices.append(document_i)
            continue
        # Prefer the earliest position with enough support and one nat of
        # entropy. Fall back to the highest-entropy position when the model is
        # genuinely more constrained. This rule never sees the watermark key.
        viable = [row for row in candidates
                  if len(row[2]) >= wanted_branches and row[1] >= 1.0]
        position, entropy, choices = (
            viable[0] if viable else max(candidates, key=lambda row: row[1])
        )
        branch_count = min(wanted_branches, len(choices))
        token_ids = np.asarray([item[0] for item in choices], dtype=np.int64)
        log_weights = np.asarray([item[1] for item in choices], dtype=np.float64)
        rng = np.random.default_rng(_seed(int(base_seed), 51, position))
        uniforms = np.clip(rng.random(log_weights.size), 1e-15, 1.0 - 1e-15)
        scores = log_weights - np.log(-np.log(uniforms))
        selected = np.argpartition(scores, -branch_count)[-branch_count:]
        selected = selected[np.argsort(scores[selected], kind="stable")[::-1]]
        before = [int(value) for value in completion.token_ids[:position]]
        repetitions, remainder = divmod(total, branch_count)
        selected_positions.append(position)
        selected_entropies.append(entropy)
        selected_supports.append(branch_count)
        for branch_i, selected_i in enumerate(selected):
            prefix = before + [int(token_ids[selected_i])]
            branch_prompts.append({
                "prompt_token_ids": list(prompt["prompt_token_ids"]) + prefix
            })
            branch_meta.append((
                document_i, prefix, branch_i,
                repetitions + int(branch_i < remainder),
            ))

    completion_params = [SamplingParams(
        n=repetitions,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens - len(prefix),
        min_tokens=max(0, int(request.get("min_tokens", 0)) - len(prefix)),
        stop=_sentence_stops(),
        include_stop_str_in_output=True,
        seed=_seed(request["seeds"][document_i], 52, branch_i),
    ) for document_i, prefix, branch_i, repetitions in branch_meta]
    completion_outputs = (
        llm.generate(branch_prompts, completion_params, use_tqdm=True)
        if branch_prompts else []
    )
    groups: list[list[str]] = [[] for _ in prompts]
    for (document_i, prefix, _branch_i, _repetitions), output in zip(
        branch_meta, completion_outputs
    ):
        for completion in output.outputs:
            token_ids = prefix + [int(value) for value in completion.token_ids]
            groups[document_i].append(tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ))
    if fallback_indices:
        fallback_request = dict(request)
        fallback_request["seeds"] = [
            request["seeds"][index] for index in fallback_indices
        ]
        fallback_groups = _iid_groups(
            llm, [prompts[index] for index in fallback_indices],
            fallback_request,
        )
        for document_i, group in zip(fallback_indices, fallback_groups):
            groups[document_i] = group
    print(
        "entropy-stratified: mean branch position "
        f"{float(np.mean(selected_positions)) + 1:.2f}; entropy "
        f"{float(np.mean(selected_entropies)):.2f}; branches "
        f"{float(np.mean(selected_supports)):.2f}; IID fallbacks "
        f"{len(fallback_indices)}/{len(prompts)}"
        if selected_positions else
        f"entropy-stratified: IID fallbacks {len(fallback_indices)}/{len(prompts)}",
        file=sys.stderr, flush=True,
    )
    return groups


def _selected_token_logprobs(completion) -> list[float]:
    """Processed log-probability of each emitted token, when requested."""

    if not completion.logprobs:
        return []
    values: list[float] = []
    for token_id, row in zip(completion.token_ids, completion.logprobs):
        item = row.get(int(token_id))
        if item is None or not math.isfinite(float(item.logprob)):
            return []
        values.append(float(item.logprob))
    return values


def _hot_first_token_groups(llm, tokenizer, prompts: list[dict],
                            request: dict) -> list[list[str]]:
    """Heat only the opening choice, then continue with the base sampler.

    The opening token is drawn without replacement from a top-k-capped
    distribution at the proposal temperature.  The rest of every candidate
    uses the benchmark's ordinary temperature/top-p/repetition settings.
    Finally, a key-independent mean-log-likelihood ranking removes the tail
    of the proposal pool before SemField sees any candidates.
    """

    if str(request.get("stop_mode", "sentence")) != "sentence":
        raise ValueError("hot-first-token proposals require sentence mode")
    retained = int(request["n_candidates"])
    pool_size = int(request.get("proposal_pool_size", max(32, retained)))
    top_k = int(request.get("proposal_top_k", 50))
    hot_temperature = float(request.get("proposal_temperature", 1.3))
    first_gap = float(request.get("proposal_first_token_logprob_gap", 5.0))
    base_temperature = float(request["temperature"])
    top_p = float(request["top_p"])
    repetition_penalty = float(request.get("repetition_penalty", 1.0))
    max_tokens = int(request["max_new_tokens"])
    if retained < 2 or pool_size < retained:
        raise ValueError("hot-first-token needs pool_size >= retained >= 2")
    if top_k < pool_size:
        raise ValueError("proposal_top_k must cover proposal_pool_size")
    if hot_temperature <= 0.0 or first_gap < 0.0:
        raise ValueError("invalid hot-first-token proposal parameters")

    # Probe at temperature one so differences between returned processed
    # log-probabilities equal differences between the post-penalty logits.
    # Restricting this pass to top-k gives a hard guard against degenerate
    # openings before any hot sampling occurs.
    probe_params = [SamplingParams(
        n=1,
        temperature=1.0,
        top_p=1.0,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        max_tokens=1,
        min_tokens=1,
        logprobs=top_k,
        seed=_seed(seed, 40),
    ) for seed in request["seeds"]]
    probe_outputs = llm.generate(prompts, probe_params, use_tqdm=True)

    branch_prompts: list[dict] = []
    # document, first-token list, branch index, repetitions, base first log-p
    branch_meta: list[tuple[int, list[int], int, int, float]] = []
    effective_counts: list[int] = []
    for document_i, (prompt, output, base_seed) in enumerate(zip(
        prompts, probe_outputs, request["seeds"]
    )):
        completion = output.outputs[0]
        if not completion.logprobs:
            raise RuntimeError("vLLM returned no opening-token log probabilities")
        special_ids = set(int(value) for value in tokenizer.all_special_ids)
        choices = [
            (int(token_id), float(value.logprob))
            for token_id, value in completion.logprobs[0].items()
            if int(token_id) not in special_ids
            if math.isfinite(float(value.logprob))
        ]
        if not choices:
            raise RuntimeError("vLLM returned no finite opening-token choices")
        token_ids = np.asarray([row[0] for row in choices], dtype=np.int64)
        unit_log_weights = np.asarray([row[1] for row in choices], dtype=np.float64)

        # Reconstruct probabilities within the capped support at the ordinary
        # temperature.  The unknown probe normalization cancels before this
        # softmax.  The gap is measured on this base-temperature scale.
        base_logits = unit_log_weights / base_temperature
        base_logits -= float(base_logits.max())
        keep = base_logits >= -first_gap
        token_ids = token_ids[keep]
        base_logits = base_logits[keep]
        if token_ids.size == 0:
            raise RuntimeError("opening-token likelihood guard removed all choices")
        base_log_probs = base_logits - float(np.logaddexp.reduce(base_logits))

        branch_count = min(pool_size, int(token_ids.size))
        rng = np.random.default_rng(_seed(int(base_seed), 41))
        uniforms = np.clip(rng.random(token_ids.size), 1e-15, 1.0 - 1e-15)
        gumbels = -np.log(-np.log(uniforms))
        hot_scores = unit_log_weights[keep] / hot_temperature + gumbels
        selected = np.argpartition(hot_scores, -branch_count)[-branch_count:]
        selected = selected[np.argsort(hot_scores[selected], kind="stable")[::-1]]
        repetitions, remainder = divmod(pool_size, branch_count)
        effective_counts.append(branch_count)
        for branch_i, selected_i in enumerate(selected):
            first = [int(token_ids[selected_i])]
            branch_prompts.append({
                "prompt_token_ids": list(prompt["prompt_token_ids"]) + first
            })
            branch_meta.append((
                document_i, first, branch_i,
                repetitions + int(branch_i < remainder),
                float(base_log_probs[selected_i]),
            ))

    completion_params = [SamplingParams(
        n=int(repetitions),
        temperature=base_temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens - 1,
        min_tokens=max(0, int(request.get("min_tokens", 0)) - 1),
        stop=_sentence_stops(),
        include_stop_str_in_output=True,
        logprobs=1,
        seed=_seed(request["seeds"][document_i], 42, branch_i),
    ) for document_i, _first, branch_i, repetitions, _first_lp in branch_meta]
    completion_outputs = llm.generate(
        branch_prompts, completion_params, use_tqdm=True
    )

    pools: list[list[tuple[float, str]]] = [[] for _ in prompts]
    for (document_i, first, _branch_i, _repetitions, first_lp), output in zip(
        branch_meta, completion_outputs
    ):
        for completion in output.outputs:
            continuation_lps = _selected_token_logprobs(completion)
            likelihood = (
                float((first_lp + sum(continuation_lps))
                      / (1 + len(continuation_lps)))
                if continuation_lps else -math.inf
            )
            token_ids = first + [int(value) for value in completion.token_ids]
            text = tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            pools[document_i].append((likelihood, text))

    groups: list[list[str]] = []
    for pool in pools:
        # This is a guard, not a watermark decision: it never sees the key or
        # semantic field. Stable sorting makes equal-likelihood behavior fully
        # reproducible. Incomplete sentences are still removed by the shared
        # first_sentence parser in the parent process.
        ordered = sorted(pool, key=lambda row: row[0], reverse=True)
        groups.append([text for _score, text in ordered[:retained]])
    print(
        "hot-first-token: retained "
        f"{retained}/{pool_size}; mean distinct opening branches "
        f"{float(np.mean(effective_counts)):.2f}",
        file=sys.stderr, flush=True,
    )
    return groups


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-model-len", type=int, default=560)
    parser.add_argument("--max-num-seqs", type=int, default=192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()
    backend_name = os.environ.get("SEMFIELD_VLLM_ATTENTION_BACKEND", "FLASHINFER")
    try:
        attention_backend = AttentionBackendEnum[backend_name]
    except KeyError as exc:
        choices = ", ".join(member.name for member in AttentionBackendEnum)
        raise ValueError(
            f"Unknown SEMFIELD_VLLM_ATTENTION_BACKEND={backend_name!r}; "
            f"choose one of: {choices}"
        ) from exc
    llm = LLM(
        model=args.model, dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=True, trust_remote_code=True,
        attention_backend=attention_backend,
        # Branch-stratified mode reads the distribution after temperature,
        # penalties and top-p masking. Ordinary IID requests do not request
        # log-probabilities and are unchanged by these engine settings.
        logprobs_mode="processed_logprobs", max_logprobs=1024,
        logits_processors=[ArithmeticLogitsProcessor],
    )
    tokenizer = llm.get_tokenizer()
    tokenizer.truncation_side = "left"
    print("VLLM_CANDIDATE_READY", flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("command") == "close":
            break
        prompts = []
        for context in request["contexts"]:
            raw_ids = tokenizer.encode(context, add_special_tokens=False)
            # A pre-rendered chat prompt may already carry a BOS token. Avoid
            # adding a second one, while retaining the normal BOS behavior for
            # plain base-model prompts.
            if (raw_ids and tokenizer.bos_token_id is not None
                    and raw_ids[0] == tokenizer.bos_token_id):
                ids = raw_ids
            else:
                ids = tokenizer.encode(context, add_special_tokens=True)
            ids = ids[-int(request["max_context_tokens"]):]
            prompts.append({"prompt_token_ids": ids})
        stop_mode = str(request.get("stop_mode", "sentence"))
        if stop_mode not in {"sentence", "native"}:
            response = {"id": request["id"],
                        "error": f"unsupported stop_mode {stop_mode!r}"}
            print(PREFIX + json.dumps(response), flush=True)
            continue
        try:
            proposal_mode = str(request.get("proposal_mode", "iid"))
            if proposal_mode == "iid":
                groups = _iid_groups(llm, prompts, request)
            elif proposal_mode == "arithmetic":
                groups = _arithmetic_groups(llm, prompts, request)
            elif proposal_mode == "branch-stratified":
                groups = _branch_stratified_groups(
                    llm, tokenizer, prompts, request
                )
            elif proposal_mode == "entropy-stratified":
                groups = _entropy_stratified_groups(
                    llm, tokenizer, prompts, request
                )
            elif proposal_mode == "hot-first-token":
                groups = _hot_first_token_groups(
                    llm, tokenizer, prompts, request
                )
            else:
                raise ValueError(f"unsupported proposal mode {proposal_mode!r}")
            response = {"id": request["id"], "groups": groups}
        except BaseException as exc:
            response = {"id": request["id"],
                        "error": f"{type(exc).__name__}: {exc}"}
        print(PREFIX + json.dumps(response, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
