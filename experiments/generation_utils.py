"""Generation helpers shared by active sentence-level watermark experiments."""
from __future__ import annotations

from semfield.corpora import load_documents, split_sentences
from semfield.generation import sample_candidate_sentences


def build_prompts(corpus_dir: str, n_documents: int,
                  prompt_sentences: int = 3) -> tuple[list[str], list[str], str]:
    """Return matched prompts and held-out human continuations."""
    documents, source = load_documents(
        corpus_dir=corpus_dir, max_documents=max(3 * n_documents, 32)
    )
    prompts, references = [], []
    for text in documents:
        sentences = split_sentences(text)
        if len(sentences) < prompt_sentences + 8:
            continue
        prompts.append(" ".join(sentences[:prompt_sentences]))
        references.append(" ".join(sentences[prompt_sentences:]))
        if len(prompts) == n_documents:
            break
    if len(prompts) != n_documents:
        raise RuntimeError(
            f"requested {n_documents} documents but found only "
            f"{len(prompts)} usable prompts"
        )
    return prompts, references, source


def sample_completed(model, tokenizer, contexts: list[str], target: int,
                     *, seed: int, max_new_tokens: int,
                     max_context_tokens: int, temperature: float,
                     top_p: float, repetition_penalty: float, max_rows: int | None,
                     max_retry_rounds: int,
                     sample_ids: list[int] | None = None,
                     min_tokens: int = 0,
                     stop_mode: str = "sentence",
                     proposal_mode: str = "iid",
                     proposal_branches: int = 8,
                     proposal_prefix_tokens: int = 8,
                     proposal_pool_size: int = 32,
                     proposal_temperature: float = 1.3,
                     proposal_top_k: int = 50,
                     proposal_first_token_logprob_gap: float = 5.0,
                     ) -> tuple[list[list[str]], dict]:
    """Sample ``target`` completed sentences at every unchanged context."""
    collected: list[list[str]] = [[] for _ in contexts]
    if sample_ids is None:
        sample_ids = list(range(len(contexts)))
    if len(sample_ids) != len(contexts):
        raise ValueError("sample_ids must align with contexts")
    pending = list(range(len(contexts)))
    requested = completed = rounds = 0
    while pending and rounds < max_retry_rounds:
        round_contexts = [contexts[index] for index in pending]
        groups = sample_candidate_sentences(
            model, tokenizer, round_contexts, n_candidates=target,
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, repetition_penalty=repetition_penalty, max_rows=max_rows,
            max_context_tokens=max_context_tokens,
            seed=seed + rounds * 1_000_003,
            sample_ids=[sample_ids[index] for index in pending],
            min_tokens=min_tokens,
            stop_mode=stop_mode,
            proposal_mode=proposal_mode,
            proposal_branches=proposal_branches,
            proposal_prefix_tokens=proposal_prefix_tokens,
            proposal_pool_size=proposal_pool_size,
            proposal_temperature=proposal_temperature,
            proposal_top_k=proposal_top_k,
            proposal_first_token_logprob_gap=proposal_first_token_logprob_gap,
        )
        requested += len(round_contexts) * (
            proposal_pool_size
            if proposal_mode == "hot-first-token" else target
        )
        next_pending = []
        for original_index, group in zip(pending, groups):
            completed += len(group)
            collected[original_index].extend(group)
            if len(collected[original_index]) < target:
                next_pending.append(original_index)
        pending = next_pending
        rounds += 1
    for index in range(len(collected)):
        collected[index] = collected[index][:target]
    return collected, {
        "requested": requested,
        "completed": completed,
        "rounds": rounds,
        "failed_indices": pending,
    }
