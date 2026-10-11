"""Sampling candidate sentences from a language model.

SemField is a tournament: at each step the model proposes n candidate sentences and
the key picks one. Everything the scheme can achieve is bounded by a single
property of the proposal step -- how often the n candidates land in *different*
semantic cells. If they all land in the same cell the key has no choice, the
emitted sentence is whatever the model would have said anyway, and no signal is
written.

That quantity is the Gini coefficient of the model's conditional cell
distribution,

    gini = 1 - sum_b p_b^2,

and it has a direct reading: for two independent candidates, gini is exactly
the probability that they occupy different cells, which is the probability the
tournament has a real choice at all. The detection signal is proportional to
it,

    snr(n) = sqrt(3) * gini * (n - 1) / (n + 1),

so a model that is narrow in context is a model SemField cannot watermark cheaply,
however good the detector is.

Note which distribution this is. Experiment 3 uses the *marginal* spread of
sentences over cells, which is wide because a corpus covers many topics. What
governs the tournament is the *conditional* spread given the preceding text,
which is far narrower: after "The patient was admitted to the" a model has few
plausible continuations and they may well share a cell. This module measures
the conditional distribution, which is the parameter no experiment in this
repository has yet observed.

Only sampling is needed here, not selection. The probe runs without a key and
without a watermark, because the question it answers -- is there enough
conditional spread for a tournament to exploit -- is a property of the model
alone.
"""

from __future__ import annotations

import json
import re
import os
import sys
import subprocess
import atexit
import signal
from pathlib import Path

import numpy as np

from . import _hf_env  # noqa: F401  must precede any huggingface import

__all__ = ["GENERATORS", "default_precision_for_model", "load_generator",
           "sample_candidate_sentences", "first_sentence", "unbiased_gini"]


def default_precision_for_model(model_id: str) -> str:
    """Return the default dtype policy for a model identifier.

    OPT checkpoints retain the benchmark's fp16 behavior.  Other model
    families default to bf16.  Callers may still pass an explicit override.
    """

    return "fp16" if "opt" in str(model_id).lower() else "bf16"


# Base models, not instruction-tuned: the watermarking literature measures
# distortion against the base model's own distribution, and an instruction
# tune has already moved it.
# "rows" records the historical sequence budget used by experiments. vLLM
# now controls active sequences through ``vllm_max_num_seqs``.
GENERATORS: dict[str, dict] = {
    "opt-1.3b": {
        "id": "facebook/opt-1.3b",
        "params": "1.3B", "dtype": "fp16", "rows": 128,
        "note": "what SemStamp used; keep for comparability, not for quality",
    },
    "llama3.2-1b": {
        "id": "meta-llama/Llama-3.2-1B",
        "params": "1.2B", "dtype": "bf16", "rows": 128,
        "note": "base model at OPT-1.3B's scale; the like-for-like generator swap",
    },
    "llama3.2-1b-instruct": {
        "id": "meta-llama/Llama-3.2-1B-Instruct",
        "params": "1.2B", "dtype": "bf16", "rows": 128,
        "note": "instruction-tuned Llama; input prompts must already be "
                "rendered with the checkpoint's chat template",
    },
    "qwen3-1.7b": {
        "id": "Qwen/Qwen3-1.7B-Base",
        "params": "1.7B", "dtype": "bf16", "rows": 128,
        "note": "the Base release, not the instruct/thinking Qwen3-1.7B: this "
                "benchmark continues a prompt, it does not chat",
    },
    "olmo2-1b": {
        "id": "allenai/OLMo-2-0425-1B",
        "params": "1B", "dtype": "bf16", "rows": 128,
        "note": "OLMo 2 1B base model; text-only, fully open checkpoint",
    },
    "qwen3-4b-instruct-2507": {
        "id": "Qwen/Qwen3-4B-Instruct-2507",
        "params": "4B", "dtype": "bf16", "rows": 192,
        "note": "Qwen3 post-trained instruction model; prompts must already "
                "be rendered with its chat template",
    },
}


class VLLMCandidateGenerator:
    """Persistent cross-environment vLLM candidate generator."""

    is_vllm_generator = True

    def __init__(self, model_id: str, *, dtype: str, max_model_len: int,
                 python: str, max_num_seqs: int, request_size: int,
                 gpu_memory_utilization: float, tensor_parallel_size: int):
        worker = (Path(__file__).resolve().parents[1] / "experiments" /
                  "exp47_vllm_candidate_worker.py")
        self.request_size = int(request_size)
        self.max_model_len = int(max_model_len)
        command = [
            python, str(worker), "--model", model_id, "--dtype", dtype,
            "--max-model-len", str(max_model_len),
            "--max-num-seqs", str(max_num_seqs),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--tensor-parallel-size", str(tensor_parallel_size),
        ]
        worker_env = os.environ.copy()
        worker_env["PYTHONPATH"] = os.pathsep.join(
            p for p in worker_env.get("PYTHONPATH", "").split(os.pathsep)
            if "site-packages" not in p or "venv-vllm" in p
        )
        # The vLLM engine starts auxiliary subprocesses (including FlashInfer
        # JIT, which invokes ``ninja`` by name).  Ensure the executable's
        # environment is propagated even when the parent was launched from a
        # different virtual environment or through a scheduler.
        python_bin = str(Path(python).resolve().parent)
        worker_env["PATH"] = os.pathsep.join(
            part for part in (python_bin, worker_env.get("PATH", "")) if part
        )
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1, start_new_session=True, env=worker_env)
        assert self.process.stdout is not None
        try:
            for line in self.process.stdout:
                print(line, end="", flush=True)
                if line.strip() == "VLLM_CANDIDATE_READY":
                    break
            else:
                raise RuntimeError("vLLM candidate worker exited during startup")
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)
        self.request_id = 0

    def sample(self, contexts: list[str], *, n_candidates: int,
               max_new_tokens: int, temperature: float, top_p: float,
               repetition_penalty: float = 1.0,
               max_context_tokens: int, seed: int,
               sample_ids: list[int] | None = None,
               min_tokens: int = 0,
               stop_mode: str = "sentence",
               proposal_mode: str = "iid",
               proposal_branches: int = 8,
               proposal_prefix_tokens: int = 8,
               proposal_pool_size: int = 32,
               proposal_temperature: float = 1.3,
               proposal_top_k: int = 50,
               proposal_first_token_logprob_gap: float = 5.0) -> list[list[str]]:
        output: list[list[str]] = []
        if sample_ids is None:
            sample_ids = list(range(len(contexts)))
        if len(sample_ids) != len(contexts):
            raise ValueError("sample_ids must align with contexts")
        if stop_mode not in {"sentence", "native"}:
            raise ValueError("stop_mode must be 'sentence' or 'native'")
        if proposal_mode not in {
            "iid", "arithmetic", "branch-stratified", "entropy-stratified",
            "hot-first-token",
        }:
            raise ValueError(
                "proposal_mode must be iid, arithmetic, branch-stratified, "
                "entropy-stratified, or hot-first-token"
            )
        if proposal_mode in {"branch-stratified", "entropy-stratified"}:
            if proposal_branches < 2:
                raise ValueError("proposal_branches must be at least two")
            if n_candidates % proposal_branches:
                raise ValueError(
                    "n_candidates must be divisible by proposal_branches"
                )
            if not 1 <= proposal_prefix_tokens < max_new_tokens:
                raise ValueError(
                    "proposal_prefix_tokens must lie in [1, max_new_tokens)"
                )
        if proposal_mode == "hot-first-token":
            if proposal_pool_size < n_candidates:
                raise ValueError(
                    "proposal_pool_size must be at least n_candidates"
                )
            if proposal_top_k < proposal_pool_size:
                raise ValueError("proposal_top_k must cover proposal_pool_size")
            if proposal_temperature <= 0.0:
                raise ValueError("proposal_temperature must be positive")
            if proposal_first_token_logprob_gap < 0.0:
                raise ValueError(
                    "proposal_first_token_logprob_gap must be non-negative"
                )
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        for start in range(0, len(contexts), self.request_size):
            chunk = contexts[start:start + self.request_size]
            self.request_id += 1
            request = {
                "id": self.request_id, "contexts": chunk,
                "n_candidates": n_candidates,
                "max_new_tokens": max_new_tokens,
                "max_context_tokens": max_context_tokens,
                "temperature": temperature, "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "min_tokens": min_tokens,
                "stop_mode": stop_mode,
                "proposal_mode": proposal_mode,
                "proposal_branches": int(proposal_branches),
                "proposal_prefix_tokens": int(proposal_prefix_tokens),
                "proposal_pool_size": int(proposal_pool_size),
                "proposal_temperature": float(proposal_temperature),
                "proposal_top_k": int(proposal_top_k),
                "proposal_first_token_logprob_gap": float(
                    proposal_first_token_logprob_gap
                ),
                # vLLM stores request seeds in a signed int64 array.  The
                # benchmark's SHA-256-derived seeds are unsigned 64-bit, so
                # normalize them deterministically at the API boundary.
                "seeds": [int(seed + 1_000_003 * int(value)) & ((1 << 63) - 1)
                          for value in sample_ids[start:start + len(chunk)]],
            }
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            for line in self.process.stdout:
                if not line.startswith("VLLM_CANDIDATES "):
                    print(line, end="", flush=True)
                    continue
                response = json.loads(line[len("VLLM_CANDIDATES "):])
                if response.get("id") != self.request_id:
                    raise RuntimeError("vLLM candidate response id mismatch")
                if "error" in response:
                    raise RuntimeError(response["error"])
                output.extend(response["groups"])
                break
            else:
                raise RuntimeError("vLLM candidate worker exited without response")
        return output

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        try:
            assert self.process.stdin is not None
            if not self.process.stdin.closed:
                self.process.stdin.write(json.dumps({"command": "close"}) + "\n")
                self.process.stdin.flush()
                self.process.stdin.close()
            self.process.wait(timeout=20)
            return
        except BaseException:
            pass
        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            self.process.wait(timeout=10)
        except BaseException:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            except BaseException:
                pass


def load_generator(name: str, device: str | None = None, dtype=None,
                   liger: bool = True,
                   attn_implementation: str | None = "flash_attention_2",
                   vllm_python: str = sys.executable,
                   vllm_max_num_seqs: int = 192,
                   vllm_request_size: int = 512,
                   vllm_gpu_memory_utilization: float = 0.90,
                   vllm_tensor_parallel_size: int = 1,
                   max_model_len: int = 560):
    """Load a causal language model by short name or full Hugging Face id.

    The worker defaults to this Python interpreter. Pass ``vllm_python`` to
    use a separate environment with vLLM installed. Model weights use the
    standard Hugging Face cache selected by ``HF_HOME``.
    """
    spec = GENERATORS.get(name)
    model_id = spec["id"] if spec else name
    if not Path(vllm_python).is_file():
        raise FileNotFoundError(f"vLLM Python not found: {vllm_python}")
    if dtype is None:
        want = default_precision_for_model(model_id)
    else:
        requested = str(dtype).lower().removeprefix("torch.")
        if requested in {"fp16", "float16"}:
            want = "fp16"
        elif requested in {"bf16", "bfloat16"}:
            want = "bf16"
        else:
            raise ValueError("dtype must be fp16/float16 or bf16/bfloat16")
    resolved = "bfloat16" if want == "bf16" else "float16"
    generator = VLLMCandidateGenerator(
        model_id, dtype=resolved, max_model_len=max_model_len,
        python=vllm_python, max_num_seqs=vllm_max_num_seqs,
        request_size=vllm_request_size,
        gpu_memory_utilization=vllm_gpu_memory_utilization,
        tensor_parallel_size=vllm_tensor_parallel_size)
    print(f"  {name}: {model_id}, {resolved} via offline vLLM")
    return generator, None


# A sentence ends at . ! or ? followed by whitespace. The generator is asked
# for a little more than one sentence and the first complete one is kept, so
# the boundary is decided on text the model actually produced rather than by
# truncating it mid-clause.
_END = re.compile(r"[.!?][\"')\]]*(?=\s|$)")

# A period can close an abbreviation instead of a sentence: "Gen.", "Rep.",
# "Jan.", "U.S.". Common enough in news prose that a prompt or continuation
# regularly ends on one. Skipped when deciding where a sentence closes, so
# that "Gen." isn't mistaken for a one-word sentence and rejected while a
# real sentence two words later is thrown away with it.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "etc",
    "inc", "corp", "co", "ltd", "rep", "sen", "gov", "gen", "lt", "col",
    "capt", "cmdr", "adm", "sgt", "att", "atty", "supt", "det", "rev",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
    "nov", "dec", "no", "ave", "blvd", "u.s", "u.k", "d.c",
}
_LAST_WORD = re.compile(r"([A-Za-z]+(?:\.[A-Za-z]+)*)\.$")


def _ends_on_abbreviation(prefix: str) -> bool:
    found = _LAST_WORD.search(prefix)
    return found is not None and found.group(1).lower() in _ABBREVIATIONS


def first_sentence(text: str, min_words: int = 4) -> str | None:
    """The first complete sentence of a continuation, or None if there is none.

    Returning None matters: a candidate that never closed a sentence within the
    token budget is not a short sentence, it is a failed sample, and counting
    it as one would bias the cell distribution towards whatever truncation
    produces.

    Every "." match is tried in order, skipping ones that close a known
    abbreviation, until one produces a sentence with enough words.
    """
    text = " ".join(text.split())
    for match in _END.finditer(text):
        if text[match.start()] == "." and _ends_on_abbreviation(text[: match.start() + 1]):
            continue
        sentence = text[: match.end()].strip()
        if len(sentence.split()) >= min_words:
            return sentence
    return None



def sample_candidate_sentences(model, tokenizer, contexts: list[str],
                               n_candidates: int, max_new_tokens: int = 48,
                               temperature: float = 1.0, top_p: float = 1.0,
                               repetition_penalty: float = 1.0,
                               max_rows: int | None = None,
                               max_context_tokens: int = 512,
                               seed: int = 0,
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
                               ) -> list[list[str]]:
    """Sample n_candidates continuation sentences for each context.

    min_tokens forbids EOS and the sentence-boundary stop strings below until
    that many tokens are out. It defaults to 0, its old always-on behaviour:
    generation may stop after a single token. A context whose accumulated
    text looks like the end of an article (a "read more" footer, say) can
    make the model emit nothing at all, every time, regardless of sampling;
    forcing a minimum length is what actually gets a real continuation out of
    it. It also helps a colder failure mode -- a continuation that opens on
    an abbreviation ("Gen.", "Rep.") stops at that abbreviation's period
    before the real sentence after it is ever generated, so no amount of
    smarter sentence-boundary detection downstream can recover it.

    Returns one list per context, holding the candidates whose first sentence
    completed. Lists may be shorter than n_candidates, and a context that
    produced fewer than two usable candidates should be dropped by the caller
    rather than padded.

    ``max_rows`` remains in the signature for compatibility with stored
    experiment configurations. vLLM controls concurrency through its own
    ``max_num_seqs`` setting.

    ``stop_mode="native"`` is reserved for recovery callers that need the
    model to finish naturally before :func:`first_sentence` extracts the
    first complete sentence.  Normal experiments use the default
    sentence-boundary stop mode.

    Sampling is done at temperature 1 with no truncation of the tail by
    default, because the quantity being measured is the model's own conditional
    distribution. Lowering the temperature or the nucleus narrows that
    distribution, which is a real effect on the watermark and is worth
    sweeping, but it must not be smuggled in as a default.
    """
    if not getattr(model, "is_vllm_generator", False):
        raise TypeError("candidate generation requires VLLMCandidateGenerator")
    raw = model.sample(
        contexts, n_candidates=n_candidates,
        max_new_tokens=max_new_tokens, temperature=temperature,
        top_p=top_p, repetition_penalty=repetition_penalty,
        max_context_tokens=max_context_tokens, seed=seed,
        sample_ids=sample_ids, min_tokens=min_tokens,
        stop_mode=stop_mode,
        proposal_mode=proposal_mode,
        proposal_branches=proposal_branches,
        proposal_prefix_tokens=proposal_prefix_tokens,
        proposal_pool_size=proposal_pool_size,
        proposal_temperature=proposal_temperature,
        proposal_top_k=proposal_top_k,
        proposal_first_token_logprob_gap=proposal_first_token_logprob_gap)
    return [[sentence for sentence in
             (first_sentence(text) for text in group) if sentence]
            for group in raw]


def unbiased_gini(cell_ids) -> float:
    """Unbiased estimate of gini = 1 - sum_b p_b^2 from a finite sample.

    The plug-in estimator sum_b (c_b/m)^2 is biased upwards for sum p_b^2,
    because a cell's own draw always matches itself, so it reports the model as
    narrower than it is and would understate SemField's signal. The U-statistic

        sum_b c_b (c_b - 1) / (m (m - 1))

    counts ordered pairs of *distinct* draws and is unbiased. With m candidates
    per step the difference is of order 1/m, which at m = 32 is larger than the
    effect being measured, so it is not a detail that can be skipped.
    """
    ids = np.asarray(cell_ids)
    m = ids.size
    if m < 2:
        return float("nan")
    _, counts = np.unique(ids, return_counts=True)
    counts = counts.astype(np.float64)
    collision = (counts * (counts - 1)).sum() / (m * (m - 1))
    return float(1.0 - collision)
