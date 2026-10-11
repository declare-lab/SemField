"""Generate resumable full-rewrite attacks using AliMark's released protocols.

Unlike the older partial-paraphrase experiment, every input document is
rewritten.  ``pegasus`` and ``parrot`` rewrite every sentence; ``dipper``,
``gpt35``, and ``llm-rewrite`` receive the complete document.  The generation JSON schema is shared
with the existing SemStamp and SemField robustness evaluators.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
import random
import re
import subprocess
import signal
import sys
import tempfile

import torch
from itertools import groupby
from pathlib import Path
from string import punctuation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LABEL_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("inputs must be LABEL=PATH")
    label, raw = value.split("=", 1)
    path = Path(raw).expanduser().resolve()
    if not LABEL_RE.fullmatch(label) or not path.is_file():
        raise argparse.ArgumentTypeError(f"invalid input {value!r}")
    return label, path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False,
                      allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_documents(path: Path, limit: int | None) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    documents = payload.get("documents")
    if not isinstance(documents, list):
        raise ValueError(f"{path} has no documents list")
    output = []
    for position, row in enumerate(documents[:limit]):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: document {position} is not an object")
        sentences = row.get("sentences")
        if not isinstance(sentences, list):
            text = str(row.get("text", ""))
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text)
        sentences = [str(value).strip() for value in sentences
                     if str(value).strip()]
        if not sentences:
            # Exclude failed generations rather than sending empty rows to Pegasus.
            print(f"warning: {path}: skipping empty document {row.get('document_id', position)}", file=sys.stderr, flush=True)
            continue
        output.append({
            "document_id": int(row.get("document_id", position)),
            "prompt": str(row.get("prompt", "")),
            "sentences": sentences,
            "text": " ".join(sentences),
        })
    return output


def clean_text(value: str) -> str:
    excluded = set(punctuation) - {"."}
    excluded.add("\n")
    chunks: list[str] = []
    for key, values in groupby(value):
        if key in excluded:
            chunks.append(key)
        else:
            chunks.extend(values)
    return "".join(chunks)


def well_formed(value: str, end_sentence: bool = False) -> str:
    value = value.strip()
    if value:
        value = value[0].upper() + value[1:]
    value = value.replace("  ", " ").replace(" i ", " I ")
    if end_sentence and value and value[-1] not in "!.?":
        value += "."
    return clean_text(value).strip()


def torch_dtype(name: str):
    import torch
    return {"fp16": torch.float16, "bf16": torch.bfloat16,
            "fp32": torch.float32}[name]


def alimark_llm_prompt(text: str) -> str:
    """AliMark's full-document rewrite instruction, without API wording."""
    return (
        "Please rewrite the following text, avoiding the use of same words "
        "or phrases as the original text as much as possible. You are able "
        "to merge or split sentences but must preserve the original meaning. "
        "Return only the rewritten text, without commentary or analysis:\n"
        f"{text}"
    )


def strip_thinking(value: str) -> str:
    """Defensive cleanup; LLM rewrite also disables thinking in its chat template."""
    return re.sub(r"<think>.*?</think>\s*", "", value,
                  flags=re.IGNORECASE | re.DOTALL).strip()


class PresencePenaltyLogitsProcessor:
    """OpenAI/Qwen-style presence penalty for local Transformers generation."""

    def __init__(self, penalty: float):
        self.penalty = float(penalty)

    def __call__(self, input_ids, scores):
        if self.penalty == 0:
            return scores
        # Mark every token present in each sequence and subtract once. This is
        # exactly the row-wise unique operation, with constant GPU launches.
        present = scores.new_zeros(scores.shape, dtype=torch.bool)
        present.scatter_(1, input_ids, True)
        scores.sub_(present.to(dtype=scores.dtype), alpha=self.penalty)
        return scores


class Rewriter:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.backend = args.attack
        self.worker = None
        local = not args.allow_download
        if self.backend == "pegasus":
            from transformers import (PegasusForConditionalGeneration,
                                      PegasusTokenizer)
            self.tokenizer = PegasusTokenizer.from_pretrained(
                args.pegasus_model, local_files_only=local)
            self.model = PegasusForConditionalGeneration.from_pretrained(
                args.pegasus_model, local_files_only=local,
                dtype=torch_dtype(args.precision)).to(args.device).eval()
        elif self.backend == "parrot":
            ali_root = ROOT / "third_party" / "AliMark"
            sys.path.insert(0, str(ali_root))
            from paraphraser.semstamp_paraphrasers import SParrot
            self.model = SParrot(model_tag=args.parrot_model, use_gpu=True)
        elif self.backend == "dipper":
            ali_root = ROOT / "third_party" / "AliMark"
            sys.path.insert(0, str(ali_root))
            from paraphraser.evaluation.tools.text_editor import DipperParaphraser
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            tokenizer = T5Tokenizer.from_pretrained(
                args.dipper_tokenizer, local_files_only=local)
            model = T5ForConditionalGeneration.from_pretrained(
                args.dipper_model, local_files_only=local,
                device_map="auto", dtype=torch_dtype(args.precision))
            self.model = DipperParaphraser(
                tokenizer=tokenizer, model=model, lex_diversity=60,
                order_diversity=0, sent_interval=1, max_new_tokens=100,
                do_sample=True, top_p=0.75, top_k=None)
        elif self.backend == "llm-rewrite":
            if args.llm_rewrite_backend == "vllm":
                worker = ROOT / "experiments" / "exp46_vllm_rewrite_worker.py"
                command = [
                    args.vllm_python, str(worker), "--model", args.qwen_model,
                    "--dtype", {"fp16": "float16", "bf16": "bfloat16",
                                "fp32": "float32"}[args.precision],
                    "--max-input-tokens", str(args.qwen_max_input_tokens),
                    "--max-new-tokens", str(args.qwen_max_new_tokens),
                    "--max-num-seqs", str(args.vllm_max_num_seqs),
                    "--gpu-memory-utilization",
                    str(args.vllm_gpu_memory_utilization),
                    "--tensor-parallel-size", str(args.vllm_tensor_parallel_size),
                ]
                worker_env = os.environ.copy()
                # Keep auxiliary Parrot paths in the parent, but prevent a
                # foreign Torch site-packages directory from contaminating
                # the dedicated vLLM environment.
                worker_env["PYTHONPATH"] = os.pathsep.join(
                    p for p in worker_env.get("PYTHONPATH", "").split(os.pathsep)
                    if "site-packages" not in p or "venv-vllm" in p
                )
                # vLLM/FlashInfer may invoke ninja by name during JIT
                # compilation.  Propagate the selected vLLM interpreter's
                # bin directory explicitly, even when the parent attack
                # process runs from the separate rewrite environment.
                worker_env["PATH"] = os.pathsep.join(
                    part for part in (
                        str(Path(args.vllm_python).resolve().parent),
                        worker_env.get("PATH", ""),
                    ) if part
                )
                self.worker = subprocess.Popen(
                    command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    text=True, bufsize=1, start_new_session=True,
                    env=worker_env)
                assert self.worker.stdout is not None
                try:
                    for line in self.worker.stdout:
                        print(line, end="", flush=True)
                        if line.strip() == "VLLM_READY":
                            break
                    else:
                        raise RuntimeError("vLLM rewrite worker exited during startup")
                except BaseException:
                    self.close()
                    raise
                atexit.register(self.close)
                self.request_id = 0
                print(f"LLM rewrite backend: offline vLLM; request chunk: "
                      f"{args.vllm_request_size}; max active sequences: "
                      f"{args.vllm_max_num_seqs}", flush=True)
                return
            from transformers import AutoModelForMultimodalLM, AutoProcessor

            if os.environ.get("USE_LIGER", "1") not in {"0", "false", "False"}:
                try:
                    from liger_kernel.transformers.monkey_patch import (
                        apply_liger_kernel_to_qwen3_5,
                    )
                    # Liger currently has no Qwen3.5 RoPE kernel. RMSNorm and
                    # SwiGLU are supported, so fuse those and retain stock RoPE.
                    apply_liger_kernel_to_qwen3_5(
                        rope=False, rms_norm=True, swiglu=True,
                        fused_linear_cross_entropy=False, cross_entropy=False)
                    print("Qwen rewrite Liger: RMSNorm+SwiGLU enabled; stock RoPE",
                          flush=True)
                except Exception as exc:
                    print(f"Qwen rewrite Liger unavailable ({type(exc).__name__}); "
                          "stock kernels", flush=True)
            self.processor = AutoProcessor.from_pretrained(
                args.qwen_model, local_files_only=local)
            self.processor.tokenizer.padding_side = "left"
            self.processor.tokenizer.truncation_side = "left"
            self.model = AutoModelForMultimodalLM.from_pretrained(
                args.qwen_model, local_files_only=local, device_map="auto",
                dtype=torch_dtype(args.precision), low_cpu_mem_usage=True,
                attn_implementation=args.qwen_attn_implementation).eval()
            print(f"Qwen rewrite attention: {args.qwen_attn_implementation}; "
                  f"batch size: {args.qwen_batch_size}", flush=True)
        else:
            from openai import OpenAI
            key = os.getenv("OPENROUTER_API_KEY")
            if not key:
                raise RuntimeError("gpt35 requires OPENROUTER_API_KEY")
            self.client = OpenAI(base_url="https://openrouter.ai/api/v1",
                                 api_key=key)

    def close(self) -> None:
        """Close the rewrite worker and all of its descendants.

        vLLM can fail during engine initialization after spawning an engine
        process.  Without process-group cleanup, a retry starts a second model
        beside the orphan and immediately runs the GPU out of memory.
        """
        worker = getattr(self, "worker", None)
        if worker is None or worker.poll() is not None:
            return
        try:
            if worker.stdin is not None and not worker.stdin.closed:
                worker.stdin.write(json.dumps({"command": "close"}) + "\n")
                worker.stdin.flush()
                worker.stdin.close()
            worker.wait(timeout=20)
            return
        except BaseException:
            pass
        try:
            os.killpg(os.getpgid(worker.pid), signal.SIGTERM)
            worker.wait(timeout=10)
        except BaseException:
            try:
                os.killpg(os.getpgid(worker.pid), signal.SIGKILL)
            except BaseException:
                pass

    def pegasus(self, text: str) -> str:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
        output = []
        for sentence in sentences:
            encoded = self.tokenizer(
                [sentence], truncation=True, padding="longest",
                return_tensors="pt", max_length=60).to(self.args.device)
            with torch.inference_mode():
                generated = self.model.generate(
                    **encoded, max_length=60, num_beams=10,
                    num_return_sequences=1, temperature=2.0, do_sample=True,
                    repetition_penalty=1.03)
            # AliMark/SemStamp retain the first of the ten sampled beams. The
            # search still runs all ten beams; returning only the top one is
            # bit-identical to decoding generated[0] out of ten, and skips
            # finalising and decoding the nine that were always discarded.
            candidate = self.tokenizer.decode(generated[0],
                                              skip_special_tokens=True)
            output.append(well_formed(candidate))
        return " ".join(value for value in output if value)

    def parrot(self, text: str) -> str:
        from nltk.tokenize import sent_tokenize
        output = []
        for sentence in sent_tokenize(text):
            candidates = self.model.augment(
                input_phrase=sentence, use_gpu=True,
                diversity_ranker="levenshtein", do_diverse=True,
                max_return_phrases=10, max_length=60,
                adequacy_threshold=0.8, fluency_threshold=0.8)
            if not candidates:
                raise RuntimeError("Parrot returned no candidate")
            output.append(well_formed(candidates[0], end_sentence=True))
        return " ".join(output)

    def dipper(self, text: str) -> str:
        return str(self.model.edit(text, "")).strip()

    def gpt35(self, text: str) -> str:
        prompt = alimark_llm_prompt(text)
        response = self.client.chat.completions.create(
            model=self.args.gpt_model,
            messages=[
                {"role": "system", "content":
                 "You are a helpful assistant to rewrite the text."},
                {"role": "user", "content": prompt},
            ],
            temperature=1, max_tokens=512, top_p=1,
            frequency_penalty=0, presence_penalty=0)
        return str(response.choices[0].message.content).strip()

    def llm_rewrite(self, texts: list[str]) -> list[str]:
        if self.args.llm_rewrite_backend == "vllm":
            self.request_id += 1
            request = {
                "request_id": self.request_id, "texts": texts,
                "seeds": self.args._rewrite_seeds,
                "temperature": self.args.qwen_temperature,
                "top_p": self.args.qwen_top_p,
                "top_k": self.args.qwen_top_k,
                "presence_penalty": self.args.qwen_presence_penalty,
            }
            assert self.worker.stdin is not None
            assert self.worker.stdout is not None
            self.worker.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self.worker.stdin.flush()
            for line in self.worker.stdout:
                if not line.startswith("VLLM_RESULT "):
                    print(line, end="", flush=True)
                    continue
                response = json.loads(line[len("VLLM_RESULT "):])
                if response.get("request_id") != self.request_id:
                    raise RuntimeError("vLLM worker response id mismatch")
                if "error" in response:
                    raise RuntimeError(response["error"])
                return [str(value) for value in response["answers"]]
            raise RuntimeError("vLLM rewrite worker exited without a response")
        conversations = [[{
            "role": "user",
            "content": [{"type": "text", "text": alimark_llm_prompt(text)}],
        }] for text in texts]
        encoded = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
            processor_kwargs={
                "truncation": True,
                "max_length": self.args.qwen_max_input_tokens,
                "padding": True,
            },
            enable_thinking=False, preserve_thinking=False)
        device = next(parameter.device for parameter in self.model.parameters()
                      if parameter.device.type != "meta")
        encoded = encoded.to(device)
        prompt_tokens = encoded["input_ids"].shape[-1]
        from transformers import LogitsProcessorList
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded, do_sample=True,
                logits_processor=LogitsProcessorList([
                    PresencePenaltyLogitsProcessor(
                        self.args.qwen_presence_penalty)
                ]),
                max_new_tokens=self.args.qwen_max_new_tokens,
                temperature=self.args.qwen_temperature,
                top_p=self.args.qwen_top_p, top_k=self.args.qwen_top_k,
                repetition_penalty=1.0,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id)
        answers = self.processor.batch_decode(
            generated[:, prompt_tokens:], skip_special_tokens=True)
        return [strip_thinking(answer) for answer in answers]

    def __call__(self, text: str) -> str:
        return getattr(self, self.backend)(text)


def set_seed(value: int) -> None:
    random.seed(value)
    try:
        import numpy as np
        import torch
        np.random.seed(value % (2**32))
        torch.manual_seed(value)
    except ImportError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True,
                        metavar="LABEL=PATH")
    parser.add_argument("--cell-reference", action="append", default=[],
                        metavar="LABEL=GENERATION_JSON")
    parser.add_argument("--attack", required=True,
                        choices=["pegasus", "parrot", "dipper", "gpt35", "llm-rewrite"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--max-tries", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"],
                        default="bf16")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pegasus-model", default="tuner007/pegasus_paraphrase")
    parser.add_argument("--parrot-model",
                        default="prithivida/parrot_paraphraser_on_T5")
    parser.add_argument("--dipper-tokenizer", default="google/t5-v1_1-xxl")
    parser.add_argument("--dipper-model",
                        default="kalpeshk2011/dipper-paraphraser-xxl")
    parser.add_argument("--gpt-model", default="gpt-3.5-turbo")
    parser.add_argument("--qwen-model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--qwen-max-input-tokens", type=int, default=4096)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=1024)
    parser.add_argument("--qwen-temperature", type=float, default=0.7)
    parser.add_argument("--qwen-top-p", type=float, default=0.8)
    parser.add_argument("--qwen-top-k", type=int, default=20)
    parser.add_argument("--qwen-presence-penalty", type=float, default=1.5)
    parser.add_argument("--qwen-batch-size", type=int, default=16)
    parser.add_argument("--qwen-attn-implementation",
                        choices=["flash_attention_2", "sdpa", "eager"],
                        default="flash_attention_2")
    parser.add_argument("--llm-rewrite-backend",
                        choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--vllm-python",
                        default=sys.executable)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=32)
    parser.add_argument("--vllm-request-size", type=int, default=256)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float,
                        default=0.92)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()
    if args.max_tries < 1:
        parser.error("--max-tries must be positive")
    if args.num_shards < 1:
        parser.error("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must lie in [0, --num-shards)")
    if args.qwen_batch_size < 1:
        parser.error("--qwen-batch-size must be positive")
    if args.vllm_max_num_seqs < 1:
        parser.error("--vllm-max-num-seqs must be positive")
    if args.vllm_request_size < 1:
        parser.error("--vllm-request-size must be positive")
    if not 0 < args.vllm_gpu_memory_utilization < 1:
        parser.error("--vllm-gpu-memory-utilization must be between 0 and 1")
    if (args.attack == "llm-rewrite" and args.llm_rewrite_backend == "vllm"
            and not Path(args.vllm_python).is_file()):
        parser.error(f"vLLM Python does not exist: {args.vllm_python}")

    inputs = dict(named_path(value) for value in args.input)
    references = dict(named_path(value) for value in args.cell_reference)
    if len(inputs) != len(args.input) or len(references) != len(args.cell_reference):
        parser.error("duplicate input/cell-reference label")
    if not set(references).issubset(inputs):
        parser.error("cell-reference label has no matching input")

    rewriter = Rewriter(args)
    from nltk.tokenize import sent_tokenize
    from experiments.exp24_generate_paraphrases import _cell_encoder
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for label, source in inputs.items():
        documents = load_documents(source, args.max_documents)
        documents = [
            row for row in documents
            if int(row["document_id"]) % args.num_shards == args.shard_index
        ]
        reference = references.get(label)
        contract = {
            "protocol": "AliMark-release-full-rewrite-v1",
            "attack": args.attack,
            "label": label,
            "source": str(source),
            "source_sha256": sha256(source),
            "cell_reference": None if reference is None else str(reference),
            "cell_reference_sha256": None if reference is None else sha256(reference),
            "max_documents": args.max_documents,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "seed": args.seed,
            "models": {"pegasus": args.pegasus_model,
                       "parrot": args.parrot_model,
                       "dipper": args.dipper_model,
                       "gpt35": args.gpt_model},
        }
        if args.attack == "llm-rewrite":
            contract["models"]["llm-rewrite"] = args.qwen_model
            contract["llm-rewrite"] = {
                "non_thinking": True,
                "prompt": "AliMark full-document rewrite",
                "max_input_tokens": args.qwen_max_input_tokens,
                "max_new_tokens": args.qwen_max_new_tokens,
                "temperature": args.qwen_temperature,
                "top_p": args.qwen_top_p,
                "top_k": args.qwen_top_k,
                "presence_penalty": args.qwen_presence_penalty,
                "batch_size": args.qwen_batch_size,
                "attention": args.qwen_attn_implementation,
                "liger": (os.environ.get("USE_LIGER", "1") not in
                          {"0", "false", "False"}
                          if args.llm_rewrite_backend == "transformers" else False),
                "backend": args.llm_rewrite_backend,
            }
            if args.llm_rewrite_backend == "vllm":
                contract["llm-rewrite"]["vllm"] = {
                    "python": args.vllm_python,
                    "max_num_seqs": args.vllm_max_num_seqs,
                    "request_size": args.vllm_request_size,
                    "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
                    "tensor_parallel_size": args.vllm_tensor_parallel_size,
                    "prefix_caching": True,
                }
        path = output_dir / f"{label}.json"
        journal_path = path.with_suffix(path.suffix + ".partial.jsonl")
        complete: dict[int, dict] = {}
        if path.exists():
            if not args.resume:
                parser.error(f"{path} exists; pass --resume")
            old = json.loads(path.read_text(encoding="utf-8"))
            if old.get("contract") != contract:
                parser.error(f"resume contract differs for {path}")
            complete = {int(row["document_id"]): row
                        for row in old.get("documents", [])}
        if journal_path.exists():
            with journal_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    if entry.get("contract") != contract:
                        parser.error(
                            f"resume journal contract differs for {journal_path}"
                        )
                    row = entry["document"]
                    complete[int(row["document_id"])] = row
        encoder = (_cell_encoder(reference, args.device, None)
                   if reference is not None else None)
        result = {"schema_version": 1,
                  "experiment": "exp35_alimark_full_rewrite",
                  "contract": contract, "status": "partial", "documents": []}
        pending = [(position, document)
                   for position, document in enumerate(documents)
                   if document["document_id"] not in complete]
        skip_failed = {int(value) for value in os.environ.get("SKIP_FAILED_IDS", "").split(",") if value.strip().isdigit()}
        pending = [item for item in pending if item[1]["document_id"] not in skip_failed]
        from tqdm.auto import tqdm
        progress = tqdm(total=len(documents), initial=len(complete),
                        desc=label, unit="doc", dynamic_ncols=True)
        batch_size = (args.vllm_request_size
                      if (args.attack == "llm-rewrite"
                          and args.llm_rewrite_backend == "vllm")
                      else args.qwen_batch_size
                      if args.attack == "llm-rewrite" else 1)
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset + batch_size]
            rewritten_batch: list[str] = []
            error = None
            first_id = batch[0][1]["document_id"]
            for attempt in range(1, args.max_tries + 1):
                args._rewrite_seed = args.seed + 1_000_003 * first_id + attempt
                args._rewrite_seeds = [
                    args.seed + 1_000_003 * document["document_id"] + attempt
                    for _, document in batch
                ]
                if not (args.attack == "llm-rewrite"
                        and args.llm_rewrite_backend == "vllm"):
                    set_seed(args._rewrite_seed)
                try:
                    texts = [document["text"] for _, document in batch]
                    if args.attack == "llm-rewrite":
                        rewritten_batch = [value.strip() for value in
                                           rewriter.llm_rewrite(texts)]
                    else:
                        rewritten_batch = [rewriter(texts[0]).strip()]
                    if (len(rewritten_batch) == len(batch)
                            and all(rewritten_batch)):
                        break
                    error = "rewriter returned an empty or incomplete batch"
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    ids = [document["document_id"] for _, document in batch]
                    print(f"{label} documents {ids} try {attempt}: {error}",
                          flush=True)
            if (len(rewritten_batch) != len(batch)
                    or not all(rewritten_batch)):
                ids = [document["document_id"] for _, document in batch]
                raise RuntimeError(
                    f"{label} documents {ids} failed after "
                    f"{args.max_tries} tries: {error}")
            completed_batch = []
            for (position, document), rewritten in zip(batch, rewritten_batch):
                document_id = document["document_id"]
                sentences = [value.strip() for value in sent_tokenize(rewritten)
                             if value.strip()]
                if not sentences:
                    raise RuntimeError(
                        f"{label} document {document_id} became empty")
                row = {"document_id": document_id,
                       "prompt": document["prompt"],
                       "text": rewritten, "sentences": sentences,
                       "source_sentence_count": len(document["sentences"]),
                       "rewritten_sentence_count": len(sentences)}
                if encoder is not None:
                    row["cells"] = list(encoder(sentences))
                complete[document_id] = row
                completed_batch.append(row)
                progress.update(1)
            # Append only the new rows. This keeps checkpoint I/O linear while
            # preserving exact batch-level resume after interruption.
            with journal_path.open("a", encoding="utf-8") as handle:
                for row in completed_batch:
                    json.dump({"contract": contract, "document": row},
                              handle, ensure_ascii=False, allow_nan=False)
                    handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        progress.close()
        result["status"] = "complete"
        result["documents"] = [complete[key] for key in sorted(complete)]
        atomic_json(path, result)
        journal_path.unlink(missing_ok=True)
        print(f"wrote {path}")
    if (args.attack == "llm-rewrite" and args.llm_rewrite_backend == "vllm"):
        assert rewriter.worker.stdin is not None
        rewriter.worker.stdin.write(json.dumps({"command": "close"}) + "\n")
        rewriter.worker.stdin.flush()
        rewriter.worker.stdin.close()
        rewriter.worker.wait(timeout=30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
