"""Sentence encoders, and why the choice of encoder is a robustness axis.

The design effect is

    rho = sum_b n_b^2 / T,

and by the Cauchy-Schwarz inequality the effective sample size T / rho is at
most M, the number of distinct regions the document occupies. M is bounded by
the size of the partition, which is a design constant: 2^d for a hyperplane
map, K for a cluster map. No encoder changes that bound.

So the encoder cannot decide whether the problem exists. It can only decide
how large rho is within the range [T/M, T] that the bound already fixes.

Measured, the stronger encoder is the worse one. Qwen3-0.6B embeddings sit in
a much tighter cone than MiniLM's -- mean pairwise cosine 0.23 to 0.30 against
0.05 to 0.15 -- and a tighter cone is cut more unevenly by hyperplanes through
the origin, so rho rises: 6.6 to 9.7 on c4-realnewslike, 19.3 to 29.3 on
pubmed, 535 to 968 on booksum. The effect is specific to hyperplane maps.
Cluster maps are flat or slightly better, because centroids move to where the
mass actually is.

This makes the finding stronger than the bound alone. It is not merely that no
encoder escapes the ceiling; the best encoder available makes calibration
worse, so the problem cannot be engineered away by improving the embedder.

This is why the encoder should be treated as something to sweep, not something
to tune. If rho > 1 for every encoder from a 22-million-parameter model up to
the strongest available, the finding does not depend on the encoder, and a
reviewer cannot dismiss it as an artefact of a weak one.

One property to watch is anisotropy. Sentence embeddings tend to occupy a
narrow cone rather than filling the space, so hyperplanes through the origin
cut them very unevenly. anisotropy() below measures this, and it is the
mechanism that connects the encoder to rho.

Encoding is the whole cost of experiment 1: the corpora run to hundreds of
thousands of sentences and everything downstream is linear in that count with a
small constant. Four things here keep it from dominating.

  device      A GPU is used when one is present, and every visible GPU when
              there is more than one. The old default of cpu was a factor of
              forty on its own, and it is the only change here that is worth
              an order of magnitude.
  dtype       Per model, and only where it pays. See PRECISION below.
  length cap  The Qwen3 encoders advertise a 32768-token window, and
              sentence-transformers pads each batch to its longest member, so
              one pathological line drags a whole batch with it. Sentences do
              not need it.
  cache       Embeddings are memoised to disk under a key that covers the
              model, the width and the exact sentence list, so a re-run with
              different statistics settings costs nothing.

PRECISION. Half precision is not free here, because the cell map reads the
*sign* of a projection. A sentence sitting within rounding distance of a
hyperplane can change cell, which moves rho by a per-cent or so. That is far
below the effect being measured -- rho runs from 2 to 23 against a null of 1 --
but it is a real perturbation, so it is only spent where it buys something.
Measured on this hardware, encoding 646k sentences:

    minilm      fp32  0.8 min    fp16  0.6 min     1.4x   -> stay in fp32
    mpnet       fp32  4.6 min    fp16  1.9 min     2.4x   -> fp16
    qwen3-0.6b  fp32 19.4 min    bf16  6.7 min     2.9x   -> bf16

The choice of *which* half format follows the checkpoint. MiniLM and MPNet are
fp32 encoder-only checkpoints with small, well-behaved activations, and fp16
holds them to a cosine of 0.99975 against fp32 where bf16 manages only 0.9980,
so fp16 is both the more faithful and the equally fast option. The Qwen3
encoders are Qwen3ForCausalLM decoder backbones released as bfloat16, and a
decoder's activation outliers are what fp16's narrow exponent range is bad at,
so they run in the format they were trained in. Pass precision= to override
any of this; "fp32" everywhere reproduces a CPU run to 3e-07.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from . import _hf_env  # noqa: F401  must precede any huggingface import

__all__ = ["EMBEDDERS", "load_embedder", "embed", "anisotropy",
           "resolve_devices", "cache_path"]


# Ordered from cheapest to most expensive. The "cpu_ok" flag marks models that
# are practical without a GPU for a few thousand sentences. "dtype" is the
# precision the backbone was trained in, which is what it should be run in.
EMBEDDERS: dict[str, dict] = {
    "minilm": {
        "id": "sentence-transformers/all-MiniLM-L6-v2",
        "params": "22M", "dim": 384, "cpu_ok": True, "dtype": "fp32",
        "batch": 2048,
        "note": "cheap baseline; run this first",
    },
    "mpnet": {
        "id": "sentence-transformers/all-mpnet-base-v2",
        "params": "110M", "dim": 768, "cpu_ok": True, "dtype": "fp16",
        "batch": 1024,
        "note": "the backbone SemStamp fine-tunes",
    },
    "gte-base": {
        "id": "thenlper/gte-base",
        "params": "110M", "dim": 768, "cpu_ok": True, "dtype": "fp16",
        "batch": 1024,
        "note": "strong mid-size retrieval encoder",
    },
    "bge-base": {
        "id": "BAAI/bge-base-en-v1.5",
        "params": "110M", "dim": 768, "cpu_ok": True, "dtype": "fp16",
        "batch": 1024,
        "note": "strong 768D encoder; matched-width ablation",
    },
    "bge-large": {
        "id": "BAAI/bge-large-en-v1.5",
        "params": "335M", "dim": 1024, "cpu_ok": True, "dtype": "fp16",
        "batch": 512,
        "note": "slower on cpu but workable",
    },
    "qwen3-0.6b": {
        "id": "Qwen/Qwen3-Embedding-0.6B",
        "params": "0.6B", "dim": 1024, "cpu_ok": True, "dtype": "bf16",
        "batch": 512,
        "note": "output dim is adjustable from 32 to 1024; cpu-feasible",
    },
    "qwen3-4b": {
        "id": "Qwen/Qwen3-Embedding-4B",
        "params": "4B", "dim": 2560, "cpu_ok": False, "dtype": "bf16",
        "batch": 256,
        "note": "needs a gpu in practice",
    },
    "qwen3-8b": {
        "id": "Qwen/Qwen3-Embedding-8B",
        "params": "8B", "dim": 4096, "cpu_ok": False, "dtype": "bf16",
        "batch": 128,
        "note": "top of the MTEB multilingual board; ~16GB in bf16, gpu only",
    },
}

# A ceiling, not a setting: it is applied with min() against whatever the model
# already asks for, so it never shortens a model's own window. MiniLM (256) and
# MPNet (384) are untouched; the Qwen3 encoders advertise 32768, which is a
# document-scale window offered to a sentence-scale job, and one 1689-token
# line from a failed sentence split would otherwise pad its whole batch out to
# match. 512 is chosen because it truncates nothing that is really a sentence:
# it is above the longest sentence in c4-realnewslike, and it clips 0.002% of
# booksum, all of them splitter artefacts. Setting this lower is not free --
# at 128 it clips 0.43% of booksum, which is enough to move the k-means
# centroids and shift rho in the second digit.
MAX_SEQ_LENGTH = 512


def resolve_devices(device: str | None = None) -> list[str]:
    """Which devices to encode on.

    None means "use everything there is": every visible GPU, or the CPU if
    there is none. An explicit string is honoured as given, and "cuda:all"
    asks for every GPU by name.
    """
    import torch

    if device not in (None, "auto", "cuda:all"):
        return [device]
    if not torch.cuda.is_available():
        return ["cpu"]
    n = torch.cuda.device_count()
    return [f"cuda:{i}" for i in range(n)]


def _torch_dtype(spec: dict | None, device: str, precision: str = "auto"):
    """The precision to run in: see PRECISION in the module docstring.

    "auto" takes the per-model choice from EMBEDDERS, which is fp32 where half
    precision buys less than about 1.5x, fp16 for the fp32 encoder-only
    checkpoints, and bf16 for the Qwen3 decoder backbones that were released in
    it. Anything else is honoured as asked. The cpu always gets fp32, because
    half precision there is emulated and slower, not faster.
    """
    import torch

    if not device.startswith("cuda"):
        return torch.float32
    name = precision if precision != "auto" else (spec or {}).get("dtype", "fp16")
    if name == "fp32":
        return torch.float32
    if name == "bf16":
        if not torch.cuda.is_bf16_supported():
            return torch.float16
        return torch.bfloat16
    return torch.float16


def load_embedder(name: str, output_dim: int | None = None,
                  device: str | None = None, dtype=None,
                  max_seq_length: int | None = MAX_SEQ_LENGTH,
                  precision: str = "auto"):
    """Load an encoder by short name or by full Hugging Face id.

    output_dim truncates the vector, which the Qwen3 encoders support natively
    because they are trained with nested representations. Truncating is useful
    here for a specific reason: it separates the effect of encoder quality from
    the effect of vector width, so a 4096-dimensional encoder can be compared
    against a 384-dimensional one at matched width.

    device names the device the weights are placed on. Multi-GPU encoding is a
    property of the call, not of the model, so it is decided in embed().
    """
    from sentence_transformers import SentenceTransformer

    spec = EMBEDDERS.get(name)
    model_id = spec["id"] if spec else name
    devices = resolve_devices(device)
    home = devices[0]

    if spec and not spec["cpu_ok"] and home == "cpu":
        print(f"note: {name} ({spec['params']}) is slow without a gpu. "
              f"qwen3-0.6b gives most of the quality at a fraction of the cost.")

    resolved = dtype or _torch_dtype(spec, home, precision)
    kwargs: dict = {"device": home, "model_kwargs": {"dtype": resolved}}
    if output_dim is not None:
        kwargs["truncate_dim"] = output_dim

    model = SentenceTransformer(model_id, **kwargs)
    if max_seq_length is not None:
        model.max_seq_length = min(model.max_seq_length, max_seq_length)
    print(f"  {name}: {str(resolved).removeprefix('torch.')}, "
          f"max_seq_length {model.max_seq_length}")
    return model


def _default_batch(model, name: str | None, device: str) -> int:
    """A batch size that keeps the GPU busy without a hand-tuned flag.

    32, the sentence-transformers default, leaves a modern accelerator almost
    idle on 20-word inputs. The per-model figures are chosen to fill a batch
    without approaching the memory of the smallest card worth running this on.
    """
    if not device.startswith("cuda"):
        return 128
    spec = EMBEDDERS.get(name or "")
    return int(spec["batch"]) if spec else 256


def cache_path(sentences: list[str], model_name: str,
               output_dim: int | None, root: str | Path,
               precision: str = "auto") -> Path:
    """Where the embeddings of exactly this sentence list are memoised.

    The key hashes the sentences themselves, not the corpus name, so a change
    to the sentence splitter or to --max-documents invalidates the entry
    rather than silently reusing a stale one. Precision is in the key too: an
    fp16 run and an fp32 run of the same encoder are different numbers and must
    not be served from each other's entry.
    """
    digest = hashlib.blake2b(digest_size=16)
    digest.update(
        f"{model_name}|{output_dim}|{precision}|{len(sentences)}\0".encode())
    for s in sentences:
        digest.update(s.encode("utf-8", "ignore"))
        digest.update(b"\0")
    return Path(root) / f"{digest.hexdigest()}.npy"


def embed(model, sentences: list[str], batch_size: int | None = None,
          normalize: bool = True, devices: list[str] | None = None,
          model_name: str | None = None, cache_dir: str | Path | None = None,
          output_dim: int | None = None, precision: str = "auto",
          show_progress_bar: bool = True) -> np.ndarray:
    """Encode sentences.

    Documents are encoded without an instruction prefix. The Qwen3 encoders
    accept an instruction on the query side only, and every sentence here plays
    the same role, so no prefix is used. This keeps the comparison across
    encoders like for like.

    When more than one device is given the work is split across them by
    sentence-transformers' own process pool, which tags each chunk, sorts the
    returned chunks by that tag and concatenates them, so row i of the result
    is always sentence i however the work was distributed.

    Row order is exact; the last bits are not. The chunk boundaries depend on
    how many devices there are, so a sentence batches with different
    neighbours, pads to a different width and comes back differing in the final
    ulp. That is the same order of perturbation as the choice of precision and
    it is bounded by it, but it does mean a run is reproducible for a fixed
    device count rather than across all of them. Pass a single device, for
    instance device="cuda:0", when an exactly reproducible run matters more
    than the factor of three.

    Vectors come back as float32 whatever precision the forward pass used.
    Everything downstream reads a sign or a nearest centroid, so float32 is
    ample and it halves the memory of a 646k-sentence corpus.
    """
    devices = devices or resolve_devices(None)
    home = devices[0]

    if cache_dir is not None:
        path = cache_path(sentences, model_name or str(model), output_dim,
                          cache_dir, precision)
        if path.exists():
            print(f"  embeddings read from cache: {path.name}")
            return np.load(path)

    if batch_size is None:
        batch_size = _default_batch(model, model_name, home)

    # A co-tenant can leave too little free memory for the temporary attention
    # matrix even when the encoder itself fits. Retry the complete encode with
    # a smaller batch; this changes only padding/batching, never the inputs,
    # model, precision, or output order. The retry is deliberately here rather
    # than in each caller so every experiment gets the same OOM behaviour.
    requested_batch = max(1, int(batch_size))
    current_batch = requested_batch
    while True:
        kwargs = dict(batch_size=current_batch, convert_to_numpy=True,
                      normalize_embeddings=normalize,
                      show_progress_bar=show_progress_bar)
        try:
            # A pool costs a few seconds to start, so it only pays on a large
            # corpus. On a retry use one device: a partially initialized pool
            # otherwise keeps its allocations alive while the next attempt
            # starts.
            if len(devices) > 1 and len(sentences) > 20_000 and current_batch == requested_batch:
                print(f"  encoding on {len(devices)} devices: {', '.join(devices)}")
                vectors = model.encode(
                    sentences, device=devices,
                    chunk_size=max(4096, current_batch * 8), **kwargs,
                )
            else:
                vectors = model.encode(sentences, device=home, **kwargs)
            break
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" not in message or current_batch <= 1:
                raise
            next_batch = max(1, current_batch // 2)
            print(
                f"  embedding OOM at batch_size={current_batch}; "
                f"retrying batch_size={next_batch}", flush=True,
            )
            current_batch = next_batch
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    if cache_dir is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, vectors)
        tmp.replace(path)
    return vectors


def anisotropy(vectors: np.ndarray, max_rows: int = 200_000,
               seed: int = 0) -> dict:
    """How tightly the embeddings cluster in one direction.

    Returns three numbers.

      mean_cosine       average cosine similarity between random pairs. Zero
                        would mean the vectors fill the space evenly; values
                        near one mean they lie in a narrow cone.
      first_eigenvalue  share of the total variance taken by the leading
                        principal direction.
      participation     effective number of directions actually used, computed
                        as (sum of eigenvalues)^2 / sum of squared eigenvalues.

    A narrow cone is what makes a hyperplane partition uneven, so these
    quantities are the mechanism linking the encoder to the design effect.

    The spectrum is taken from the d-by-d Gram matrix rather than from a full
    SVD of the n-by-d matrix. The eigenvalues of X'X are exactly the squared
    singular values of X, so this is the same answer, but it is one BLAS
    product plus a tiny eigendecomposition instead of a decomposition of a
    646000-row matrix. Rows are subsampled above max_rows because these are
    descriptive statistics of a distribution, and a 200000-row sample fixes
    them well past the three digits reported.
    """
    v = np.asarray(vectors)
    if v.dtype != np.float32 and v.dtype != np.float64:
        v = v.astype(np.float32)

    rng = np.random.default_rng(seed)
    if v.shape[0] > max_rows:
        v = v[rng.choice(v.shape[0], max_rows, replace=False)]

    v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

    n = v.shape[0]
    idx_a = rng.integers(0, n, size=min(20000, n * 4))
    idx_b = rng.integers(0, n, size=idx_a.size)
    keep = idx_a != idx_b
    mean_cos = float((v[idx_a[keep]] * v[idx_b[keep]]).sum(axis=1).mean())

    centred = (v - v.mean(axis=0, keepdims=True)).astype(np.float64)
    gram = centred.T @ centred                      # d x d, one gemm
    eigenvalues = np.linalg.eigvalsh(gram)[::-1]    # = squared singular values
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = eigenvalues.sum()
    return {
        "mean_cosine": mean_cos,
        "first_eigenvalue": float(eigenvalues[0] / total),
        "participation": float(total**2 / np.square(eigenvalues).sum()),
    }
