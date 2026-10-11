"""Set portable defaults for Hugging Face caches.

Import this module before huggingface_hub, transformers, datasets or
sentence_transformers, since those libraries read cache variables at import
time. Defaults follow XDG cache conventions, and explicit environment values
always take precedence.

Layout is the standard one: ``$HF_HOME/hub`` holds model and dataset repos,
``$HF_HOME/datasets`` holds Arrow data, and ``$HF_HOME/token`` holds the
credential.
"""

from __future__ import annotations

import os

_default_cache_root = os.environ.get(
    "XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")
)
HF_HOME = os.environ.setdefault(
    "HF_HOME", os.path.join(_default_cache_root, "huggingface")
)

# SEMFIELD_EMBCACHE remains an explicit override for the embedding memo cache.
EMBEDDING_CACHE = os.environ.get(
    "SEMFIELD_EMBCACHE", os.path.join(HF_HOME, "semfield", "embeddings")
)

os.environ.setdefault("HF_HUB_CACHE", os.path.join(HF_HOME, "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(HF_HOME, "datasets"))
os.environ.setdefault("HF_ASSETS_CACHE", os.path.join(HF_HOME, "assets"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", os.environ["HF_HUB_CACHE"])
os.environ.setdefault("TORCH_HOME", os.path.join(HF_HOME, "torch"))

# Tokenisation happens inside the dataloader workers; the fork warning that
# huggingface/tokenizers prints otherwise is noise.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__all__ = ["HF_HOME", "EMBEDDING_CACHE"]
