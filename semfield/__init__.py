"""SemField: semantic watermarking benchmark.

``_hf_env`` is imported first and deliberately: it pins HF_HOME and friends to
the data volume, and must run before anything imports huggingface.
"""

from . import _hf_env  # noqa: F401  must precede any huggingface import

__version__ = "0.1.0"
