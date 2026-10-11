"""Optional, output-equivalent Liger acceleration for inference.

Two paths, chosen by ``config.model_type``:

* Models Liger ships a patcher for (Llama, Qwen3, ...) get the
  official one, which swaps RMSNorm, RoPE and SwiGLU for fused kernels.
* Everything else (OPT, GPT-2 and friends) falls back to replacing plain
  ``torch.nn.LayerNorm`` modules, which is all Liger can offer there.

Both cross-entropy options are forced off. They are training optimisations:
``fused_linear_cross_entropy`` deliberately avoids materialising logits, and
this codebase reads logits both to generate and to score perplexity.

Liger is an accelerator, never a correctness requirement, so nothing here is
allowed to take a run down -- a failed swap prints why and returns the model
untouched.
"""
from __future__ import annotations

import inspect
import os


def _apply_official_patcher(model, model_type):
    """Return True if Liger has a patcher for this model type and it applied."""
    try:
        from liger_kernel.transformers.monkey_patch import (
            MODEL_TYPE_TO_APPLY_LIGER_FN,
        )
    except ImportError:
        return False
    patcher = MODEL_TYPE_TO_APPLY_LIGER_FN.get(model_type)
    if patcher is None:
        return False

    # Kernel sets differ per family (some add layer_norm/geglu, others do
    # not), so only pass what this patcher actually accepts.
    accepted = inspect.signature(patcher).parameters
    kwargs = {"model": model}
    for name, value in (
        ("rope", True),
        ("rms_norm", True),
        ("swiglu", True),
        ("geglu", True),
        ("layer_norm", True),
        ("cross_entropy", False),
        ("fused_linear_cross_entropy", False),
    ):
        if name in accepted:
            kwargs[name] = value
    patcher(**kwargs)
    enabled = sorted(k for k, v in kwargs.items() if v is True)
    print(
        f"Liger inference enabled for {model_type}: {', '.join(enabled)}",
        flush=True,
    )
    return True


def _replace_layernorms(model):
    """Replace compatible ``torch.nn.LayerNorm`` modules in-place."""
    import torch
    from liger_kernel.transformers import LigerLayerNorm

    replaced = 0

    def visit(parent):
        nonlocal replaced
        for name, child in list(parent.named_children()):
            if isinstance(child, torch.nn.LayerNorm):
                shape = child.normalized_shape
                if len(shape) != 1:
                    raise RuntimeError(f"Liger cannot replace LayerNorm shape {shape}")
                layer = LigerLayerNorm(
                    shape[0], eps=child.eps, bias=child.bias is not None
                ).to(device=child.weight.device, dtype=child.weight.dtype)
                with torch.no_grad():
                    layer.weight.copy_(child.weight)
                    if child.bias is not None:
                        layer.bias.copy_(child.bias)
                setattr(parent, name, layer)
                replaced += 1
            else:
                visit(child)

    visit(model)
    return replaced


def apply_liger_to_model(model):
    """Accelerate ``model`` in-place where Liger supports it."""
    if os.environ.get("USE_LIGER", "1") in {"0", "false", "False"}:
        return model

    # Every Liger kernel is Triton, so it can only run on CUDA tensors. On a
    # cpu model the swap succeeds and then dies at the first forward with
    # "Pointer argument cannot be accessed from Triton", far from the cause.
    try:
        on_cuda = next(model.parameters()).is_cuda
    except StopIteration:
        on_cuda = False
    if not on_cuda:
        print("Liger inference skipped: model is not on cuda", flush=True)
        return model

    model_type = getattr(getattr(model, "config", None), "model_type", None)
    try:
        if _apply_official_patcher(model, model_type):
            return model
        replaced = _replace_layernorms(model)
    except Exception as exc:  # a kernel swap must never take the run down
        print(f"Liger inference skipped ({type(exc).__name__}: {exc})", flush=True)
        return model

    if replaced == 0:
        print(
            f"Liger inference skipped: no patcher for model_type={model_type!r} "
            f"and no torch.nn.LayerNorm in {type(model).__name__}; "
            f"running unaccelerated",
            flush=True,
        )
        return model
    print(f"Liger inference enabled: replaced {replaced} LayerNorm modules", flush=True)
    return model
