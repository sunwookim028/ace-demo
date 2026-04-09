# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Post-training quantization (PTQ) for RETR using torchao.

All quantization is weight-only or dynamic-activation at inference time.
No training, no QAT, no modification of trusted source files.

Usage
-----
    from quantize_torchao import apply_ptq, register_attention_act_quant_hooks

    model = RETR(...)
    model.load_state_dict(torch.load(ckpt))
    model.eval()

    apply_ptq(model, scheme="int8wo", component="all")
    # Optionally also quantize attention activations:
    hooks = register_attention_act_quant_hooks(model, bits=8)

Design notes
------------
* torchao `quantize_()` covers all nn.Linear and nn.Conv2d layers identified by
  `filter_fn`.  Weights are stored as INT8/INT4 and dequantized to float32 on
  compute, so downstream detection/segmentation heads always receive float32.

* The custom MultiheadAttention in attention.py uses torch.bmm (not
  F.scaled_dot_product_attention), so torchao cannot quantize the matmul itself.
  We address this by registering forward pre-hooks on each MultiheadAttention
  instance that fake-quantize the Q/K/V float tensors before they enter forward().
  This is a *simulation* of INT8 activation quantization (straight-through
  estimator); it measures accuracy impact without hardware INT8 matmul speed.

* Encoder self_attn uses standard nn.MultiheadAttention with fused in_proj_weight.
  torchao quantizes its in_proj_weight as a single [1536, 512] linear.
  Activation hooks are NOT applied to the encoder (standard MHA, different path).

* Layers NOT quantized: detection/segmentation heads, nn.LayerNorm, nn.GroupNorm,
  nn.Embedding (query_embed).  These are small or directly output-sensitive.
"""

import torch
import torch.nn as nn

try:
    from torchao.quantization import (
        int4_weight_only,
        int8_dynamic_activation_int8_weight,
        int8_weight_only,
        quantize_,
    )
except ImportError as e:
    raise ImportError(
        "torchao >= 0.6 is required. "
        "Activate the retr-quant conda environment: conda activate retr-quant"
    ) from e

from models.module_retr.attention import MultiheadAttention as CustomMHA

# ---------------------------------------------------------------------------
# Quantization schemes
# ---------------------------------------------------------------------------

SCHEMES = {
    "int8wo": int8_weight_only(),
    "int4wo_g128": int4_weight_only(group_size=128),
    "int4wo_g64": int4_weight_only(group_size=64),
    "int4wo_g32": int4_weight_only(group_size=32),
    "int8dq": int8_dynamic_activation_int8_weight(),
}

# ---------------------------------------------------------------------------
# Component filter functions
# ---------------------------------------------------------------------------
# Each function returns True for a named module that should be quantized.
# torchao calls filter_fn(module, full_qualified_name) for every leaf module.

def _is_backbone(mod: nn.Module, fqn: str) -> bool:
    """ResNet18 Conv2d layers + input_proj in RETR (retr.py level)."""
    # retr.backbone.*  and  retr.input_proj / retr.input_proj_ver
    return (
        isinstance(mod, (nn.Conv2d, nn.Linear))
        and ("backbone" in fqn or "input_proj" in fqn)
        # exclude detection/seg heads that also contain 'proj'
        and "class_embed" not in fqn
        and "bbox_embed" not in fqn
        and "seg" not in fqn.lower()
    )


def _is_encoder(mod: nn.Module, fqn: str) -> bool:
    """All Linear/Conv2d inside transformer encoder layers."""
    return (
        isinstance(mod, (nn.Linear, nn.Conv2d))
        and "transformer" in fqn
        and "encoder" in fqn
    )


def _is_decoder(mod: nn.Module, fqn: str) -> bool:
    """All Linear/Conv2d inside transformer decoder layers."""
    return (
        isinstance(mod, (nn.Linear, nn.Conv2d))
        and "transformer" in fqn
        and "decoder" in fqn
    )


def _is_transformer(mod: nn.Module, fqn: str) -> bool:
    """Encoder + decoder combined."""
    return _is_encoder(mod, fqn) or _is_decoder(mod, fqn)


def _is_ffn(mod: nn.Module, fqn: str) -> bool:
    """Feed-forward (linear1 / linear2) inside encoder and decoder."""
    return (
        isinstance(mod, nn.Linear)
        and "transformer" in fqn
        and ("linear1" in fqn or "linear2" in fqn)
    )


def _is_proj(mod: nn.Module, fqn: str) -> bool:
    """Projection linears in encoder and decoder (non-FFN)."""
    return _is_transformer(mod, fqn) and not _is_ffn(mod, fqn)


def _is_all(mod: nn.Module, fqn: str) -> bool:
    """Everything quantizable: backbone + encoder + decoder."""
    return _is_backbone(mod, fqn) or _is_transformer(mod, fqn)


COMPONENT_FILTERS = {
    "backbone": _is_backbone,
    "encoder": _is_encoder,
    "decoder": _is_decoder,
    "transformer": _is_transformer,
    "ffn": _is_ffn,
    "proj": _is_proj,
    "all": _is_all,
}

# ---------------------------------------------------------------------------
# Main PTQ entry point
# ---------------------------------------------------------------------------

def apply_ptq(model: nn.Module, scheme: str, component: str) -> nn.Module:
    """
    Apply torchao post-training weight quantization to *model* in-place.

    Parameters
    ----------
    model     : RETR model, already loaded with pretrained weights and in eval().
    scheme    : one of SCHEMES keys — "int8wo", "int4wo_g128", "int4wo_g64",
                "int4wo_g32", "int8dq"
    component : one of COMPONENT_FILTERS keys — "backbone", "encoder", "decoder",
                "transformer", "ffn", "proj", "all"

    Returns
    -------
    The same model object (modified in-place).
    """
    if scheme not in SCHEMES:
        raise ValueError(f"Unknown scheme '{scheme}'. Choose from: {list(SCHEMES)}")
    if component not in COMPONENT_FILTERS:
        raise ValueError(f"Unknown component '{component}'. Choose from: {list(COMPONENT_FILTERS)}")

    quant_config = SCHEMES[scheme]
    filter_fn = COMPONENT_FILTERS[component]

    quantize_(model, quant_config, filter_fn=filter_fn)
    return model


# ---------------------------------------------------------------------------
# Attention activation quantization via forward pre-hooks
# ---------------------------------------------------------------------------

def _make_fake_quant_int8(bits: int = 8):
    """
    Returns a function that fake-quantizes a float32 tensor to INT8 range
    using per-tensor symmetric quantization (straight-through estimator).

    The scale is derived from the absolute max of the tensor, then the tensor
    is rounded to the nearest integer in [-2^(bits-1), 2^(bits-1)-1] and
    dequantized back to float32.  This simulates INT8 without an actual INT8
    matmul — it measures accuracy impact only.
    """
    quant_max = 2 ** (bits - 1) - 1  # 127 for INT8

    def fake_quant(x: torch.Tensor) -> torch.Tensor:
        scale = x.abs().max() / quant_max
        if scale == 0:
            return x
        x_q = torch.clamp(torch.round(x / scale), -quant_max - 1, quant_max)
        return x_q * scale

    return fake_quant


def register_attention_act_quant_hooks(
    model: nn.Module, bits: int = 8
) -> list:
    """
    Register forward pre-hooks on all CustomMHA (decoder attention) instances
    to fake-quantize the Q/K/V input tensors before torch.bmm.

    The custom MultiheadAttention (attention.py) receives already-projected
    Q/K/V tensors as positional args (query, key, value).  The hook intercepts
    these and applies symmetric per-tensor INT{bits} fake-quantization.

    Parameters
    ----------
    model : RETR model (eval mode, weights loaded).
    bits  : quantization precision for activations (8 by default).

    Returns
    -------
    List of hook handles — call handle.remove() on each to undo.
    """
    fake_quant = _make_fake_quant_int8(bits)
    handles = []

    for name, mod in model.named_modules():
        if isinstance(mod, CustomMHA):
            def make_hook(module_name):
                def hook(module, args):
                    # args = (query, key, value, ...)
                    # Only quantize the first three positional arguments.
                    new_args = list(args)
                    for i in range(min(3, len(new_args))):
                        if isinstance(new_args[i], torch.Tensor):
                            new_args[i] = fake_quant(new_args[i])
                    return tuple(new_args)
                return hook

            handle = mod.register_forward_pre_hook(make_hook(name))
            handles.append(handle)

    return handles


# ---------------------------------------------------------------------------
# Utility: model size in MB
# ---------------------------------------------------------------------------

def model_size_mb(model: nn.Module) -> float:
    """Total parameter + buffer memory in MB (approximate)."""
    total = 0
    for p in model.parameters():
        total += p.nelement() * p.element_size()
    for b in model.buffers():
        total += b.nelement() * b.element_size()
    return total / (1024 ** 2)
