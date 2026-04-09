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

* Encoder self_attn: the original fused nn.MultiheadAttention (in_proj_weight
  [1536, 512]) has been replaced in transformer.py with explicit nn.Linear
  modules (self_attn_q_proj, self_attn_k_proj, self_attn_v_proj,
  self_attn_out_proj) using F.multi_head_attention_forward with
  use_separate_proj_weight=True.  torchao covers all four naturally via the
  standard filter_fn path.  Checkpoints are migrated by
  migrate_encoder_mha_state_dict() at load time.

* BF16 scheme: applies torch.bfloat16 dtype conversion to the full model.
  Component-level BF16 is not supported because activation dtype mismatches at
  module boundaries cause runtime errors; full-model conversion is the standard
  approach (cf. model.to(torch.bfloat16) in HuggingFace inference).
  Inputs must also be cast to bfloat16 before forward — handled in eval_ptq.py.

* Layers NOT quantized: detection/segmentation heads, nn.LayerNorm, nn.GroupNorm,
  nn.Embedding (query_embed).  These are small or directly output-sensitive.
"""

import torch
import torch.nn as nn

try:
    from torchao.quantization import (
        Int4WeightOnlyConfig,
        Int8DynamicActivationInt8WeightConfig,
        Int8WeightOnlyConfig,
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
    "bf16": "bf16",  # sentinel — handled as dtype conversion, not torchao
    "int8wo": Int8WeightOnlyConfig(),
    "int4wo_g128": Int4WeightOnlyConfig(group_size=128),
    "int4wo_g64": Int4WeightOnlyConfig(group_size=64),
    "int4wo_g32": Int4WeightOnlyConfig(group_size=32),
    "int8dq": Int8DynamicActivationInt8WeightConfig(),
}

DTYPE_SCHEMES = {"bf16"}  # schemes that use dtype conversion instead of torchao

# ---------------------------------------------------------------------------
# Component filter functions
# ---------------------------------------------------------------------------
# Each function returns True for a named module that should be quantized.
# torchao calls filter_fn(module, full_qualified_name) for every leaf module.

def _is_backbone(mod: nn.Module, fqn: str) -> bool:
    """ResNet18 + input_proj linear layers only.

    NOTE: torchao 0.17.0 has a shape-mismatch bug when applying per-channel
    INT8 to 1×1 Conv2d (scale.view([out,in,1,1]) fails for size-1 scale).
    All backbone Conv2d layers are excluded until this is resolved upstream.
    The backbone has no nn.Linear layers (all Conv2d), so this filter currently
    matches nothing — backbone quantization is a no-op for INT8/INT4 schemes.
    """
    return (
        isinstance(mod, nn.Linear)
        and ("backbone" in fqn or "input_proj" in fqn)
        and "class_embed" not in fqn
        and "bbox_embed" not in fqn
        and "seg" not in fqn.lower()
    )


def _is_encoder(mod: nn.Module, fqn: str) -> bool:
    """All Linear layers inside the transformer encoder layers.
    (Module path: model.detr.encoder.layers.N.*)
    """
    return isinstance(mod, nn.Linear) and "encoder" in fqn


def _is_decoder(mod: nn.Module, fqn: str) -> bool:
    """All Linear layers inside the transformer decoder layers.
    (Module path: model.detr.decoder.layers.N.* and decoder.query_scale/ref_point_head)
    """
    return isinstance(mod, nn.Linear) and "decoder" in fqn


def _is_transformer(mod: nn.Module, fqn: str) -> bool:
    """Encoder + decoder combined."""
    return _is_encoder(mod, fqn) or _is_decoder(mod, fqn)


def _is_ffn(mod: nn.Module, fqn: str) -> bool:
    """Feed-forward (linear1 / linear2) inside encoder and decoder."""
    return (
        isinstance(mod, nn.Linear)
        and ("encoder" in fqn or "decoder" in fqn)
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
# Checkpoint key migration
# ---------------------------------------------------------------------------

def migrate_encoder_mha_state_dict(state_dict: dict) -> dict:
    """
    Remap checkpoint keys from the original fused nn.MultiheadAttention encoder
    self_attn to the refactored explicit Q/K/V Linear modules in
    ConditionalTransformerEncoderLayer (transformer.py).

    Original keys (per encoder layer N):
        model.detr.encoder.layers.N.self_attn.in_proj_weight  [1536, 512]
        model.detr.encoder.layers.N.self_attn.in_proj_bias    [1536]
        model.detr.encoder.layers.N.self_attn.out_proj.weight [512, 512]
        model.detr.encoder.layers.N.self_attn.out_proj.bias   [512]

    New keys (refactored):
        model.detr.encoder.layers.N.self_attn_q_proj.weight   [512, 512]
        model.detr.encoder.layers.N.self_attn_q_proj.bias     [512]
        model.detr.encoder.layers.N.self_attn_k_proj.weight   [512, 512]
        model.detr.encoder.layers.N.self_attn_k_proj.bias     [512]
        model.detr.encoder.layers.N.self_attn_v_proj.weight   [512, 512]
        model.detr.encoder.layers.N.self_attn_v_proj.bias     [512]
        model.detr.encoder.layers.N.self_attn_out_proj.weight [512, 512]
        model.detr.encoder.layers.N.self_attn_out_proj.bias   [512]
    """
    new_sd = {}
    for key, val in state_dict.items():
        if "encoder" not in key or "self_attn" not in key:
            new_sd[key] = val
            continue

        prefix = key.rsplit("self_attn", 1)[0]  # e.g. 'model.detr.encoder.layers.0.'

        if key.endswith(".self_attn.in_proj_weight"):
            d = val.shape[0] // 3
            new_sd[prefix + "self_attn_q_proj.weight"] = val[0:d].clone()
            new_sd[prefix + "self_attn_k_proj.weight"] = val[d:2*d].clone()
            new_sd[prefix + "self_attn_v_proj.weight"] = val[2*d:3*d].clone()
        elif key.endswith(".self_attn.in_proj_bias"):
            d = val.shape[0] // 3
            new_sd[prefix + "self_attn_q_proj.bias"] = val[0:d].clone()
            new_sd[prefix + "self_attn_k_proj.bias"] = val[d:2*d].clone()
            new_sd[prefix + "self_attn_v_proj.bias"] = val[2*d:3*d].clone()
        elif key.endswith(".self_attn.out_proj.weight"):
            new_sd[prefix + "self_attn_out_proj.weight"] = val
        elif key.endswith(".self_attn.out_proj.bias"):
            new_sd[prefix + "self_attn_out_proj.bias"] = val
        else:
            new_sd[key] = val

    return new_sd


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

    # BF16: use torch.autocast rather than model.to(bfloat16).
    # transformer.py has hardcoded dtype=torch.float32 in positional encodings;
    # model.to(bfloat16) causes dtype mismatches at those boundaries.
    # autocast handles mixed-dtype ops transparently — the standard approach
    # (used by HuggingFace, PyTorch docs) for BF16 inference.
    # Model weights stay float32; autocast casts eligible ops to bfloat16 on the fly.
    # The eval_ptq.py forward loop wraps in torch.autocast when scheme == "bf16".
    if scheme in DTYPE_SCHEMES:
        if component != "all":
            raise ValueError(
                f"scheme='{scheme}' requires component='all' — autocast applies "
                "globally; per-component BF16 is not supported."
            )
        return model  # no model mutation; autocast is applied in eval_ptq.py

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
    """Total parameter + buffer memory in MB (approximate).

    Handles torchao AffineQuantizedTensor weights by summing their actual
    int_data + scale storage rather than the logical float32 size.
    """
    try:
        from torchao.dtypes import AffineQuantizedTensor
    except ImportError:
        AffineQuantizedTensor = None

    total = 0
    for p in model.parameters():
        if AffineQuantizedTensor is not None and isinstance(p, AffineQuantizedTensor):
            impl = p.tensor_impl
            total += impl.int_data.nelement() * impl.int_data.element_size()
            total += impl.scale.nelement() * impl.scale.element_size()
        else:
            total += p.nelement() * p.element_size()
    for b in model.buffers():
        total += b.nelement() * b.element_size()
    return total / (1024 ** 2)
