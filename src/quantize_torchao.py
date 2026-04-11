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
from models.module_retr.transformer import ConditionalTransformerEncoderLayer

# ---------------------------------------------------------------------------
# Quantization schemes
# ---------------------------------------------------------------------------

SCHEMES = {
    "bf16": "bf16",  # sentinel — handled as dtype conversion, not torchao
    "int8wo": Int8WeightOnlyConfig(),
    "int8dq": Int8DynamicActivationInt8WeightConfig(),
    # INT4 weight-only: symmetric per-group fake-quant (weights rounded to INT4,
    # dequantized back to FP32 before matmul).  Correctly models accuracy impact
    # of INT4 weight quantization without requiring mslk/tinygemm kernels.
    "int4fq_g128": 128,
    "int4fq_g64": 64,
    "int4fq_g32": 32,
}

DTYPE_SCHEMES = {"bf16"}  # schemes that use dtype conversion instead of torchao
INT4_FQ_SCHEMES = {"int4fq_g128", "int4fq_g64", "int4fq_g32"}  # fake-quant INT4

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
# INT4 fake-quant (weight-only, per-group symmetric)
# ---------------------------------------------------------------------------

def _apply_int4_weight_fake_quant(
    model: nn.Module,
    group_size: int,
    filter_fn,
) -> None:
    """
    Apply symmetric per-group INT4 fake-quantization to matched nn.Linear weights.

    For each weight matrix W [out, in]:
      1. Partition input dim into groups of `group_size`.
      2. Compute per-group scale = max(|W_group|) / 7  (INT4 symmetric range [-8, 7]).
      3. Quantize: W_q = clamp(round(W / scale), -8, 7).
      4. Dequantize: W_dq = W_q * scale.
      5. Replace mod.weight.data with W_dq (FP32, same shape).

    Runtime matmul remains FP32 (weight-only quantization).  The rounding noise
    introduced here is identical to what hardware INT4 weight-only would produce,
    so accuracy results are directly comparable to hardware deployment.

    Stores metadata in model._int4_fq_modules = {module_name: (out_f, in_f, g)}
    for `model_size_mb` to compute theoretical INT4 storage size.
    """
    fq_meta = {}
    for name, mod in model.named_modules():
        if not (isinstance(mod, nn.Linear) and filter_fn(mod, name)):
            continue
        w = mod.weight.data  # [out_features, in_features]
        out_f, in_f = w.shape
        if in_f % group_size != 0:
            # Skip layers not divisible by group_size (rare; small bias layers).
            continue
        n_groups = in_f // group_size
        w_g = w.reshape(out_f, n_groups, group_size)  # [out, n_groups, g]
        # Per-group symmetric scale: maps max absolute weight to INT4 max (7).
        scale = w_g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 7.0
        w_q = (w_g / scale).round().clamp(-8, 7)
        mod.weight.data = (w_q * scale).reshape(out_f, in_f)
        fq_meta[name] = (out_f, in_f, group_size)
    model._int4_fq_modules = fq_meta


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

    filter_fn = COMPONENT_FILTERS[component]

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

    # INT4 fake-quant: symmetric per-group weight rounding, FP32 matmul.
    if scheme in INT4_FQ_SCHEMES:
        group_size = SCHEMES[scheme]  # int: 128, 64, or 32
        _apply_int4_weight_fake_quant(model, group_size, filter_fn)
        return model

    quant_config = SCHEMES[scheme]
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


def _make_fake_quant_uint8():
    """Asymmetric per-tensor UINT8 for non-negative tensors (e.g. softmax output).

    softmax output ∈ [0, 1], always non-negative.  Symmetric INT8 wastes half
    the range; this uses the full [0, 255] unsigned range instead.
    scale = max(x) / 255.0, clamp to [0, 255], dequant back to float32.
    """
    def fake_quant(x: torch.Tensor) -> torch.Tensor:
        scale = x.max() / 255.0
        if scale == 0:
            return x
        x_q = torch.clamp(torch.round(x / scale), 0, 255)
        return x_q * scale
    return fake_quant


def register_attention_weights_quant_hooks(model: nn.Module) -> list:
    """
    Fake-quantize the attention weight matrix (softmax output) before AV bmm.
    Uses UINT8 asymmetric (softmax output ∈ [0,1], always non-negative).

    Sets attn_weights_fake_quant attribute on each affected module; the local
    multi_head_attention_forward in attention.py applies it after softmax/dropout
    and before torch.bmm(attn_output_weights, v).

    Returns list of _ModAttrHandle objects (same pattern as
    register_attention_bmm_quant_hooks); call .remove() on each to reset to None.
    """
    fake_quant = _make_fake_quant_uint8()
    handles = []

    for _, mod in model.named_modules():
        if isinstance(mod, (CustomMHA, ConditionalTransformerEncoderLayer)):
            mod.attn_weights_fake_quant = fake_quant
            handles.append(_ModAttrHandle(mod, "attn_weights_fake_quant", None))

    return handles


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


class _ModAttrHandle:
    """Cleanup handle for attribute-based hooks (no PyTorch hook handle)."""
    def __init__(self, mod, attr, default):
        self._mod = mod
        self._attr = attr
        self._default = default

    def remove(self):
        setattr(self._mod, self._attr, self._default)


def register_attention_bmm_quant_hooks(
    model: nn.Module, bits: int = 8
) -> list:
    """
    Fake-quantize Q, K, V at the bmm input (post-projection, post-head-split).
    Covers both encoder (ConditionalTransformerEncoderLayer) and decoder (CustomMHA).

    scale = max(|x|) / 127  (per-tensor symmetric INT8, STE)

    Sets qkv_fake_quant attribute on each affected module; the local
    multi_head_attention_forward in attention.py applies it just before torch.bmm.
    The encoder's forward_post was refactored to use the local MHA so this
    attribute is honoured there too.

    Parameters
    ----------
    model : RETR model (eval mode, weights loaded).
    bits  : quantization precision (8).

    Returns
    -------
    List of handle objects with .remove() to reset qkv_fake_quant to None.
    """
    fake_quant = _make_fake_quant_int8(bits)
    handles = []

    for _, mod in model.named_modules():
        if isinstance(mod, (CustomMHA, ConditionalTransformerEncoderLayer)):
            mod.qkv_fake_quant = fake_quant
            handles.append(_ModAttrHandle(mod, "qkv_fake_quant", None))

    return handles


# ---------------------------------------------------------------------------
# Utility: model size in MB
# ---------------------------------------------------------------------------

def model_size_mb(model: nn.Module) -> float:
    """Total parameter + buffer memory in MB (approximate).

    Handles three cases:
    - torchao AffineQuantizedTensor: sums actual int_data + scale storage.
    - INT4 fake-quant (model._int4_fq_modules set): computes theoretical INT4
      packed storage (4 bits/weight) + FP32 scales, matching hardware layout.
    - All other parameters: actual element count × element size.
    """
    try:
        from torchao.dtypes import AffineQuantizedTensor
    except ImportError:
        AffineQuantizedTensor = None

    try:
        from torchao.quantization.linear_activation_quantized_tensor import (
            LinearActivationQuantizedTensor,
        )
    except ImportError:
        LinearActivationQuantizedTensor = None

    int4_fq_mods = getattr(model, "_int4_fq_modules", {})

    def _aqt_storage(p):
        """Bytes for an AffineQuantizedTensor: int_data + scale."""
        impl = p.tensor_impl
        return (
            impl.int_data.nelement() * impl.int_data.element_size()
            + impl.scale.nelement() * impl.scale.element_size()
        )

    total = 0
    for param_name, p in model.named_parameters():
        parts = param_name.rsplit(".", 1)
        mod_name = parts[0] if len(parts) == 2 else ""
        param_base = parts[1] if len(parts) == 2 else parts[0]

        if mod_name in int4_fq_mods and param_base == "weight":
            out_f, in_f, g = int4_fq_mods[mod_name]
            # INT4 packed: 2 weights per byte
            total += (out_f * in_f + 1) // 2
            # FP32 scale per group per output channel
            total += out_f * (in_f // g) * 4
        elif AffineQuantizedTensor is not None and isinstance(p, AffineQuantizedTensor):
            total += _aqt_storage(p)
        elif (
            LinearActivationQuantizedTensor is not None
            and isinstance(p, LinearActivationQuantizedTensor)
            and AffineQuantizedTensor is not None
            and isinstance(p.original_weight_tensor, AffineQuantizedTensor)
        ):
            # int8dq: activation quantization wrapper around an AQT weight
            total += _aqt_storage(p.original_weight_tensor)
        else:
            total += p.nelement() * p.element_size()

    for b in model.buffers():
        total += b.nelement() * b.element_size()
    return total / (1024 ** 2)
