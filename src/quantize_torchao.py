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
import torch.nn.functional as F

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
    "fp16": "fp16",  # sentinel — float16 autocast (primarily for GPU runs)
    "int8wo": Int8WeightOnlyConfig(),
    "int8dq": Int8DynamicActivationInt8WeightConfig(),
    # INT4 weight-only: symmetric per-group fake-quant (weights rounded to INT4,
    # dequantized back to FP32 before matmul).  Correctly models accuracy impact
    # of INT4 weight quantization without requiring mslk/tinygemm kernels.
    "int4fq_g128": 128,
    "int4fq_g64": 64,
    "int4fq_g32": 32,
}

DTYPE_SCHEMES = {"bf16", "fp16"}  # schemes that use dtype conversion instead of torchao
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


# NVFP4 E2M1 representable positive values and their rounding boundaries.
# E2M1: exponent bias=1, normal values ±{0.5,1,1.5,2,3,4,6}, plus zero.
_FP4_POS_VALS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_FP4_BOUNDARIES = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def _make_fake_quant_nvfp4(block_size: int = 16, mx_scale: bool = False):
    """NVFP4 E2M1 block-scaled fake-quant (straight-through estimator).

    Matches the NVFP4 hardware quantizer: 128 activations/cycle, block size 16.

    Per block of `block_size` consecutive elements along the last dimension:
      max_abs = max(|x_block|)
      scale   = max_abs / 6.0          # FP8 scale (full precision)
      [mx_scale=True: round up to nearest power-of-2 — ~3x cheaper, MX format]
      x_q     = round_fp4(x / scale) * scale

    FP4 E2M1 representable values: ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}.
    """
    pos_vals = _FP4_POS_VALS
    boundaries = _FP4_BOUNDARIES

    def fake_quant(x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        last = orig_shape[-1]

        # Pad last dim to a multiple of block_size (keep padding out of scale calc)
        if last % block_size != 0:
            pad = block_size - last % block_size
            x_work = F.pad(x, (0, pad))
        else:
            x_work = x

        padded_last = x_work.shape[-1]
        n_blocks = padded_last // block_size

        # reshape to [..., n_blocks, block_size] — each block gets its own scale
        blocked = x_work.reshape(*x_work.shape[:-1], n_blocks, block_size)

        max_abs = blocked.abs().amax(dim=-1, keepdim=True)
        if mx_scale:
            # Round scale exponent up so max value never overflows FP4 range
            scale = torch.pow(2.0, torch.ceil(torch.log2((max_abs / 6.0).clamp(min=1e-12))))
        else:
            scale = (max_abs / 6.0).clamp(min=1e-12)

        x_scaled = blocked / scale  # values nominally in [-6, 6]

        sign = x_scaled.sign()
        abs_x = x_scaled.abs().clamp(max=6.0)
        pv = pos_vals.to(x.device)
        bv = boundaries.to(x.device)
        idx = torch.bucketize(abs_x.contiguous(), bv)
        x_dq = (sign * pv[idx] * scale).reshape(*orig_shape[:-1], padded_last)

        if last % block_size != 0:
            x_dq = x_dq[..., :last]
        return x_dq

    return fake_quant


def register_attention_bmm_fp4_hooks(
    model: nn.Module, block_size: int = 16, mx_scale: bool = False
) -> list:
    """Fake-quantize Q/K/V at bmm input using NVFP4 E2M1 block quantization.

    Covers both encoder (ConditionalTransformerEncoderLayer) and decoder (CustomMHA).
    Mutually exclusive with register_attention_bmm_quant_hooks — both write qkv_fake_quant.
    """
    fake_quant = _make_fake_quant_nvfp4(block_size=block_size, mx_scale=mx_scale)
    handles = []
    for _, mod in model.named_modules():
        if isinstance(mod, (CustomMHA, ConditionalTransformerEncoderLayer)):
            mod.qkv_fake_quant = fake_quant
            handles.append(_ModAttrHandle(mod, "qkv_fake_quant", None))
    return handles


def register_attention_weights_fp4_hooks(
    model: nn.Module, block_size: int = 16, mx_scale: bool = False
) -> list:
    """Fake-quantize attention weights (post-softmax) using NVFP4 E2M1 block quantization.

    Mutually exclusive with register_attention_weights_quant_hooks — both write
    attn_weights_fake_quant.
    """
    fake_quant = _make_fake_quant_nvfp4(block_size=block_size, mx_scale=mx_scale)
    handles = []
    for _, mod in model.named_modules():
        if isinstance(mod, (CustomMHA, ConditionalTransformerEncoderLayer)):
            mod.attn_weights_fake_quant = fake_quant
            handles.append(_ModAttrHandle(mod, "attn_weights_fake_quant", None))
    return handles


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


def apply_fp16_half_backbone(model: nn.Module) -> list:
    """Convert the RETR backbone + input_proj{,_ver} Conv2d/BN weights to FP16.

    Adds a forward pre-hook on backbone to cast input tensors to FP16, and a
    forward post-hook on input_proj/input_proj_ver to cast output back to FP32
    (so downstream transformer sees FP32).

    Skips any module whose weight is an AffineQuantizedTensor (silent-corruption
    guard: `.half()` on AQT is a no-op that leaves int8 weights but corrupts
    downstream dtype chain).

    Returns a list of handle objects with `.remove()` — currently empty because
    dtype mutation is not easily reversible.
    """
    try:
        from torchao.dtypes import AffineQuantizedTensor
    except ImportError:
        AffineQuantizedTensor = None

    # RETR wraps: retr.model is the ConditionalDETR, which may be further wrapped
    # by DETRsegm (retr.model.detr.*). Walk to the module that actually owns
    # .backbone, .input_proj, .input_proj_ver.
    inner = model
    for attr in ("model", "detr"):
        nxt = getattr(inner, attr, None)
        if nxt is not None and hasattr(nxt, "backbone"):
            inner = nxt
            break
        if nxt is not None:
            inner = nxt
    backbone = getattr(inner, "backbone", None)
    if backbone is None:
        raise ValueError("apply_fp16_half_backbone: could not locate .backbone attr")

    def _has_aqt(m):
        if AffineQuantizedTensor is None:
            return False
        for p in m.parameters(recurse=True):
            if isinstance(p.data, AffineQuantizedTensor):
                return True
        return False

    if _has_aqt(backbone):
        raise RuntimeError("Backbone contains AQT (int8) weight; refusing .half()")
    backbone.half()

    def _cast_input_to_fp16(mod, args):
        new = []
        for a in args:
            if isinstance(a, torch.Tensor):
                new.append(a.half())
            elif hasattr(a, "tensors") and isinstance(a.tensors, torch.Tensor):
                # NestedTensor: mutate in place — dataloader doesn't re-use.
                a.tensors = a.tensors.half()
                new.append(a)
            else:
                new.append(a)
        return tuple(new)

    handles = [backbone.register_forward_pre_hook(_cast_input_to_fp16)]

    # Backbone returns (features, pos). features is a list of NestedTensors (FPN
    # levels); the level-0 feature feeds input_proj, deeper levels feed mask_head
    # and segmentation adapters directly. pos is a list of Tensors. Downstream
    # ops outside backbone are fp32 (input_proj Conv2d, mask_head adapters,
    # ca_kpos_proj Linear), so cast every output tensor back to fp32 at the
    # backbone boundary to keep the fp16 scope strictly inside the backbone.
    def _cast_fp16_to_fp32(x):
        if isinstance(x, torch.Tensor) and x.dtype == torch.float16:
            return x.float()
        if hasattr(x, "tensors") and isinstance(x.tensors, torch.Tensor):
            if x.tensors.dtype == torch.float16:
                x.tensors = x.tensors.float()
            return x
        if isinstance(x, (list, tuple)):
            return type(x)(_cast_fp16_to_fp32(xi) for xi in x)
        return x

    def _cast_backbone_out_to_fp32(mod, inp, out):
        return _cast_fp16_to_fp32(out)
    handles.append(backbone.register_forward_hook(_cast_backbone_out_to_fp32))

    return handles


def register_fp32_ln_hooks(model: nn.Module, scope: str = "all") -> list:
    """Force nn.LayerNorm to run in FP32 regardless of surrounding autocast.

    Pre-hook casts LN input to FP32; post-hook leaves output FP32 (downstream
    autocast will recast on its own). Matches the PARTITIONING.md decoder LN
    recipe where decoder LN must stay FP32 on the FPGA chiplet.

    scope ∈ {"encoder", "decoder", "transformer", "all"}: which LN modules to target.
    """
    handles = []

    def _in_scope(fqn: str) -> bool:
        if scope == "all":
            return True
        if scope == "transformer":
            return "encoder" in fqn or "decoder" in fqn
        return scope in fqn

    def _pre(mod, args):
        if not args:
            return args
        x = args[0]
        if isinstance(x, torch.Tensor) and x.dtype != torch.float32:
            return (x.float(),) + tuple(args[1:])
        return args

    # Preserve LN param dtype as FP32 to match chiplet recipe; saves us from
    # dtype mismatches if upstream halves LN weights.
    for name, mod in model.named_modules():
        if isinstance(mod, nn.LayerNorm) and _in_scope(name):
            if mod.weight is not None and mod.weight.dtype != torch.float32:
                mod.weight.data = mod.weight.data.float()
            if mod.bias is not None and mod.bias.dtype != torch.float32:
                mod.bias.data = mod.bias.data.float()
            handles.append(mod.register_forward_pre_hook(_pre))

    return handles


def register_fp32_softmax_hooks(model: nn.Module, scope: str = "all") -> list:
    """Set softmax_dtype=torch.float32 attribute on in-scope attention modules.

    Read by multi_head_attention_forward in attention.py. Forces softmax compute
    to FP32 (up-cast input, compute, down-cast output) even when surrounding
    autocast is FP16/BF16.

    scope ∈ {"encoder", "decoder", "transformer", "all"}.
    """
    handles = []

    def _in_scope(fqn: str) -> bool:
        if scope == "all":
            return True
        if scope == "transformer":
            return "encoder" in fqn or "decoder" in fqn
        return scope in fqn

    for name, mod in model.named_modules():
        if isinstance(mod, (CustomMHA, ConditionalTransformerEncoderLayer)) and _in_scope(name):
            mod.softmax_dtype = torch.float32
            handles.append(_ModAttrHandle(mod, "softmax_dtype", None))

    return handles


def register_fp16_linear_out_hooks(model: nn.Module, scope: str = "transformer") -> list:
    """Cast nn.Linear output FP32→FP16 immediately after the forward pass.

    torchao int8dq hardcodes INT32 accumulate → FP32 output. This post-hook
    re-casts to FP16 so that residual adds after each W8A8 linear run in FP16
    rather than upcasting to FP32.

    scope ∈ {"encoder", "decoder", "transformer", "all"}.
    """
    handles = []

    def _in_scope(fqn: str) -> bool:
        if scope == "all":
            return True
        if scope == "transformer":
            return "encoder" in fqn or "decoder" in fqn
        return scope in fqn

    def _post(mod, args, output):
        if isinstance(output, torch.Tensor) and output.dtype != torch.float16:
            return output.half()
        return output

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and _in_scope(name):
            handles.append(mod.register_forward_hook(_post))

    return handles


def register_fp16_ln_hooks(model: nn.Module, scope: str = "all") -> list:
    """Force nn.LayerNorm to run in FP16.

    Pre-hook casts input to FP16; LN params are cast to FP16. Output stays FP16
    so downstream residual adds inherit the dtype. Opposite of register_fp32_ln_hooks.

    scope ∈ {"encoder", "decoder", "transformer", "all"}.
    """
    handles = []

    def _in_scope(fqn: str) -> bool:
        if scope == "all":
            return True
        if scope == "transformer":
            return "encoder" in fqn or "decoder" in fqn
        return scope in fqn

    def _pre(mod, args):
        if not args:
            return args
        x = args[0]
        if isinstance(x, torch.Tensor) and x.dtype != torch.float16:
            return (x.half(),) + tuple(args[1:])
        return args

    for name, mod in model.named_modules():
        if isinstance(mod, nn.LayerNorm) and _in_scope(name):
            if mod.weight is not None:
                mod.weight.data = mod.weight.data.half()
            if mod.bias is not None:
                mod.bias.data = mod.bias.data.half()
            handles.append(mod.register_forward_pre_hook(_pre))

    return handles


def register_fp16_softmax_hooks(model: nn.Module, scope: str = "all") -> list:
    """Set softmax_dtype=torch.float16 on in-scope attention modules.

    Forces softmax to run in FP16 instead of the autocast default (FP32).
    On encoder SA (512×512 maps) this risks overflow — use with awareness.

    scope ∈ {"encoder", "decoder", "transformer", "all"}.
    """
    handles = []

    def _in_scope(fqn: str) -> bool:
        if scope == "all":
            return True
        if scope == "transformer":
            return "encoder" in fqn or "decoder" in fqn
        return scope in fqn

    for name, mod in model.named_modules():
        if isinstance(mod, (CustomMHA, ConditionalTransformerEncoderLayer)) and _in_scope(name):
            mod.softmax_dtype = torch.float16
            handles.append(_ModAttrHandle(mod, "softmax_dtype", None))

    return handles


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
