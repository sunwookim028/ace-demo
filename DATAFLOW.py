"""DATAFLOW.py — functional simulator of the RETR P2S1 ASIC pipeline.

End-to-end inference.  Executable; reproduces the same output as eval_ptq.py
run with:

    --scheme int8dq --component transformer --attn_int8
    --use_autocast_fp16 --fp16_half_backbone
    --fp16_ln all --fp16_softmax all --ln_input_bits 4

╔══════════════════════════════════════════════════════════════════════════╗
║ Dataflow typing discipline                                               ║
╠══════════════════════════════════════════════════════════════════════════╣
║ All activations flow in **FP16** from radar input to box outputs.        ║
║ The ONLY FP32 in the dataflow is the softmax row-sum accumulator         ║
║ (mirrors a wider HW accumulator behind the FP16 softmax FPU).            ║
║ INT8 MAC units use an INT32 accumulator internally; the DQT stage casts  ║
║ back to FP16 at the unit boundary.                                       ║
╚══════════════════════════════════════════════════════════════════════════╝

Hardware map
    CGRA — FP16 vector engine: backbone (ResNet18+FPN), input_proj 1×1 conv,
           class/bbox heads, box_affine_transformer, box geometry (sigmoid,
           cxcywh↔xyxy), sine/cos positional embeddings, top-K selection.
    CIM  — INT8 W8A8 systolic array: every transformer Linear and every
           attention BMM.  MAC output DQT'd to FP16 in place.

Unit tags used in code comments:
    QNT : FP16 → INT8 symmetric quantization (scale = absmax/127, round, clip)
    MAC : INT8 × INT8 → INT32 systolic matmul
    DQT : INT32 × (FP16 act_scale · FP16 weight_scale) → FP16 (fused at MAC output)
    MXQ : FP16 → FP4 E2M1 grid on blocks of 32 (fake-quant; LN input path)
    LN  : F.layer_norm with FP16 affine (preceded by MXQ)
    SFM : FP16 softmax with FP32 row-sum accumulator (the single FP32 exception)
    FPU : FP16 elementwise unit (adds, muls, ReLU, sigmoid, sin/cos)

Every Linear (weight-container: Int8DynLinear) and every attention BMM is
expanded INLINE in main() as the four-stage HW sequence:
    QNT(act, per-token)  →  MAC  →  DQT  →  FPU (bias add / residual / relu)
When the same FP16 activation feeds multiple MACs (e.g. src → {q_c, k_c, v_c}),
a single QNT is performed and the INT8 tensor is reused — this mirrors the CIM
dataflow and is called out with a "shared QNT" comment.
Every LayerNorm (weight-container: FP16BlockLN) is expanded inline as:
    MXQ(input, block=32) →  F.layer_norm(FP16 weight+bias) → FP16 out.

Run:
    python DATAFLOW.py --cuda --batch 1 --save-out /tmp/dataflow_out.pt
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# Import RETR from the sibling src/ tree.
# ──────────────────────────────────────────────────────────────────────────────

_REPO = Path(__file__).resolve().parent
_SRC = _REPO / "src"
sys.path.insert(0, str(_SRC))
os.chdir(_SRC)

from models import RETR                                            # noqa: E402
from models.module_retr import box_ops                             # noqa: E402
from models.module_retr.misc import (                              # noqa: E402
    nested_tensor_from_tensor_list,
    inverse_sigmoid,
)
from quantize_torchao import migrate_encoder_mha_state_dict        # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Symbol table (P2S1)
# ──────────────────────────────────────────────────────────────────────────────
D, H, FF = 256, 4, 2048       # hidden dim / heads / FFN dim
N_E, N_D = 512, 10            # encoder tokens (256 hor + 256 ver) / decoder queries
L_ENC, L_DEC = 6, 6
H_SA = D // H                 # 64  — decoder SA & half-head in encoder TPE concat
H_CA = 2 * D // H             # 128 — encoder SA head, decoder CA head (2d-wide)


# ═════════════════════════════════════════════════════════════════════════════
# HW PRIMITIVES
# ═════════════════════════════════════════════════════════════════════════════

# ── QNT ──────────────────────────────────────────────────────────────────────

def qnt_act(x_fp16: torch.Tensor, dim: int):
    """QNT unit.  FP16 → INT8 symmetric quantization along `dim`.

    scale  = absmax(x, dim=dim, keepdim) / 127       (FP16, per-row or per-col)
    x_i8   = round(clip(x / scale, -127, 127))       (INT8)

    Returns (x_i8, scale_fp16).  `dim=-1` gives per-token (used for act / lhs),
    `dim=-2` gives per-col (used for rhs of matmul along its K axis).
    """
    assert x_fp16.dtype == torch.float16, f"QNT expects FP16, got {x_fp16.dtype}"
    scale = x_fp16.abs().amax(dim=dim, keepdim=True).clamp(min=1e-6) / 127.0  # FP16
    x_i8  = (x_fp16 / scale).clamp(-127, 127).round().to(torch.int8)          # INT8
    return x_i8, scale


# ── MAC ──────────────────────────────────────────────────────────────────────

_INT8_MM_MIN_MN = 32          # torch._int_mm CUDA: requires M>16, N>16; K must be % 8


def int8_mac(a_i8: torch.Tensor, b_i8: torch.Tensor) -> torch.Tensor:
    """MAC unit.  INT8 × INT8 → INT32 systolic matmul.

    Accepts 2D [M,K]·[K,N] or 3D [B,M,K]·[B,K,N].  CPU path uses INT32 matmul
    (exact integer math).  CUDA path uses torch._int_mm per batch with a shape
    pad (the kernel requires M>16, N>16, K%8==0 — padded tail is sliced off).
    """
    squeezed = a_i8.dim() == 2
    if squeezed:
        a_i8, b_i8 = a_i8.unsqueeze(0), b_i8.unsqueeze(0)
    B, M, K = a_i8.shape
    _, _, N = b_i8.shape
    M_eff = max(M, _INT8_MM_MIN_MN)
    N_eff = max(N, _INT8_MM_MIN_MN)
    K_eff = ((K + 7) // 8) * 8
    if K_eff > K:
        a_i8 = F.pad(a_i8, (0, K_eff - K))
        b_i8 = F.pad(b_i8, (0, 0, 0, K_eff - K))
    if M_eff > M:
        a_i8 = F.pad(a_i8, (0, 0, 0, M_eff - M))
    if N_eff > N:
        b_i8 = F.pad(b_i8, (0, N_eff - N))
    if a_i8.is_cuda:
        out = torch.stack([torch._int_mm(a_i8[i].contiguous(), b_i8[i].contiguous())
                           for i in range(B)])                                # INT32
    else:
        out = torch.matmul(a_i8.to(torch.int32), b_i8.to(torch.int32))        # INT32 (exact)
    out = out[:, :M, :N]
    return out.squeeze(0) if squeezed else out


# ── DQT ──────────────────────────────────────────────────────────────────────

def dqt_fp16(acc_i32: torch.Tensor, s_a_fp16: torch.Tensor, s_b_fp16: torch.Tensor) -> torch.Tensor:
    """DQT unit.  INT32 × (FP16 act_scale · FP16 weight_scale) → FP16.

    The post-MAC dequant is the single HW stage where an INT32 accumulator meets
    two FP16 scales.  We widen to FP32 inside this unit (not in the dataflow) so
    the multiply doesn't overflow FP16 for large INT32 magnitudes, and cast back
    to FP16 at the unit boundary.
    """
    assert acc_i32.dtype == torch.int32
    return (acc_i32.float() * (s_a_fp16.float() * s_b_fp16.float())).to(torch.float16)


# ── MXQ (FP4 E2M1 block quant) ───────────────────────────────────────────────

_FP4_POS_VALS   = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_FP4_BOUNDARIES = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
_FP4_MAX = 6.0


def mx_fp4(x_fp16: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """MXQ unit.  Fake-quantize FP16 → FP4 E2M1 grid on blocks of `block_size`
    along the last dim.  Per-block scale is a power-of-two.  Output FP16.

    Exponent extraction (log2 / ceil) runs in FP32 internally (LUT math, not a
    dataflow tensor).
    """
    assert x_fp16.dtype == torch.float16
    orig_shape = x_fp16.shape
    x32 = x_fp16.float()                                           # unit-internal widen
    last = orig_shape[-1]
    pad = (block_size - last % block_size) % block_size
    if pad:
        x32 = F.pad(x32, (0, pad))
    blocks = x32.reshape(*x32.shape[:-1], -1, block_size)
    max_abs = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale   = torch.pow(2.0, torch.ceil(torch.log2(max_abs / _FP4_MAX)))
    scaled  = (blocks / scale).clamp(-_FP4_MAX, _FP4_MAX)
    idx     = torch.bucketize(scaled.abs().contiguous(), _FP4_BOUNDARIES.to(x_fp16.device))
    dq      = (scaled.sign() * _FP4_POS_VALS.to(x_fp16.device)[idx] * scale).reshape(
        *orig_shape[:-1], last + pad
    )
    if pad:
        dq = dq[..., :last]
    return dq.to(torch.float16)                                    # unit-output narrow


# ── LN (MXQ + F.layer_norm) ──────────────────────────────────────────────────

def fp16_ln(x_fp16: torch.Tensor, weight_fp16: torch.Tensor, bias_fp16: torch.Tensor,
            dim: int, block_size: int = 32) -> torch.Tensor:
    """LN unit.  MXQ(x, block_size) → F.layer_norm with FP16 affine.  FP16 in/out."""
    assert x_fp16.dtype == torch.float16
    x_mxq = mx_fp4(x_fp16, block_size=block_size)                  # MXQ FP16 → FP16 (FP4 grid)
    return F.layer_norm(x_mxq, (dim,), weight_fp16, bias_fp16)     # FPU FP16


# ── SFM (softmax with FP32 row-sum accumulator) ──────────────────────────────

def fp16_softmax_fp32_accum(x_fp16: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """SFM unit.  The single FP32 exception in the pipeline.

    Pipeline:
        xmax  = max(x, dim)                     FP16 reduction
        e     = exp(x - xmax)                   FP16 elementwise
        s_32  = sum(e, dim) — accumulated in FP32
        inv_s = (1/s_32).to(FP16)               reciprocal in FP32, narrow at unit boundary
        y     = e * inv_s                       FP16 multiply

    This matches the HW softmax FPU which uses a wider accumulator for stability
    but produces FP16 probabilities.
    """
    assert x_fp16.dtype == torch.float16
    xmax  = x_fp16.amax(dim=dim, keepdim=True)                     # FP16
    e     = (x_fp16 - xmax).exp()                                  # FP16
    s_32  = e.float().sum(dim=dim, keepdim=True)                   # FP32 accumulator
    inv_s = s_32.reciprocal().to(torch.float16)                    # FP16
    return e * inv_s                                               # FP16


# ── Sine/cos 3D positional embedding (used per decoder layer) ────────────────

def sineembed_3d(pos_fp16: torch.Tensor) -> torch.Tensor:
    """Sine/cos positional embedding for a 3D object center [*, 3] → [*, 256].
    All FP16. The constant dim_t LUT is FP16; values that overflow saturate and
    give zero contribution (same behavior on HW)."""
    assert pos_fp16.dtype == torch.float16
    scale, dim = 2 * math.pi, 86
    dim_t = 10000 ** (3 * (torch.arange(dim, dtype=pos_fp16.dtype, device=pos_fp16.device) // 3) / dim)
    outs = []
    for i in range(3):
        p = (pos_fp16[..., i] * scale).unsqueeze(-1) / dim_t        # FP16
        outs.append(torch.stack((p[..., 0::2].sin(), p[..., 1::2].cos()), dim=-1).flatten(-2))
    return torch.cat(outs, dim=-1)[..., :D]                         # FP16 [*, 256]


# ═════════════════════════════════════════════════════════════════════════════
# WEIGHT STORAGE CONTAINERS (used by load_state_dict)
# ═════════════════════════════════════════════════════════════════════════════

class Int8DynLinear(nn.Module):
    """INT8 W8A8 Linear.  Stores:
        w_i8     [out, in]   INT8     symmetric per-output-channel quantized weight
        w_scale  [out, 1]    FP16     per-output-channel FP16 scale
        bias     [out]       FP16

    forward() wraps the HW sequence:
        x(FP16) → QNT(per-token) → MAC(INT32) → DQT(→FP16) → + bias → FP16
    Used in main() directly; every call site is commented with FP16 shapes.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("w_i8",    torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("w_scale", torch.ones(out_features, 1, dtype=torch.float16))
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16)) if bias else None

    @classmethod
    def from_linear(cls, lin: nn.Linear) -> "Int8DynLinear":
        m = cls(lin.in_features, lin.out_features, bias=lin.bias is not None).to(lin.weight.device)
        w = lin.weight.data.float()
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127.0
        m.w_i8.copy_((w / scale).clamp(-127, 127).round().to(torch.int8))
        m.w_scale.copy_(scale.to(torch.float16))
        if lin.bias is not None:
            m.bias.data.copy_(lin.bias.data.to(torch.float16))
        return m

    def forward(self, x_fp16: torch.Tensor) -> torch.Tensor:
        in_shape = x_fp16.shape
        x2 = x_fp16.reshape(-1, in_shape[-1])                                 # FP16 [N, in]
        x_i8, s_x = qnt_act(x2, dim=-1)                                       # INT8 [N,in], s FP16 [N,1]
        acc = int8_mac(x_i8, self.w_i8.t().contiguous())                      # INT32 [N, out]
        y = dqt_fp16(acc, s_x, self.w_scale.t())                              # FP16 [N, out]
        if self.bias is not None:
            y = y + self.bias                                                 # FP16
        return y.reshape(*in_shape[:-1], self.out_features)


class FP16BlockLN(nn.Module):
    """FP4-block-quantized LayerNorm.  Stores:
        weight  [dim]  FP16
        bias    [dim]  FP16
    forward() = fp16_ln(x, weight, bias, dim, block_size).
    """

    def __init__(self, dim: int, block_size: int = 32):
        super().__init__()
        self.dim, self.block_size = dim, block_size
        self.weight = nn.Parameter(torch.ones(dim,  dtype=torch.float16))
        self.bias   = nn.Parameter(torch.zeros(dim, dtype=torch.float16))

    @classmethod
    def from_layernorm(cls, ln: nn.LayerNorm, block_size: int = 32) -> "FP16BlockLN":
        m = cls(ln.normalized_shape[0], block_size=block_size).to(ln.weight.device)
        m.weight.data.copy_(ln.weight.data.to(torch.float16))
        m.bias.data.copy_(ln.bias.data.to(torch.float16))
        return m

    def forward(self, x_fp16: torch.Tensor) -> torch.Tensor:
        return fp16_ln(x_fp16, self.weight, self.bias, self.dim, self.block_size)


# ═════════════════════════════════════════════════════════════════════════════
# Model mutation: FP16 cast of CGRA modules + INT8/FP16-LN swap of CIM modules
# ═════════════════════════════════════════════════════════════════════════════

def quantize_transformer_inplace(detr: nn.Module) -> None:
    """Walk `detr` and swap every transformer-scope nn.Linear → Int8DynLinear,
    nn.LayerNorm → FP16BlockLN, in place."""
    for parent_fqn, parent in detr.named_modules():
        for child_name, child in list(parent.named_children()):
            fqn = f"{parent_fqn}.{child_name}" if parent_fqn else child_name
            if not ("encoder" in fqn or "decoder" in fqn):
                continue
            if isinstance(child, nn.Linear):
                setattr(parent, child_name, Int8DynLinear.from_linear(child))
            elif isinstance(child, nn.LayerNorm):
                setattr(parent, child_name, FP16BlockLN.from_layernorm(child))


def fp16_cast_cgra(retr: nn.Module) -> None:
    """Cast the CGRA-mapped modules (backbone, input_proj, query_embed, heads,
    box_affine_transformer) to FP16 in place.  BatchNorm stats and affine params
    are cast too — CGRA runs BN in FP16."""
    core = retr.model.detr if hasattr(retr.model, "detr") else retr.model
    core.backbone.half()
    core.input_proj.half()
    core.input_proj_ver.half()
    core.query_embed.half()
    core.class_embed.half()
    core.bbox_embed.half()
    core.box_affine_transformer.half()


# ═════════════════════════════════════════════════════════════════════════════
# END-TO-END PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained",
                    default=str(_REPO / "logs/pretrained_model/p2s1_retr_detseg.pth"))
    ap.add_argument("--task", default="DET", choices=["DET", "DETSEG"])
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--h-w", nargs=2, type=int, default=[128, 256])
    ap.add_argument("--save-out", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Build RETR + load pretrained checkpoint.
    #    The shipped checkpoint is DETSEG (keys rooted at model.detr.*); we
    #    build in SEG mode to match, and unwrap DETR from DETRsegm below.
    # ─────────────────────────────────────────────────────────────────────────
    retr = RETR(task="SEG").to(device).eval()
    sd = torch.load(args.pretrained, map_location=device, weights_only=False)
    sd = migrate_encoder_mha_state_dict(sd)
    retr.load_state_dict(sd)
    detr = retr.model.detr if hasattr(retr.model, "detr") else retr.model

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Apply the recipe
    # ─────────────────────────────────────────────────────────────────────────
    fp16_cast_cgra(retr)                          # CGRA: backbone + input_proj + heads → FP16
    quantize_transformer_inplace(detr)            # CIM : encoder/decoder Linears → INT8dq, LN → FP16+MXQ

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Dummy inputs (horizontal + vertical radar maps), FP16 to CGRA
    # ─────────────────────────────────────────────────────────────────────────
    B = args.batch
    rawH, rawW = args.h_w
    hor = torch.randn(B, 4, rawH, rawW, device=device, dtype=torch.float16)   # FP16 [B,4,H,W]
    ver = torch.randn(B, 4, rawH, rawW, device=device, dtype=torch.float16)   # FP16 [B,4,H,W]

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Backbone (CGRA, FP16 native)
    #    ResNet18 → conv1(4-ch,7×7,s2) + BN + ReLU + maxpool → 4 residual stages
    #    (C2..C5 at strides 4,8,16,32; out channels 64,128,256,512).  The RETR
    #    backbone picks C5 only, then FPN=LastLevelMaxPool projects to 64 ch.
    #    Joiner appends a 2-D sine positional embedding (dim_t computed FP32,
    #    cast to FP16 at the CGRA output boundary below).
    # ─────────────────────────────────────────────────────────────────────────
    samples_h = nested_tensor_from_tensor_list(hor)                # NestedTensor(FP16 [B,4,H,W], mask[B,H,W])
    samples_v = nested_tensor_from_tensor_list(ver)
    feats_h, pos_h_raw = detr.backbone(samples_h)                  # feats: NestedTensor(FP16 [B,64,h,w]); pos: FP32 [B,256,h,w]
    feats_v, pos_v_raw = detr.backbone(samples_v)
    src_h_fp16, _ = feats_h[0].decompose()                         # FP16 [B, 64, h, w]
    src_v_fp16, _ = feats_v[0].decompose()
    pos_h_fp16 = pos_h_raw[0].to(torch.float16)                    # CVT FP32→FP16 at CGRA boundary; FP16 [B,256,h,w]
    pos_v_fp16 = pos_v_raw[0].to(torch.float16)

    # ─────────────────────────────────────────────────────────────────────────
    # 5. input_proj (1×1 Conv2d [64→256]) — CGRA FP16
    # ─────────────────────────────────────────────────────────────────────────
    src_proj_h = detr.input_proj(src_h_fp16)                       # FP16 [B, 256, h, w]
    src_proj_v = detr.input_proj_ver(src_v_fp16)                   # FP16 [B, 256, h, w]

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Top-K feature selection (CGRA, FP16)
    #    L2-norm over the 256 channels at each spatial location; pick the top
    #    256 tokens (= 16×16) from each view.
    # ─────────────────────────────────────────────────────────────────────────
    topk_fea_h, topk_pos_h, topk_fea_v, topk_pos_v = detr.topk_selection(
        src_proj_h, pos_h_fp16, src_proj_v, pos_v_fp16
    )                                                              # each FP16 [B, 256, 16, 16]

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Encoder input assembly — concat hor/ver tokens along the sequence axis
    # ─────────────────────────────────────────────────────────────────────────
    src = torch.cat([topk_fea_h.flatten(2).permute(2, 0, 1),       # FP16 [256, B, 256]
                     topk_fea_v.flatten(2).permute(2, 0, 1)],      # FP16 [256, B, 256]
                    dim=0).contiguous()                            # FP16 [N_E=512, B, 256]
    pos_embed = torch.cat([topk_pos_h.flatten(2).permute(2, 0, 1),
                           topk_pos_v.flatten(2).permute(2, 0, 1)],
                          dim=0).contiguous()                      # FP16 [N_E=512, B, 256]

    # ─────────────────────────────────────────────────────────────────────────
    # 8. Encoder ×6  — every nn.Linear is expanded inline as QNT→MAC→DQT→FPU.
    #    All activations flow in FP16; softmax row-sum is the one FP32 spot.
    #
    #    Shared QNT per layer:
    #       QNT(src)       serves q_c, k_c, v_c              (3 MACs)
    #       QNT(pos_embed) serves k_p, v_p, q_s              (3 MACs)
    #       QNT(q_concat)  serves self_attn_q_proj           (1 MAC)
    #       QNT(k_concat)  serves self_attn_k_proj           (1 MAC)
    #       QNT(v_concat)  serves self_attn_v_proj           (1 MAC)
    #       QNT(ctx)       serves self_attn_out_proj         (1 MAC)
    #       QNT(ffn_in)    serves linear1                    (1 MAC)
    #       QNT(ffn_mid)   serves linear2                    (1 MAC)
    #       + 4 QNTs inside the 2 attention BMMs (Q, Kᵀ, attn, V)
    # ─────────────────────────────────────────────────────────────────────────
    for i in range(L_ENC):
        L = detr.encoder.layers[i]                                 # ConditionalTransformerEncoderLayer (i of 6)

        # (a) Shared QNT on src and pos_embed (each feeds three 256→256 MACs)
        src_flat  = src.reshape(-1, D)                                                        # FP16 [N_E·B, 256]
        pos_flat  = pos_embed.reshape(-1, D)                                                  # FP16 [N_E·B, 256]
        src_i8, s_src = qnt_act(src_flat, dim=-1)                                             # QNT  INT8, FP16 scale
        pos_i8, s_pos = qnt_act(pos_flat, dim=-1)                                             # QNT  INT8, FP16 scale

        # (b) Six content/pos projections — 3× MAC on src, 3× MAC on pos_embed
        W = L.ca_qcontent_proj
        acc = int8_mac(src_i8, W.w_i8.t().contiguous())                                       # MAC  INT32 [N,256]
        q_c = (dqt_fp16(acc, s_src, W.w_scale.t()) + W.bias).reshape(N_E, B, D)               # DQT+FPU → FP16 [N_E,B,256]

        W = L.ca_kcontent_proj
        acc = int8_mac(src_i8, W.w_i8.t().contiguous())                                       # MAC
        k_c = (dqt_fp16(acc, s_src, W.w_scale.t()) + W.bias).reshape(N_E, B, D)               # DQT+FPU

        W = L.ca_v_proj
        acc = int8_mac(src_i8, W.w_i8.t().contiguous())                                       # MAC
        v_c = (dqt_fp16(acc, s_src, W.w_scale.t()) + W.bias).reshape(N_E, B, D)               # DQT+FPU

        W = L.ca_kpos_proj
        acc = int8_mac(pos_i8, W.w_i8.t().contiguous())                                       # MAC
        k_p = (dqt_fp16(acc, s_pos, W.w_scale.t()) + W.bias).reshape(N_E, B, D)               # DQT+FPU

        W = L.ca_vpos_proj
        acc = int8_mac(pos_i8, W.w_i8.t().contiguous())                                       # MAC
        v_p = (dqt_fp16(acc, s_pos, W.w_scale.t()) + W.bias).reshape(N_E, B, D)               # DQT+FPU

        W = L.ca_qpos_sine_proj
        acc = int8_mac(pos_i8, W.w_i8.t().contiguous())                                       # MAC
        q_s = (dqt_fp16(acc, s_pos, W.w_scale.t()) + W.bias).reshape(N_E, B, D)               # DQT+FPU

        # (c) TPE concat: per-head content || pos halves → 2d-wide [N_E, B, 512]
        q_cat = torch.cat([q_c.view(N_E, B, H, H_SA), q_s.view(N_E, B, H, H_SA)], dim=3).reshape(N_E, B, 2 * D)
        k_cat = torch.cat([k_c.view(N_E, B, H, H_SA), k_p.view(N_E, B, H, H_SA)], dim=3).reshape(N_E, B, 2 * D)
        v_cat = torch.cat([v_c.view(N_E, B, H, H_SA), v_p.view(N_E, B, H, H_SA)], dim=3).reshape(N_E, B, 2 * D)

        # (d) SA Q/K/V projections — 3 independent QNTs (different tensors), 3× 512→512 MAC
        qcat_flat = q_cat.reshape(-1, 2 * D)                                                  # FP16 [N_E·B, 512]
        kcat_flat = k_cat.reshape(-1, 2 * D)
        vcat_flat = v_cat.reshape(-1, 2 * D)
        qcat_i8, s_qcat = qnt_act(qcat_flat, dim=-1)                                          # QNT
        kcat_i8, s_kcat = qnt_act(kcat_flat, dim=-1)                                          # QNT
        vcat_i8, s_vcat = qnt_act(vcat_flat, dim=-1)                                          # QNT

        W = L.self_attn_q_proj
        acc = int8_mac(qcat_i8, W.w_i8.t().contiguous())                                      # MAC  INT32 [N,512]
        q   = (dqt_fp16(acc, s_qcat, W.w_scale.t()) + W.bias).reshape(N_E, B, 2 * D)          # DQT+FPU
        q   = q * (H_CA ** -0.5)                                                              # FPU  scale

        W = L.self_attn_k_proj
        acc = int8_mac(kcat_i8, W.w_i8.t().contiguous())                                      # MAC
        k   = (dqt_fp16(acc, s_kcat, W.w_scale.t()) + W.bias).reshape(N_E, B, 2 * D)          # DQT+FPU

        W = L.self_attn_v_proj
        acc = int8_mac(vcat_i8, W.w_i8.t().contiguous())                                      # MAC
        v   = (dqt_fp16(acc, s_vcat, W.w_scale.t()) + W.bias).reshape(N_E, B, 2 * D)          # DQT+FPU

        # (e) Multi-head reshape to per-head sequences
        q = q.view(N_E, B * H, H_CA).transpose(0, 1).contiguous()                             # FP16 [B·H=4, 512, 128]
        k = k.view(N_E, B * H, H_CA).transpose(0, 1).contiguous()                             # FP16 [B·H, 512, 128]
        v = v.view(N_E, B * H, H_CA).transpose(0, 1).contiguous()                             # FP16 [B·H, 512, 128]

        # (f) Attention score BMM: Q · Kᵀ      [B·H,512,128] · [B·H,128,512] → [B·H,512,512]
        kT      = k.transpose(1, 2).contiguous()                                              # FP16 [B·H, 128, 512]
        q_i8,  s_q  = qnt_act(q,  dim=-1)                                                     # QNT  per-token
        kT_i8, s_kT = qnt_act(kT, dim=-2)                                                     # QNT  per-col
        acc_qk  = int8_mac(q_i8, kT_i8)                                                       # MAC  INT32
        scores  = dqt_fp16(acc_qk, s_q, s_kT)                                                 # DQT  FP16

        # (g) Softmax — FP16 body, FP32 sum accumulator (single FP32 exception)
        attn = fp16_softmax_fp32_accum(scores, dim=-1)                                        # SFM  FP16

        # (h) Context BMM: attn · V             [B·H,512,512] · [B·H,512,128] → [B·H,512,128]
        a_i8, s_a = qnt_act(attn, dim=-1)                                                     # QNT  per-token
        v_i8, s_v = qnt_act(v,    dim=-2)                                                     # QNT  per-col
        acc_av    = int8_mac(a_i8, v_i8)                                                      # MAC  INT32
        ctx       = dqt_fp16(acc_av, s_a, s_v)                                                # DQT  FP16
        ctx       = ctx.transpose(0, 1).contiguous().reshape(N_E, B, 2 * D)                   # FP16 [N_E, B, 512]

        # (i) Out-projection — 512→512 MAC, keep the first D=256 channels (TPE inner slice)
        ctx_flat = ctx.reshape(-1, 2 * D)                                                     # FP16 [N_E·B, 512]
        ctx_i8, s_ctx = qnt_act(ctx_flat, dim=-1)                                             # QNT
        W = L.self_attn_out_proj
        acc = int8_mac(ctx_i8, W.w_i8.t().contiguous())                                       # MAC  INT32 [N, 512]
        out = (dqt_fp16(acc, s_ctx, W.w_scale.t()) + W.bias).reshape(N_E, B, 2 * D)[..., :D]  # DQT+FPU, slice → [N_E,B,256]

        # (j) Residual + LN1 — inline MXQ + F.layer_norm (FP16 affine)
        src_r  = src + out                                                                    # FPU  residual
        src_mx = mx_fp4(src_r, block_size=32)                                                 # MXQ  FP16→FP4E2M1 block=32
        src    = F.layer_norm(src_mx, (D,), L.norm1.weight, L.norm1.bias)                     # LN   FP16

        # (k) FFN: linear1 (256→2048) + ReLU + linear2 (2048→256)
        ffn_flat = src.reshape(-1, D)                                                         # FP16 [N_E·B, 256]
        ffn_i8, s_ffn = qnt_act(ffn_flat, dim=-1)                                             # QNT
        W = L.linear1
        acc = int8_mac(ffn_i8, W.w_i8.t().contiguous())                                       # MAC  INT32 [N, 2048]
        y   = (dqt_fp16(acc, s_ffn, W.w_scale.t()) + W.bias)                                  # DQT+FPU bias → FP16 [N, 2048]
        y   = F.relu(y)                                                                       # FPU  ReLU
        y_i8, s_y = qnt_act(y, dim=-1)                                                        # QNT  on ReLU output
        W = L.linear2
        acc = int8_mac(y_i8, W.w_i8.t().contiguous())                                         # MAC  INT32 [N, 256]
        y   = (dqt_fp16(acc, s_y, W.w_scale.t()) + W.bias).reshape(N_E, B, D)                 # DQT+FPU bias

        # (l) Residual + LN2
        src_r  = src + y                                                                      # FPU residual
        src_mx = mx_fp4(src_r, block_size=32)                                                 # MXQ
        src    = F.layer_norm(src_mx, (D,), L.norm2.weight, L.norm2.bias)                     # LN

    memory = src                                                                              # FP16 [N_E=512, B, D=256]

    # ─────────────────────────────────────────────────────────────────────────
    # 9. Decoder preamble
    #    query_embed: nn.Embedding(N_D=10, D=256) FP16
    #    ref_point_head: MLP D → D → 3   (INT8dq Linears, sigmoid head)
    # ─────────────────────────────────────────────────────────────────────────
    query_embed = detr.query_embed.weight.unsqueeze(1).repeat(1, B, 1)   # FP16 [N_D=10, B, 256]
    tgt = torch.zeros_like(query_embed)                                  # FP16 [N_D, B, 256]

    # ref_point_head MLP (2 linears, ReLU in between): 256→256 + ReLU + 256→3
    qe_flat = query_embed.reshape(-1, D)                                 # FP16 [N_D·B, 256]
    qe_i8, s_qe = qnt_act(qe_flat, dim=-1)                               # QNT
    W = detr.decoder.ref_point_head.layers[0]
    acc = int8_mac(qe_i8, W.w_i8.t().contiguous())                       # MAC  INT32 [N, 256]
    rp  = (dqt_fp16(acc, s_qe, W.w_scale.t()) + W.bias)                  # DQT+FPU → FP16 [N, 256]
    rp  = F.relu(rp)                                                     # FPU ReLU
    rp_i8, s_rp = qnt_act(rp, dim=-1)                                    # QNT
    W = detr.decoder.ref_point_head.layers[1]
    acc = int8_mac(rp_i8, W.w_i8.t().contiguous())                       # MAC  INT32 [N, 3]
    rp  = (dqt_fp16(acc, s_rp, W.w_scale.t()) + W.bias).reshape(N_D, B, 3)  # DQT+FPU → FP16 [N_D,B,3]
    reference_points = rp.sigmoid().transpose(0, 1)                      # FPU sigmoid → FP16 [B, N_D, 3]

    # ─────────────────────────────────────────────────────────────────────────
    # 10. Decoder ×6 — every nn.Linear expanded inline as QNT→MAC→DQT→FPU.
    #
    #     Shared QNT per layer:
    #       SA:  QNT(output)      serves sa_qcontent_proj, sa_kcontent_proj, sa_v_proj  (3 MACs)
    #            QNT(query_embed) serves sa_qpos_proj, sa_kpos_proj (+ ca_qpos on layer 0)
    #       CA:  QNT(memory)      serves ca_kcontent_proj, ca_v_proj  (2 MACs)
    #            QNT(output)      serves ca_qcontent_proj (single, re-QNT'd after LN1)
    #            QNT(pos_embed)   serves ca_kpos_proj     (single)
    #            QNT(sine_emb)    serves ca_qpos_sine_proj (single)
    #       FFN: QNT(ln2_out) → linear1;  QNT(relu_out) → linear2
    # ─────────────────────────────────────────────────────────────────────────
    output = tgt                                                          # FP16 [N_D, B, 256]
    intermediate = []

    # QNT(query_embed) is constant across layers (query_embed doesn't change), but
    # we keep the shared-QNT pattern inside each layer for dataflow transparency.
    for i in range(L_DEC):
        L = detr.decoder.layers[i]                                        # ConditionalTransformerDecoderLayer (i of 6)

        # (a) Reference-point sine embedding (per-layer; (b) may scale it)
        obj_center = reference_points[..., :3].transpose(0, 1)            # FP16 [N_D, B, 3]
        sine_emb   = sineembed_3d(obj_center)                             # FPU sin/cos → FP16 [N_D, B, 256]

        # (b) Query-scale TPE (layers i>0): MLP (256→256 + ReLU + 256→256) × sine_emb
        if i > 0:
            qs_flat = output.reshape(-1, D)                                                       # FP16 [N_D·B, 256]
            qs_i8, s_qs = qnt_act(qs_flat, dim=-1)                                                # QNT
            W = detr.decoder.query_scale.layers[0]
            acc = int8_mac(qs_i8, W.w_i8.t().contiguous())                                        # MAC  INT32 [N,256]
            qs  = (dqt_fp16(acc, s_qs, W.w_scale.t()) + W.bias)                                   # DQT+FPU bias
            qs  = F.relu(qs)                                                                      # FPU ReLU
            qs2_i8, s_qs2 = qnt_act(qs, dim=-1)                                                   # QNT
            W = detr.decoder.query_scale.layers[1]
            acc = int8_mac(qs2_i8, W.w_i8.t().contiguous())                                       # MAC
            pos_xform = (dqt_fp16(acc, s_qs2, W.w_scale.t()) + W.bias).reshape(N_D, B, D)         # DQT+FPU bias
            sine_emb = sine_emb * pos_xform                                                       # FPU mul

        # ── (c) Self-attention ───────────────────────────────────────────────
        # Shared QNT(output) feeds sa_qcontent/sa_kcontent/sa_v ; shared QNT(query_embed) feeds sa_qpos/sa_kpos
        out_flat = output.reshape(-1, D)                                                          # FP16 [N_D·B, 256]
        qe_flat  = query_embed.reshape(-1, D)                                                     # FP16 [N_D·B, 256]
        out_i8, s_out = qnt_act(out_flat, dim=-1)                                                 # QNT shared (3 MACs)
        qe_i8,  s_qe  = qnt_act(qe_flat,  dim=-1)                                                 # QNT shared (2 MACs; 3 if layer 0 CA also reuses — we re-QNT in CA for locality)

        W = L.sa_qcontent_proj
        acc = int8_mac(out_i8, W.w_i8.t().contiguous())                                           # MAC
        qc_sa = (dqt_fp16(acc, s_out, W.w_scale.t()) + W.bias).reshape(N_D, B, D)                 # DQT+FPU

        W = L.sa_kcontent_proj
        acc = int8_mac(out_i8, W.w_i8.t().contiguous())                                           # MAC
        kc_sa = (dqt_fp16(acc, s_out, W.w_scale.t()) + W.bias).reshape(N_D, B, D)                 # DQT+FPU

        W = L.sa_v_proj
        acc = int8_mac(out_i8, W.w_i8.t().contiguous())                                           # MAC
        v_sa_pre = (dqt_fp16(acc, s_out, W.w_scale.t()) + W.bias).reshape(N_D, B, D)              # DQT+FPU

        W = L.sa_qpos_proj
        acc = int8_mac(qe_i8, W.w_i8.t().contiguous())                                            # MAC
        qp_sa = (dqt_fp16(acc, s_qe, W.w_scale.t()) + W.bias).reshape(N_D, B, D)                  # DQT+FPU

        W = L.sa_kpos_proj
        acc = int8_mac(qe_i8, W.w_i8.t().contiguous())                                            # MAC
        kp_sa = (dqt_fp16(acc, s_qe, W.w_scale.t()) + W.bias).reshape(N_D, B, D)                  # DQT+FPU

        # Q/K assembly (adds are FPU ops) + pre-scale on Q
        q_sa = (qc_sa + qp_sa) * (H_SA ** -0.5)                                                   # FPU add + scale
        k_sa = kc_sa + kp_sa                                                                      # FPU add

        # per-head reshape
        q_sa = q_sa.view(N_D, B * H, H_SA).transpose(0, 1).contiguous()                           # FP16 [B·H, 10, 64]
        k_sa = k_sa.view(N_D, B * H, H_SA).transpose(0, 1).contiguous()                           # FP16 [B·H, 10, 64]
        v_sa = v_sa_pre.view(N_D, B * H, H_SA).transpose(0, 1).contiguous()                       # FP16 [B·H, 10, 64]

        # SA score BMM: Q·Kᵀ  [B·H,10,64]·[B·H,64,10] → [B·H,10,10]
        kT_sa           = k_sa.transpose(1, 2).contiguous()                                       # FP16 [B·H, 64, 10]
        qsa_i8,  s_qsa  = qnt_act(q_sa,  dim=-1)                                                  # QNT per-token
        ksaT_i8, s_ksaT = qnt_act(kT_sa, dim=-2)                                                  # QNT per-col
        acc_sa_qk       = int8_mac(qsa_i8, ksaT_i8)                                               # MAC
        scores_sa       = dqt_fp16(acc_sa_qk, s_qsa, s_ksaT)                                      # DQT
        attn_sa         = fp16_softmax_fp32_accum(scores_sa, dim=-1)                              # SFM

        # SA context BMM: attn·V  [B·H,10,10]·[B·H,10,64] → [B·H,10,64]
        asa_i8, s_asa = qnt_act(attn_sa, dim=-1)                                                  # QNT
        vsa_i8, s_vsa = qnt_act(v_sa,    dim=-2)                                                  # QNT per-col
        acc_sa_av     = int8_mac(asa_i8, vsa_i8)                                                  # MAC
        ctx_sa        = dqt_fp16(acc_sa_av, s_asa, s_vsa)                                         # DQT
        ctx_sa        = ctx_sa.transpose(0, 1).contiguous().reshape(N_D, B, D)                    # FP16 [N_D, B, 256]

        # SA out-proj: 256→256 MAC
        ctx_sa_flat = ctx_sa.reshape(-1, D)
        ctxsa_i8, s_ctxsa = qnt_act(ctx_sa_flat, dim=-1)                                          # QNT
        W = L.self_attn.out_proj
        acc = int8_mac(ctxsa_i8, W.w_i8.t().contiguous())                                         # MAC
        sa_out = (dqt_fp16(acc, s_ctxsa, W.w_scale.t()) + W.bias).reshape(N_D, B, D)              # DQT+FPU

        # (d) Residual + LN1 — inline MXQ + F.layer_norm
        out_r  = output + sa_out                                                                  # FPU residual
        out_mx = mx_fp4(out_r, block_size=32)                                                     # MXQ
        output = F.layer_norm(out_mx, (D,), L.norm1.weight, L.norm1.bias)                         # LN

        # ── (e) Cross-attention ──────────────────────────────────────────────
        # Shared QNT(memory) feeds ca_kcontent + ca_v ; QNT(output) re-taken post-LN1 for ca_qcontent
        out_flat = output.reshape(-1, D)                                                          # FP16 [N_D·B, 256]
        mem_flat = memory.reshape(-1, D)                                                          # FP16 [N_E·B, 256]
        pos_flat = pos_embed.reshape(-1, D)                                                       # FP16 [N_E·B, 256]
        sine_flat = sine_emb.reshape(-1, D)                                                       # FP16 [N_D·B, 256]

        out_i8,  s_out  = qnt_act(out_flat,  dim=-1)                                              # QNT for ca_qcontent
        mem_i8,  s_mem  = qnt_act(mem_flat,  dim=-1)                                              # QNT shared (ca_kcontent, ca_v)
        pos_i8,  s_pos  = qnt_act(pos_flat,  dim=-1)                                              # QNT for ca_kpos
        sine_i8, s_sine = qnt_act(sine_flat, dim=-1)                                              # QNT for ca_qpos_sine

        W = L.ca_qcontent_proj
        acc = int8_mac(out_i8, W.w_i8.t().contiguous())                                           # MAC
        qc_ca = (dqt_fp16(acc, s_out, W.w_scale.t()) + W.bias).reshape(N_D, B, D)                 # DQT+FPU

        W = L.ca_kcontent_proj
        acc = int8_mac(mem_i8, W.w_i8.t().contiguous())                                           # MAC
        kc_ca = (dqt_fp16(acc, s_mem, W.w_scale.t()) + W.bias).reshape(N_E, B, D)                 # DQT+FPU

        W = L.ca_v_proj
        acc = int8_mac(mem_i8, W.w_i8.t().contiguous())                                           # MAC
        v_ca_pre = (dqt_fp16(acc, s_mem, W.w_scale.t()) + W.bias).reshape(N_E, B, D)              # DQT+FPU

        W = L.ca_kpos_proj
        acc = int8_mac(pos_i8, W.w_i8.t().contiguous())                                           # MAC
        kp_ca = (dqt_fp16(acc, s_pos, W.w_scale.t()) + W.bias).reshape(N_E, B, D)                 # DQT+FPU

        W = L.ca_qpos_sine_proj
        acc = int8_mac(sine_i8, W.w_i8.t().contiguous())                                          # MAC
        q_sine = (dqt_fp16(acc, s_sine, W.w_scale.t()) + W.bias).reshape(N_D, B, D)               # DQT+FPU

        if i == 0:
            # Only layer 0 uses ca_qpos_proj on query_embed (reuses the QNT from earlier SA-stage block)
            qe_flat = query_embed.reshape(-1, D)
            qe_i8, s_qe = qnt_act(qe_flat, dim=-1)                                                # QNT (single-use here)
            W = L.ca_qpos_proj
            acc = int8_mac(qe_i8, W.w_i8.t().contiguous())                                        # MAC
            qp_ca = (dqt_fp16(acc, s_qe, W.w_scale.t()) + W.bias).reshape(N_D, B, D)              # DQT+FPU
            q_ca_base = qc_ca + qp_ca                                                             # FPU add
        else:
            q_ca_base = qc_ca                                                                     # FP16 [N_D, B, 256]
        k_ca_base = kc_ca + kp_ca                                                                 # FPU add

        # 2d-wide TPE concat (content || pos-feature) on Q and K → H_CA=128 per head
        q_ca = torch.cat([q_ca_base.view(N_D, B, H, H_SA),
                          q_sine.view(N_D, B, H, H_SA)], dim=3).reshape(N_D, B, 2 * D)            # FP16 [N_D,B,512]
        k_ca = torch.cat([k_ca_base.view(N_E, B, H, H_SA),
                          kp_ca.view(N_E, B, H, H_SA)],  dim=3).reshape(N_E, B, 2 * D)            # FP16 [N_E,B,512]

        q_ca = q_ca * (H_CA ** -0.5)                                                              # FPU scale
        q_ca = q_ca.view(N_D, B * H, H_CA).transpose(0, 1).contiguous()                           # FP16 [B·H, 10, 128]
        k_ca = k_ca.view(N_E, B * H, H_CA).transpose(0, 1).contiguous()                           # FP16 [B·H, 512, 128]
        v_ca = v_ca_pre.view(N_E, B * H, H_SA).transpose(0, 1).contiguous()                       # FP16 [B·H, 512, 64]

        # CA score BMM: Q·Kᵀ  [B·H,10,128]·[B·H,128,512] → [B·H,10,512]
        kT_ca           = k_ca.transpose(1, 2).contiguous()                                       # FP16 [B·H, 128, 512]
        qca_i8,  s_qca  = qnt_act(q_ca,  dim=-1)                                                  # QNT
        kcaT_i8, s_kcaT = qnt_act(kT_ca, dim=-2)                                                  # QNT
        acc_ca_qk       = int8_mac(qca_i8, kcaT_i8)                                               # MAC
        scores_ca       = dqt_fp16(acc_ca_qk, s_qca, s_kcaT)                                      # DQT
        attn_ca         = fp16_softmax_fp32_accum(scores_ca, dim=-1)                              # SFM

        # CA context BMM: attn·V  [B·H,10,512]·[B·H,512,64] → [B·H,10,64]
        aca_i8, s_aca = qnt_act(attn_ca, dim=-1)                                                  # QNT
        vca_i8, s_vca = qnt_act(v_ca,    dim=-2)                                                  # QNT
        acc_ca_av     = int8_mac(aca_i8, vca_i8)                                                  # MAC
        ctx_ca        = dqt_fp16(acc_ca_av, s_aca, s_vca)                                         # DQT
        ctx_ca        = ctx_ca.transpose(0, 1).contiguous().reshape(N_D, B, D)                    # FP16 [N_D, B, 256]

        # CA out-proj: 256→256 MAC (cross_attn.out_proj)
        ctx_ca_flat = ctx_ca.reshape(-1, D)
        ctxca_i8, s_ctxca = qnt_act(ctx_ca_flat, dim=-1)                                          # QNT
        W = L.cross_attn.out_proj
        acc = int8_mac(ctxca_i8, W.w_i8.t().contiguous())                                         # MAC
        ca_out = (dqt_fp16(acc, s_ctxca, W.w_scale.t()) + W.bias).reshape(N_D, B, D)              # DQT+FPU

        # (f) Residual + LN2
        out_r  = output + ca_out                                                                  # FPU residual
        out_mx = mx_fp4(out_r, block_size=32)                                                     # MXQ
        output = F.layer_norm(out_mx, (D,), L.norm2.weight, L.norm2.bias)                         # LN

        # (g) FFN: linear1 (256→2048) + ReLU + linear2 (2048→256)
        ffn_flat = output.reshape(-1, D)
        ffn_i8, s_ffn = qnt_act(ffn_flat, dim=-1)                                                 # QNT
        W = L.linear1
        acc = int8_mac(ffn_i8, W.w_i8.t().contiguous())                                           # MAC
        y   = (dqt_fp16(acc, s_ffn, W.w_scale.t()) + W.bias)                                      # DQT+FPU
        y   = F.relu(y)                                                                           # FPU ReLU
        y_i8, s_y = qnt_act(y, dim=-1)                                                            # QNT
        W = L.linear2
        acc = int8_mac(y_i8, W.w_i8.t().contiguous())                                             # MAC
        y   = (dqt_fp16(acc, s_y, W.w_scale.t()) + W.bias).reshape(N_D, B, D)                     # DQT+FPU

        # (h) Residual + LN3
        out_r  = output + y                                                                       # FPU residual
        out_mx = mx_fp4(out_r, block_size=32)                                                     # MXQ
        output = F.layer_norm(out_mx, (D,), L.norm3.weight, L.norm3.bias)                         # LN

        # (i) Final decoder norm (MXQ + LN) applied to every intermediate output
        inter_mx = mx_fp4(output, block_size=32)                                                  # MXQ
        inter    = F.layer_norm(inter_mx, (D,),
                                detr.decoder.norm.weight, detr.decoder.norm.bias)                 # LN
        intermediate.append(inter)                                                                # FP16 [N_D, B, 256]

    hs = torch.stack(intermediate).transpose(1, 2)                                                # FP16 [L_DEC=6, B, N_D=10, D=256]

    # ─────────────────────────────────────────────────────────────────────────
    # 11. Heads (CGRA, FP16)
    #     class_embed: Linear(D → n_classes+1) FP16
    #     bbox_embed : MLP(D → D → D → 6) FP16 (3 Linear layers + ReLU)
    #     reference_points are added to the first pos_dim box coords (inverse_sigmoid domain)
    # ─────────────────────────────────────────────────────────────────────────
    ref_pre_sig = inverse_sigmoid(reference_points)                       # FP16 [B, N_D, 3]  FPU log
    outputs_coords = []
    for lvl in range(hs.shape[0]):
        tmp = detr.bbox_embed(hs[lvl])                                    # FP16 [B, N_D, 6]  3× Linear FP16
        tmp[..., :detr.pos_dim] = tmp[..., :detr.pos_dim] + ref_pre_sig   # FP16  FPU
        outputs_coords.append(tmp.sigmoid())                              # FP16  FPU
    outputs_coord = torch.stack(outputs_coords)                           # FP16 [L_DEC, B, N_D, 6]
    outputs_class = detr.class_embed(hs)                                  # FP16 [L_DEC, B, N_D, n_classes+1]

    pred_logits = outputs_class[-1]                                       # FP16 [B, N_D, n_classes+1]
    pred_boxes  = outputs_coord[-1]                                       # FP16 [B, N_D, 6]

    # ─────────────────────────────────────────────────────────────────────────
    # 12. Plane decomposition + box_affine_transformer (CGRA, FP16)
    # ─────────────────────────────────────────────────────────────────────────
    bbox = box_ops.box_cxcyczwhd_to_xyzxyz(pred_boxes.reshape(-1, 6)).reshape(pred_boxes.shape)  # FP16 [B, N_D, 6]
    hbox = torch.stack([b[:, [0, 2, 3, 5]] for b in bbox])                # FP16 [B, N_D, 4]
    vbox = torch.stack([b[:, [1, 2, 4, 5]] for b in bbox])                # FP16 [B, N_D, 4]
    _, ibox = detr.calc_v_props(hbox, alignment=False, v_props=vbox, normed=True)
    ibox = torch.stack(ibox)                                              # FP16 [B, N_D, 4]
    pred_hboxes = box_ops.box_xyxy_to_cxcywh(hbox)                        # FP16
    pred_vboxes = box_ops.box_xyxy_to_cxcywh(vbox)                        # FP16
    iboxc       = box_ops.box_xyxy_to_cxcywh(ibox)                        # FP16

    bat_in  = torch.cat((iboxc, pred_boxes), dim=-1).view(-1, detr.num_queries)        # FP16 [B*N_D, 10]
    bat_out = detr.box_affine_transformer(bat_in).view(-1, detr.num_queries, 4)        # FP16 [B, N_D, 4]   Linear+BN1d+LReLU×2+Linear
    pred_iboxes = torch.sigmoid(iboxc - bat_out) + 1e-5                                # FP16 [B, N_D, 4]

    # ─────────────────────────────────────────────────────────────────────────
    # 13. Output dict (mirrors DETR.forward) + pixel-scaled augmented boxes
    # ─────────────────────────────────────────────────────────────────────────
    out = {
        "pred_logits":  pred_logits,
        "pred_boxes":   pred_boxes,
        "pred_hboxes":  pred_hboxes,
        "pred_vboxes":  pred_vboxes,
        "pred_iboxes":  pred_iboxes,
        "hs":           hs,
        "memory":       memory,
        "reference":    reference_points,
    }
    def _cxcywh_to_xyxy_scaled(t, w_scale, h_scale):
        t = t.clone()
        xy = box_ops.box_cxcywh_to_xyxy(t.reshape(-1, 4)).reshape(t.shape)
        xy[:, :, [0, 2]] *= w_scale
        xy[:, :, [1, 3]] *= h_scale
        return xy
    out["pred_hboxes_aug"] = _cxcywh_to_xyxy_scaled(pred_hboxes, detr.w, detr.h)
    out["pred_vboxes_aug"] = _cxcywh_to_xyxy_scaled(pred_vboxes, detr.w, detr.h)
    out["proj_boxes"]      = _cxcywh_to_xyxy_scaled(pred_iboxes, detr.iw, detr.ih)

    # ── Report ───────────────────────────────────────────────────────────────
    print(f"device={device}  task={args.task}  B={B}  in=(4,{rawH},{rawW})")
    for k, t in out.items():
        if isinstance(t, torch.Tensor):
            print(f"  {k:18s}  shape={tuple(t.shape)}  dtype={str(t.dtype).split('.')[-1]}  "
                  f"|x|max={t.abs().max().item():.4f}")

    if args.save_out is not None:
        torch.save({k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in out.items()}, args.save_out)
        print(f"\nsaved → {args.save_out}")

    print("\nOK — DATAFLOW.py end-to-end forward complete.")


if __name__ == "__main__":
    with torch.no_grad():
        main()
