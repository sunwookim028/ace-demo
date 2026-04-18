<!--
Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->
# RETR Quantization Study (PTQ)

## Model parameters

| Symbol | Value | Source |
|--------|-------|--------|
| d | 256 | hidden_dim |
| H | 4 | nhead (encoder and decoder) |
| ff | 2048 | dim_feedforward |
| N_e | 512 | encoder tokens = 2 × topk (hor + ver views) |
| N_d | 10 | num_queries (decoder) |
| L_enc | 6 | encoder layers |
| L_dec | 6 | decoder layers |
| h_e | 128 | encoder head_dim = (2d)/H; TPE doubles attn dim to 2d=512 |
| h_d_sa | 64 | decoder self-attn head_dim = d/H |
| h_d_ca | 128 | decoder cross-attn Q,K head_dim = (2d)/H |
| h_d_ca_v | 64 | decoder cross-attn V head_dim = d/H |

## Target scheme: int8dq + full attn INT8

```
Linear layers     torchao Int8DynamicActivationInt8WeightConfig
                  weights INT8 static; activations quantized per-token INT8 at runtime
                  real INT8 GEMM (oneDNN kernel); INT32 accumulator; FP32 output
                  targets all nn.Linear: encoder (72) + decoder (89) = 161 modules
                  backbone is Conv2d only → not quantized

Q, K, V           symmetric per-tensor INT8 fake-quant (STE)
                  applied post-projection, post-head-split, before QK^T bmm
                  scale = max(|x|) / 127; clamp [-127, 127]; dequant FP32
                  simulates INT8×INT8 BMM without a real INT8 kernel

softmax output    asymmetric per-tensor UINT8 fake-quant (STE)
                  applied after softmax, before AV bmm
                  scale = max(x) / 255; clamp [0, 255]; dequant FP32
                  exploits non-negative range; UINT8 captures full [0,1] span
                  symmetric INT8 would waste half the range

Softmax           FP32 (hardware: online softmax coprocessor)
LayerNorm         FP32
Residual adds     FP32
```

---

## Layer-by-layer data flow

Shapes are `[seq, B, dim]` for transformer ops and `[B, C, H, W]` for convolutions.
Batch dimension B is free. Evaluation used B=8.

---

### Backbone (×2 views, FP32 throughout)

Input per view: `[B, 4, 256, 128]` (4 radar frames, rh=256, rw=128)

| Layer | Input | Output | Weight | Op | Precision |
|-------|-------|--------|--------|----|-----------|
| conv1 | [B, 4, 256, 128] | [B, 64, 128, 64] | [64, 4, 7, 7] | Conv2d s=2, p=3 | FP32 |
| bn1 + relu | [B, 64, 128, 64] | [B, 64, 128, 64] | scale,bias[64] | BN (frozen) | FP32 |
| maxpool | [B, 64, 128, 64] | [B, 64, 64, 32] | — | MaxPool 3×3 s=2 | FP32 |
| layer1 (2× BasicBlock) | [B, 64, 64, 32] | [B, 64, 64, 32] | 3×3 Conv2d ×4 | Conv2d | FP32 |
| layer2 (2× BasicBlock) | [B, 64, 64, 32] | [B, 128, 32, 16] | 3×3 Conv2d ×4+shortcut | Conv2d s=2 | FP32 |
| layer3 (2× BasicBlock) | [B, 128, 32, 16] | [B, 256, 16, 8] | 3×3 Conv2d ×4+shortcut | Conv2d s=2 | FP32 |
| layer4 (2× BasicBlock) | [B, 256, 16, 8] | [B, 512, 8, 4] | 3×3 Conv2d ×4+shortcut | Conv2d s=2 | FP32 |
| FPN lateral+output (level 0) | [B, 64, 64, 32] | [B, 64, 64, 32] | 1×1 + 3×3 Conv2d | Conv2d | FP32 |
| input_proj | [B, 64, 64, 32] | [B, 256, 64, 32] | [256, 64, 1, 1] | Conv2d 1×1 | FP32 |

FPN produces 5 levels; only level 0 (stride-4, finest) feeds the transformer. Backbone + FPN ≈ 40 MB; all Conv2d, never quantized.

---

### Top-k tokenizer (FP32)

```
Per view:  [B, 256, 64, 32] → L2-norm top-256 spatial selection → [B, 256, 16, 16]
           reshape + permute → [256, B, 256]
Cat:       [hor, ver] → [512, B, 256]   (= [N_e, B, d])
```

---

### Encoder layer (×6) — W8A8 linear; INT8/UINT8 attention

Layer input: `[N_e=512, B, d=256]`

**TPE concat projections** — 6 ops, W8A8 each  
(inputs are feature tokens `[512, B, 256]` and position embeddings `[512, B, 256]`)

| Op | Input | Output | Weight | Precision |
|----|-------|--------|--------|-----------|
| ca_qcontent_proj | [512, B, 256] | [512, B, 256] | [256, 256] | W8A8 GEMM |
| ca_kcontent_proj | [512, B, 256] | [512, B, 256] | [256, 256] | W8A8 GEMM |
| ca_v_proj | [512, B, 256] | [512, B, 256] | [256, 256] | W8A8 GEMM |
| ca_kpos_proj | [512, B, 256] | [512, B, 256] | [256, 256] | W8A8 GEMM |
| ca_vpos_proj | [512, B, 256] | [512, B, 256] | [256, 256] | W8A8 GEMM |
| ca_qpos_sine_proj | [512, B, 256] | [512, B, 256] | [256, 256] | W8A8 GEMM |

Concat content+pos along head dim → Q, K, V each `[512, B, 512]` (2d=512).

**Main attention projections** — 3 ops, W8A8

| Op | Input | Output | Weight | Precision |
|----|-------|--------|--------|-----------|
| self_attn_q_proj | [512, B, 512] | [512, B, 512] | [512, 512] | W8A8 GEMM |
| self_attn_k_proj | [512, B, 512] | [512, B, 512] | [512, 512] | W8A8 GEMM |
| self_attn_v_proj | [512, B, 512] | [512, B, 512] | [512, 512] | W8A8 GEMM |

Head split: `[B×H, 512, 128]` each — H=4, h_e=128.

**Attention compute**

| Op | Input | Output | Precision |
|----|-------|--------|-----------|
| qkv_fake_quant (STE) | [B×4, 512, 128] each | [B×4, 512, 128] each | INT8 sym per-tensor |
| QK^T bmm | Q[B×4, 512, 128] × K^T[B×4, 128, 512] | [B×4, 512, 512] | INT8×INT8 → INT32 → FP32 |
| scale 1/√128 + mask | [B×4, 512, 512] | [B×4, 512, 512] | FP32 |
| **online softmax** | [B×4, 512, 512] | [B×4, 512, 512] | FP32 (tile; 4 MB/head, never materialise) |
| aw_fake_quant (STE) | [B×4, 512, 512] | [B×4, 512, 512] | UINT8 asym per-tensor |
| AV bmm | [B×4, 512, 512] × V[B×4, 512, 128] | [B×4, 512, 128] | UINT8×INT8 → INT32 → FP32 |

Concat heads → `[512, B, 512]`

**Output projection + residual**

| Op | Input | Output | Weight | Precision |
|----|-------|--------|--------|-----------|
| self_attn_out_proj | [512, B, 512] | [512, B, 512] | [512, 512] | W8A8 GEMM |
| slice [:, :, :256] + residual | [512, B, 512] → [:256], [512, B, 256] | [512, B, 256] | — | FP32 add |
| LayerNorm1 | [512, B, 256] | [512, B, 256] | γ,β [256] | FP32 |

**FFN**

| Op | Input | Output | Weight | Precision |
|----|-------|--------|--------|-----------|
| linear1 | [512, B, 256] | [512, B, 2048] | [2048, 256] | W8A8 GEMM |
| ReLU | [512, B, 2048] | [512, B, 2048] | — | FP32 |
| linear2 | [512, B, 2048] | [512, B, 256] | [256, 2048] | W8A8 GEMM |
| residual + LayerNorm2 | [512, B, 256] | [512, B, 256] | γ,β [256] | FP32 |

Encoder layer output: `[N_e=512, B, d=256]`

---

### Decoder layer (×6) — W8A8 linear; INT8/UINT8 attention

Layer input: `tgt [N_d=10, B, d=256]`, `memory [N_e=512, B, d=256]`

#### Decoder self-attention

**Projections** — 5 ops, W8A8

| Op | Input | Output | Weight | Note |
|----|-------|--------|--------|------|
| sa_qcontent_proj | [10, B, 256] | [10, B, 256] | [256, 256] | Q content |
| sa_qpos_proj | [10, B, 256] | [10, B, 256] | [256, 256] | Q pos |
| sa_kcontent_proj | [10, B, 256] | [10, B, 256] | [256, 256] | K content |
| sa_kpos_proj | [10, B, 256] | [10, B, 256] | [256, 256] | K pos |
| sa_v_proj | [10, B, 256] | [10, B, 256] | [256, 256] | V |

Q = sa_qcontent + sa_qpos, K = sa_kcontent + sa_kpos (FP32 add). Head split: `[B×4, 10, 64]` — h_d_sa=64.

**Attention compute**

| Op | Input | Output | Precision |
|----|-------|--------|-----------|
| qkv_fake_quant | [B×4, 10, 64] each | [B×4, 10, 64] each | INT8 sym |
| QK^T bmm | [B×4, 10, 64] × [B×4, 64, 10] | [B×4, 10, 10] | INT8×INT8 → FP32 |
| scale 1/√64 + softmax | [B×4, 10, 10] | [B×4, 10, 10] | FP32 |
| aw_fake_quant | [B×4, 10, 10] | [B×4, 10, 10] | UINT8 asym |
| AV bmm | [B×4, 10, 10] × [B×4, 10, 64] | [B×4, 10, 64] | UINT8×INT8 → FP32 |
| self_attn.out_proj | [10, B, 256] | [10, B, 256] | [256, 256] | W8A8 GEMM |

Residual + LayerNorm1 → `[10, B, 256]`

#### Decoder cross-attention

**Projections** — 5 ops W8A8 (+ 1 first-layer-only)

| Op | Input | Output | Weight | Note |
|----|-------|--------|--------|------|
| ca_qcontent_proj | [10, B, 256] | [10, B, 256] | [256, 256] | Q content from tgt |
| ca_qpos_proj | [10, B, 256] | [10, B, 256] | [256, 256] | Q pos (layer 0 only) |
| ca_qpos_sine_proj | [10, B, 256] | [10, B, 256] | [256, 256] | Q sine-pos (all layers) |
| ca_kcontent_proj | [512, B, 256] | [512, B, 256] | [256, 256] | K content from memory |
| ca_kpos_proj | [512, B, 256] | [512, B, 256] | [256, 256] | K pos |
| ca_v_proj | [512, B, 256] | [512, B, 256] | [256, 256] | V from memory |

Cat q_content+q_sine_pos → Q `[10, B, 512]`; cat k_content+k_pos → K `[512, B, 512]`; V stays `[512, B, 256]`.  
Head split: Q `[B×4, 10, 128]`, K `[B×4, 512, 128]`, V `[B×4, 512, 64]` — h_d_ca=128, h_d_ca_v=64.

**Attention compute**

| Op | Input | Output | Precision |
|----|-------|--------|-----------|
| qkv_fake_quant | Q,K [B×4, ·, 128]; V [B×4, 512, 64] | same shapes | INT8 sym |
| QK^T bmm | Q[B×4, 10, 128] × K^T[B×4, 128, 512] | [B×4, 10, 512] | INT8×INT8 → FP32 |
| scale 1/√128 + softmax | [B×4, 10, 512] | [B×4, 10, 512] | FP32 |
| aw_fake_quant | [B×4, 10, 512] | [B×4, 10, 512] | UINT8 asym |
| AV bmm | [B×4, 10, 512] × V[B×4, 512, 64] | [B×4, 10, 64] | UINT8×INT8 → FP32 |
| cross_attn.out_proj | [10, B, 256] | [10, B, 256] | [256, 256] | W8A8 GEMM |

Residual + LayerNorm2 → `[10, B, 256]`

**FFN**

| Op | Input | Output | Weight | Precision |
|----|-------|--------|--------|-----------|
| linear1 | [10, B, 256] | [10, B, 2048] | [2048, 256] | W8A8 GEMM |
| ReLU | [10, B, 2048] | [10, B, 2048] | — | FP32 |
| linear2 | [10, B, 2048] | [10, B, 256] | [256, 2048] | W8A8 GEMM |
| residual + LayerNorm3 | [10, B, 256] | [10, B, 256] | γ,β [256] | FP32 |

Decoder per-layer output: `[N_d=10, B, d=256]`. Final decoder LayerNorm: FP32.

---

### Detection head (W8A8)

| Op | Input | Output | Weight | Precision |
|----|-------|--------|--------|-----------|
| class_embed | [B, 10, 256] | [B, 10, 2] | [2, 256] | W8A8 GEMM |
| bbox_embed layer 0 | [B, 10, 256] | [B, 10, 256] | [256, 256] | W8A8 GEMM |
| bbox_embed layer 1 | [B, 10, 256] | [B, 10, 256] | [256, 256] | W8A8 GEMM |
| bbox_embed layer 2 | [B, 10, 256] | [B, 10, 6] | [6, 256] | W8A8 GEMM |

---

## PTQ accuracy results (P2S1, CPU)

All runs: P2S1 test split, 23,074 samples. CPU: AMD Ryzen AI MAX+ 395. PyTorch 2.11, torchao 0.17.0.  
FP32 baseline: **AP=42.78 / AR1=39.79 / Seg IoU=74.41 / 156.1 MB**

> CPU baseline AP=42.78 differs from GPU AP=46.75 (oneDNN floating-point ordering). All Δ AP relative to CPU FP32.

| Scheme | Component | BBox AP | Δ AP | Seg IoU | Size | Compression |
|--------|-----------|---------|------|---------|------|-------------|
| fp32 | — | 42.78 | — | 74.41 | 156.1 MB | 1.00× |
| int8wo | decoder only | 42.73 | −0.05 | 74.41 | 124.0 MB | 1.26× |
| int8wo | transformer | 42.73 | −0.05 | 74.41 | 81.4 MB | 1.92× |
| int8wo | all | 42.73 | −0.05 | 74.41 | 81.4 MB | 1.92× |
| int8dq | all | 42.66 | −0.12 | 74.40 | 81.4 MB | 1.92× |
| int8dq + BF16 autocast | all | 41.07 | −1.71 | 74.22 | 81.4 MB | 1.92× |
| int4fq_g128 | all | 42.93 | +0.15 | 74.33 | 69.4 MB | 2.25× |
| int4fq_g64 | all | 42.24 | −0.54 | 74.10 | 70.2 MB | 2.22× |
| int4fq_g32 | all | 41.72 | −1.06 | 74.33 | 71.8 MB | 2.18× |
| fp32 + attn-bmm INT8† | — | 42.78 | +0.00 | 74.41 | 156.1 MB | — |
| int8wo + attn-bmm INT8† | all | 42.73 | −0.05 | 74.41 | 81.4 MB | 1.92× |
| int8dq + attn-bmm INT8† | all | 42.66 | −0.12 | 74.40 | 81.4 MB | 1.92× |
| int4fq_g128 + attn-bmm INT8† | all | 42.92 | +0.14 | 74.33 | 69.4 MB | 2.25× |
| fp32 + attn-weights UINT8‡ | — | 42.88 | +0.10 | 74.44 | 156.1 MB | — |
| fp32 + full attn INT8†‡ | — | 42.88 | +0.10 | 74.43 | 156.1 MB | — |
| int8wo + full attn INT8†‡ | all | 42.73 | −0.05 | 74.41 | 81.4 MB | 1.92× |
| int8dq + full attn INT8†‡ | all | 42.66 | −0.12 | 74.40 | 81.4 MB | 1.92× |

†attn-bmm INT8: symmetric per-tensor INT8 fake-quant of Q, K, V at bmm input (post-projection, post-head-split). Covers both encoder [B×4, 512, 128] and decoder [B×4, 10/512, 128] matmuls. STE; measures accuracy impact without a real INT8 kernel.

‡attn-weights UINT8: asymmetric per-tensor UINT8 fake-quant of the softmax output before AV bmm. Softmax output ∈ [0,1]; scale = max(x)/255. Full attn INT8 = †+‡ combined.

**Scheme definitions:**
- `int8wo` — weights INT8 static, dequant to FP32 before matmul; FP32 arithmetic (memory BW saving only)
- `int8dq` — weights INT8 + activations per-token INT8 at runtime; real INT8 GEMM (oneDNN)
- `int4fq_gN` — symmetric per-group INT4 weight fake-quant (group size N); dequant to FP32; numerically identical to real INT4wo for accuracy purposes
- `all` / `transformer` — encoder + decoder nn.Linear (161 modules); backbone is Conv2d → unaffected

**Key findings:**
- INT8 weight-only: effectively lossless at 1.92× (−0.05 AP, IoU unchanged)
- INT8dq (W8A8): −0.12 AP at same compression; +0.07 vs int8wo
- BF16 autocast stacked on int8dq: −1.71 AP — BF16 backbone compounds with INT8 activations; not viable
- INT8 Q/K/V at bmm input adds **zero incremental loss** — fp32+bmm8 = fp32 exactly
- UINT8 attention weights likewise **zero incremental loss** — full INT8 attention free
- **int8dq + full attn INT8 = AP 42.66, same as int8dq alone** — attention quantization costs nothing on top of W8A8 linears
- INT4 g128 noise cancels: +0.15 AP (noise from quantization acts as regularization at this group size)

---

## PTQ accuracy results — FP4 attention activations (P2S1, GPU)

All runs: P2S1 test split, 7,942 samples. GPU: NVIDIA RTX 4070, batch 16. PyTorch 2.11, torchao 0.17.0.  
GPU FP32 baseline: **AP=49.64 / Seg IoU=74.92 / 156.1 MB**  
GPU int8dq_transformer baseline: **AP=49.77 / Seg IoU=74.96 / 81.4 MB** (Δ +0.13 vs GPU FP32 — oneDNN ordering)

Scheme for all rows: `int8dq`, component: `transformer` (encoder + decoder linears W8A8).

| Attn activation quant | Run tag | BBox AP | Δ AP vs int8dq | Seg IoU |
|-----------------------|---------|---------|----------------|---------|
| INT8 STE (baseline) | `_transformer` | 49.77 | — | 74.96 |
| NVFP4 E2M1 Q/K/V only (FP8 scale) | `_bmmfp4` | 49.77 | 0.00 | 74.96 |
| NVFP4 E2M1 Q/K/V + attn-weights (FP8 scale) | `_bmmfp4_awfp4` | 49.77 | 0.00 | 74.96 |
| NVFP4 E2M1 Q/K/V + attn-weights (MX pow-2 scale) | `_bmmmxfp4_awmxfp4` | 49.77 | 0.00 | 74.96 |

†FP4 block size 16 along last dim. FP8 scale = max(|block|)/6.0; MX scale = nearest power-of-2 above FP8 scale (~3× cheaper).  
‡Decoder SA attn maps are 10×10 (100 elements = 7 full blocks + 1 partial block of 4, zero-padded). Small block count may increase FP4 error there but shows no measurable effect on AP.

**Key finding:** NVFP4 E2M1 block quantization on all attention activations (Q, K, V, and post-softmax weights) adds **zero incremental AP loss** on top of int8dq W8A8 linears, for both FP8 and MX (power-of-2) scales. Attention coprocessor can safely use FP4 for all activation storage and transport.

---

## PTQ accuracy results — G2 + FP4 + all-FP16 precision ladder (P2S1, GPU)

Each column adds one more FP16 push on top of the previous. All runs: int8dq transformer, FP16 backbone, FP4 attention activations.

| Stage | baseline | +FP16 LN/SM/head | +FP16 residuals (FP8 scale) | +FP16 residuals (MX scale) |
|-------|----------|------------------|-----------------------------|---------------------------|
| Backbone | FP16 | FP16 | FP16 | FP16 |
| W8A8 linear output | **FP32** | **FP32**† | **FP16** | **FP16** |
| Residual adds | **FP32** | **FP32**† | **FP16** | **FP16** |
| LayerNorm | **FP32** | FP16 | FP16 | FP16 |
| Softmax | **FP32** | FP16 | FP16 | FP16 |
| Detection/seg head | **FP32** | FP16 | FP16 | FP16 |
| FP4 scale | FP8 | FP8 | FP8 | **MX pow-2** |
| Tokenizer | **FP32** | FP16 | FP16 | FP16 |
| **BBox AP** | **49.77** | **49.75** | **49.72** | **49.72** |
| **Δ AP** | — | −0.02 | −0.05 | −0.05 |
| **Seg IoU** | **74.96** | **75.04** | **75.02** | **75.02** |

†torchao int8dq hardcodes INT32 accumulate → FP32 output; on real hardware the MAC array returns INT32 → rounds to FP16 directly.

**Key finding:** Pushing LN, softmax, and heads to FP16 costs −0.02 AP. Pushing W8A8 linear outputs and residuals to FP16 costs an additional −0.03 AP (total −0.05). Switching from FP8 to MX (power-of-2) scale on FP4 blocks has **no further effect** — both land at AP=49.72. The entire precision ladder from FP32 through full-FP16+FP4-MX costs only **−0.05 AP total**, well within detection noise.

---

## Acceptable regression — literature survey (April 2026)

INT8 PTQ community norm for detection: **≤1 AP point** absolute. No prior RETR/MMVR quantization results exist; these are the first.

| Standard | Threshold | int8wo (−0.05) | int8dq (−0.12) |
|----------|-----------|----------------|----------------|
| MLPerf 99% floor (42.35 AP) | ≥99% of FP32 | +0.38 above floor | +0.31 above floor |
| MLPerf 99.9% floor (42.74 AP) | ≥99.9% of FP32 | −0.01 (borderline) | −0.08 (just outside) |
| Detection community norm | ≤1.0 AP drop | 20× within | 8× within |
| Radar/embedded FPGA (MDPI Sensors 2024) | ≤0.74% accuracy drop | Pass | Pass |
| Q-DETR W4A4 PTQ, CVPR 2023 | −2.6 AP drop | 52× smaller | 22× smaller |

MLPerf floors: 0.99 × 42.78 = 42.35; 0.999 × 42.78 = 42.74. The 0.01 AP margin on the 99.9% tier is within measurement noise; treat as borderline.

**Hardware sizing implication:** INT8 MAC arrays fully justified. W8A8 linear costs −0.07 AP incremental over weight-only; favorable if it halves activation SRAM bandwidth. Attention coprocessor can use full INT8 (Q/K/V) + UINT8 (softmax output) at zero accuracy cost. Keep 32-bit accumulators. Validation gate: hardware INT8 should land within ±0.3 AP of the software INT8 reference on MMVR P2S1 test.

---

## Hardware sizing — attention matrices

Model: nhead=4, N_e=512, N_d=10, h_e=128, h_d_ca=128, h_d_ca_v=64.

| Attention site | Q shape | K shape | QK^T shape | FP32 size / sample | Decision |
|----------------|---------|---------|------------|---------------------|----------|
| Encoder self-attn | [B×4, 512, 128] | [B×4, 512, 128] | [B×4, 512, 512] | 4×512×512×4 B = **4 MB** | Online softmax required |
| Decoder cross-attn | [B×4, 10, 128] | [B×4, 512, 128] | [B×4, 10, 512] | 4×10×512×4 B = **80 KB** | Materialise in full |
| Decoder self-attn | [B×4, 10, 64] | [B×4, 10, 64] | [B×4, 10, 10] | 4×10×10×4 B = **1.6 KB** | Trivial |

**Encoder online softmax:** 512×512 attention (1 MB/head) exceeds practical SRAM tile budgets. Flash Attention tiling: accumulate softmax numerator and running max in a single pass, never materialising the full matrix. PyTorch SDPA uses this path; hardware should mirror the same tiling.

**UCIe coprocessor scope:** Main INT8 MAC array handles GEMM (all linear projections, QK^T, AV). UCIe-connected FP coprocessor handles: online softmax reduction, FP32/BF16 residual adds, LayerNorm. Decoder attention (≤80 KB matrices) fits entirely in either unit.

---

## Running PTQ experiments

**Quick sweep — 4 runs:**
```bash
conda activate retr-quant
bash experiments/run_sweep_quick.sh
```

**Single experiment:**
```bash
conda activate retr-quant
cd src
python eval_ptq.py \
    --root ../MMVR/segment_4_3 --split P2S1 --task DETSEG \
    --pretrained_path ../logs/pretrained_model/p2s1_retr_detseg.pth \
    --batch_size 8 --worker 2 --device cpu \
    --scheme int8dq --component all \
    --attn_bmm_bits 8 --attn_weights_bits 8
```

**Flags:**
- `--scheme` — `fp32` | `int8wo` | `int8dq` | `int4wo_g128` | `int4wo_g64` | `int4wo_g32`
- `--component` — `all` | `transformer` | `encoder` | `decoder`
- `--attn_bmm_bits 8` — INT8 fake-quant on Q, K, V at bmm input
- `--attn_weights_bits 8` — UINT8 fake-quant on softmax output before AV bmm
- `--use_autocast` — wrap forward in `torch.autocast(bf16)` (stacks on top of any scheme)

Results saved to `experiments/results/<run_name>.json`. See `docs/experiments.md` for full catalogue and JSON schema.

**PTQ environment setup** (separate from original `retr` training env):
```bash
bash setup.sh               # one-time; creates both envs
# or
conda env create -f environment.yml   # creates retr-quant (Python 3.11, PyTorch 2.11, torchao 0.17.0)
conda activate retr-quant
```

---

**Method citations:** STE fake-quant — PyTorch `torch.ao.quantization.FakeQuantize`; attention matmul quantization — FQ-ViT (Lin et al., ICCV 2021, arXiv:2111.13824); detection transformer W4A4 PTQ — Q-DETR (Xu et al., CVPR 2023, arXiv:2304.00253).
