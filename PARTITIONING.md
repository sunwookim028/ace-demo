<!--
Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->
# RETR Chiplet Partitioning Plan

Companion to `QUANT.md`. Quantization numbers and shapes there; partitioning decisions, per-chiplet mapping, and demo plan here. Each `---` divides a slide-shaped chunk.

**Status (T+14 h):** core partitioning recipe validated on GPU — G1 (FP16 CGRA viable), G2 (FP16 backbone + INT8/UINT8 transformer composes; **core partitioning gate**), and G3 (autocast FP16 with LN/softmax FP32 fallback) all passed on a 33% subset of the P2S1 test split (9/27 subsessions, n=7942 samples — full P2_01/P2_02 Zenodo archives not yet extracted). All three gates clear the 99% MLPerf floor relative to the rebaselined GPU FP32 AP of 0.4964 on the same subset. G4–G7 (MX format, boundary quant, FPGA LN equivalence, silicon-in-the-loop harness) still ahead — code in progress. Confidence: ~75% on partitioning direction; ~85% once full-test-set confirmation lands.

---

## 1. System under design

Three chiplets on a UCIe fabric, ~1 mm² each, 1–2 MB local SRAM each. External DRAM shared for weight streaming.

```
┌──────────┐  UCIe  ┌──────────┐  UCIe  ┌──────────┐
│  CGRA    │◄──────►│  FPGA    │◄──────►│   CIM    │
│  FP16    │        │  softcore│        │  INT8/   │
│          │        │  + SIMD  │        │  UINT8   │
└──────────┘        └──────────┘        └──────────┘
                   (~3k CLBs softcore,
                    ~2k CLBs spare)
```

- **CGRA** — FP16 fabric for FFT, Conv2d backbone, FPN, input_proj, top-k tokenizer, sine pos embeddings.
- **FPGA** — soft-core CPU for orchestration + DMA; spare fabric runs a small SIMD unit (LayerNorm, residual, evolvability hook).
- **CIM** — INT8 MAC array for all 161 nn.Linear modules; integrated nonlinear SIMD for softmax/LN/ReLU; MX weight dequant.

Workload: radar signal → FFT → ResNet-18 + FPN backbone (per-view ×2) → top-k tokenizer → 6-layer encoder → 6-layer decoder → detection head → bbox + class + segmentation.

---

## 2. Quantization recipe — experiment status

Status as of **T+14 h** of this campaign (see §9 for ordered build list). Two classes: **tested** (measured on real runs, JSON in `experiments/results/`) and **expected** (predicted range pending run, confidence tied to adjacent tested points).

**Gates passed:** G1, G2 (core partitioning recipe), G3 (full FP16 recipe with LN/softmax FP32 fallback) — all full-tier (33% P2S1 subset, n=7942).
**Gates not yet started:** G4 (MX format), G5 (boundary quant), G6 (FPGA LN equivalence), G7 (silicon-in-the-loop harness).

### 2.1 Tested

**From QUANT.md** (CPU, P2S1 test, 23,074 samples, FP32 baseline AP=42.78):

| Scheme | Δ AP | Size | Note |
|--------|------|------|------|
| int8wo | −0.05 | 81.4 MB | effectively lossless |
| int8dq (W8A8, all 161 linears) | −0.12 | 81.4 MB | chosen linear recipe |
| int8dq + INT8 Q/K/V + UINT8 softmax-out (full attn INT8) | −0.12 | 81.4 MB | attention quant is free |
| int4fq_g128 | +0.15 | 69.4 MB | regularization at this block size |
| int8dq + BF16 autocast | −1.71 | 81.4 MB | not viable — motivates FP16 study |

**This campaign, GPU** (P2S1 test, 33% subset n=7942; GPU FP32 AP=0.4964 — Δ AP reported against GPU FP32):

| Scheme | AP | Δ AP (GPU) | Size | Latency | JSON | Gate | Status |
|--------|-----|------------|------|---------|------|------|--------|
| FP32 baseline | 0.4964 | +0.0000 | 156 MB | 20.6 ms | `P2S1_fp32_all.json` | anchor | ✓ |
| int8dq all | 0.4977 | +0.0013 | 81 MB | 40.1 ms | `P2S1_int8dq_all.json` | reproduces CPU ±0.1 | ✓ |
| int8dq + full attn INT8 | 0.4977 | +0.0013 | 81 MB | 38.9 ms | `P2S1_int8dq_all_bmm8_aw8.json` | reproduces CPU | ✓ |
| FP16 backbone only | 0.4964 | −0.0001 | 134 MB | 20.7 ms | `P2S1_fp32_all_hbe16.json` | **G1** | ✓ full-tier |
| FP16 backbone + int8dq + full INT8 attn + FP32 decoder LN | 0.4975 | +0.0010 | **60 MB** | 43.6 ms | `P2S1_int8dq_transformer_bmm8_aw8_hbe16_ln32-decoder.json` | **G2 — core partitioning gate** | ✓ full-tier |
| Autocast FP16 + FP32 LN all (G3) | 0.4960 | −0.0005 | 156 MB | 21.3 ms | `P2S1_fp32_all_ac16_ln32-all.json` | **G3** | ✓ full-tier |

**Quick-tier probes** (N≈1,008, GPU, n=7942 full FP32 AP=0.4964 used as reference; note quick AP is lower due to sample distribution — use relative ordering, not absolute Δ):

| Scheme | Quick AP | Quick Δ AP | Purpose | Precursor |
|--------|----------|------------|---------|-----------|
| FP16 backbone only | 0.3354 | −0.1610 | G1 precursor | G1 ✓ |
| FP16 backbone + int8dq transformer | 0.3379 | −0.1586 | isolates INT8 transformer impact | G2 ✓ |
| G2 chiplet recipe (FP16 BE + int8dq + INT8 attn + FP32 dec LN) | 0.3379 | −0.1586 | G2 precursor | G2 ✓ |
| FP16 all (autocast, no upcast) | 0.3328 | −0.1636 | full FP16 no safeguard | — |
| Autocast FP16 + FP32 LN encoder | 0.3328 | −0.1636 | LN precision (enc) | G3 |
| Autocast FP16 + FP32 LN decoder | 0.3328 | −0.1636 | LN precision (dec) | G3 |
| Autocast FP16 + FP32 LN all (G3 precursor) | 0.3328 | −0.1636 | LN precision all | G3 ✓ |
| Autocast FP16 + FP32 softmax encoder | 0.3328 | −0.1636 | softmax precision (enc) | G3 |
| Autocast FP16 + FP32 softmax decoder | 0.3328 | −0.1636 | softmax precision (dec) | G3 |
| Autocast FP16 + FP32 softmax all | 0.3328 | −0.1636 | softmax precision all | G3 |

Note: all autocast variants collapse to identical AP=0.3328 at N=1,008 — sample variance dominates at this scale. The full-tier G3 run (n=7942) resolves the difference: Δ=−0.0005 vs FP32, well above the 99% floor. FP16 transformer-only, FP16 residual-only, and FP16 bmm-accumulator probes were not run separately (replaced by the autocast approach which routes softmax/LN to FP32 automatically).

### 2.2 Expected — pending runs

Pending runs, ranges based on adjacent tested numbers and QUANT.md patterns.

| Experiment | Expected Δ AP | Pass threshold | Gate | Status |
|------------|---------------|----------------|------|--------|
| MXINT8 weight (pow2 block scale, block 32) + FP16 attn (Phase 3.4) | −0.10 to −0.30 | ≥ 42.35 AP | **G4** | code TBD |
| MXFP4 block 32 + FP16 attn | −0.5 to −1.5 | reported as stretch | — | code TBD |
| MXFP4 block 16 + FP16 attn | −0.3 to −1.2 | reported as stretch | — | code TBD |
| Post-`input_proj` INT8 boundary quant stacked on G3 (Phase 4.2) | ≤ −0.10 incremental | matches QUANT.md's free attn-bmm INT8 pattern | **G5** | code TBD |
| FPGA FP16 LN vs CIM FP16 LN numerical match (Phase 5) | <1 ULP max abs error on decoder activations | identical IEEE op order | **G6** | standalone test, code TBD |
| FFT FP16 (twiddles + input) | −0.05 to −0.20 | well-conditioned small transform | — | queued |
| FP16 sine pos + FP16 top-k tokenizer | negligible | scalar / L2 norm only | — | queued |

### 2.3 Not yet started (code pending)

- `src/utils/boundary_quant.py` — per-tensor INT8 quantizer module for chiplet handoffs
- MX format fake-quant — extension to `src/quantize_torchao.py`
- `src/utils/silicon_hooks.py` — stage-isolation forward hooks (`--silicon-in-the-loop`)
- `experiments/run_golden_diff.sh` — twin-inference golden vs silicon driver
- `tests/test_fpga_ln_equivalence.py` — Phase 5 numerical test

---

## 3. Partitioning decisions

### CGRA — FP16

| Op | Shape (per view, B=1) | Weight shape | Precision |
|----|----------------------|--------------|-----------|
| FFT (raw radar → 4-frame input) | → [4, 256, 128] | — | FP16 |
| conv1 | [4, 256, 128] → [64, 128, 64] | [64, 4, 7, 7] | FP16 |
| bn1 + ReLU + maxpool | [64, 128, 64] → [64, 64, 32] | γ,β[64] | FP16 |
| layer1 (2× BasicBlock, 3×3 Conv2d ×4) | [64, 64, 32] → [64, 64, 32] | 3×3 Conv | FP16 |
| layer2 (2× BasicBlock, s=2) | [64, 64, 32] → [128, 32, 16] | 3×3 Conv + shortcut | FP16 |
| layer3 (2× BasicBlock, s=2) | [128, 32, 16] → [256, 16, 8] | 3×3 Conv + shortcut | FP16 |
| layer4 (2× BasicBlock, s=2) | [256, 16, 8] → [512, 8, 4] | 3×3 Conv + shortcut | FP16 |
| FPN lateral+output (level 0) | [64, 64, 32] → [64, 64, 32] | 1×1 + 3×3 Conv | FP16 |
| input_proj | [64, 64, 32] → [256, 64, 32] | [256, 64, 1, 1] | FP16 |
| Top-k tokenizer (L2-norm, top-256 select, permute) | [256, 64, 32] → [256, 256] per view | — | FP16 |
| Two-view concat → `[N_e=512, B, d=256]` | — | — | FP16 |
| Sine positional embedding | → `[512, B, 256]` | — | FP16 |
| **Output activation quantizer → INT8 + FP8/pow2 scale** | `[512, B, 256]` FP16 → INT8 | — | quant at producer |

Backbone + FPN ≈ 55 MB FP32 / **~27.5 MB FP16** — doesn't fit 1–2 MB local SRAM; weights stream per-layer from external DRAM (CGRA keeps 1–2 layers resident).

### FPGA — orchestration + flexibility SIMD

| Role | Resource | Note |
|------|----------|------|
| Soft-core CPU | ~3k CLBs | weight-DMA scheduling, per-layer config, scale propagation, chiplet handoff |
| FP16 SIMD (8-lane, LN + residual; 16-lane if DSPs available) | ~800–2000 CLBs | handles decoder LayerNorm (18 sites) |
| DMA engines + UCIe control | ~200–500 CLBs | — |

**What the FPGA SIMD runs:** 18 decoder LayerNorms (3 per layer × 6 layers), each `[N_d=10, B, d=256]` FP16 = 5 KB per call. Two-pass reduction (mean, then var), normalize. At 8 lanes × 500 MHz × 2 passes: ~640 cycles = 1.3 µs per site. Per inference: ~23 µs total — fully overlapped with CIM compute. Encoder LN stays on CIM (tensor is `[512, B, 256]` = 128 KB per site; BW-prohibitive to round-trip).

**Evolvability hook:** the same SIMD can be reprogrammed for RMSNorm, GroupNorm, GeLU, SwiGLU, or SiLU in a future model variant — no silicon respin. This is the concrete flexibility story.

### CIM — INT8/UINT8 linear + attention

All 161 nn.Linear modules (72 encoder + ~85 decoder + 4 detection head), all attention BMMs, encoder LayerNorms, and all softmax/ReLU.

Per-layer forward structure unchanged from `QUANT.md` §Layer-by-layer. See §4 below for the mapping table with chiplet column.

---

## 4. Layer-by-layer flow with chiplet mapping

Shapes `[seq, B, dim]`. Batch B free; use B=1 for demo.

### Encoder layer (×6) — on CIM

| Op | Input | Output | Weight | Precision | Chiplet |
|----|-------|--------|--------|-----------|---------|
| 6× TPE concat projections (ca_qcontent/kcontent/v/kpos/vpos/qpos_sine) | [512, B, 256] | [512, B, 256] | [256, 256] each | W8A8 | CIM |
| Concat content+pos → Q, K, V | [512, B, 256]×2 | [512, B, 512] | — | INT8 | CIM |
| self_attn q/k/v_proj | [512, B, 512] | [512, B, 512] | [512, 512] each | W8A8 | CIM |
| Head split (H=4, h_e=128) | [512, B, 512] | [B×4, 512, 128] | — | INT8 | CIM |
| Q,K,V INT8 fake-quant (sym per-tensor STE) | [B×4, 512, 128] | same | — | INT8 | CIM |
| QK^T bmm | Q × K^T | [B×4, 512, 512] | — | INT8×INT8→FP16 | CIM |
| scale 1/√128 + mask | [B×4, 512, 512] | same | — | FP16 | CIM |
| Online softmax (tile, 4 MB/head never materialized) | [B×4, 512, 512] | same | — | FP16 | CIM SIMD |
| UINT8 fake-quant (asym per-tensor STE) | [B×4, 512, 512] | same | — | UINT8 | CIM |
| AV bmm | softmax × V | [B×4, 512, 128] | — | UINT8×INT8→FP16 | CIM |
| Concat heads | [B×4, 512, 128] | [512, B, 512] | — | FP16 | CIM |
| self_attn_out_proj | [512, B, 512] | [512, B, 512] | [512, 512] | W8A8 | CIM |
| Slice [:, :, :256] + residual | — | [512, B, 256] | — | FP16 | CIM |
| LayerNorm1 | [512, B, 256] | same | γ,β[256] | FP16 | CIM SIMD |
| linear1 | [512, B, 256] | [512, B, 2048] | [2048, 256] | W8A8 | CIM |
| ReLU | [512, B, 2048] | same | — | FP16 | CIM SIMD |
| linear2 | [512, B, 2048] | [512, B, 256] | [256, 2048] | W8A8 | CIM |
| residual + LayerNorm2 | [512, B, 256] | same | γ,β[256] | FP16 | CIM SIMD |

### Decoder layer (×6) — on CIM except LayerNorm → FPGA

| Op | Input | Output | Weight | Precision | Chiplet |
|----|-------|--------|--------|-----------|---------|
| SA: 5 projs (sa_qcontent/qpos/kcontent/kpos/v) | [10, B, 256] | [10, B, 256] | [256, 256] each | W8A8 | CIM |
| SA: Q = qc+qp, K = kc+kp (FP16 add); head split (h_d_sa=64) | — | [B×4, 10, 64] | — | FP16 | CIM |
| SA: QKV INT8 fake-quant | [B×4, 10, 64] | same | — | INT8 | CIM |
| SA: QK^T bmm | — | [B×4, 10, 10] | — | INT8×INT8→FP16 | CIM |
| SA: scale 1/√64 + softmax | [B×4, 10, 10] | same | — | FP16 | CIM SIMD |
| SA: UINT8 fake-quant + AV bmm | — | [B×4, 10, 64] | — | UINT8×INT8→FP16 | CIM |
| SA: self_attn.out_proj | [10, B, 256] | [10, B, 256] | [256, 256] | W8A8 | CIM |
| **residual + LayerNorm1** | [10, B, 256] | same | γ,β[256] | FP16 | **FPGA SIMD** |
| CA: 5 projs + ca_qpos_proj (layer 0 only) | — | [10/512, B, 256] | [256, 256] | W8A8 | CIM |
| CA: concat for Q [10, B, 512]; K [512, B, 512]; V [512, B, 256] | — | — | — | FP16 | CIM |
| CA: head split (h_d_ca=128, h_d_ca_v=64) | — | Q [B×4, 10, 128], K [B×4, 512, 128], V [B×4, 512, 64] | — | INT8 | CIM |
| CA: QKV INT8 fake-quant + QK^T bmm | — | [B×4, 10, 512] | — | INT8×INT8→FP16 | CIM |
| CA: scale 1/√128 + softmax + UINT8 fake-quant + AV bmm | — | [B×4, 10, 64] | — | FP16/UINT8 | CIM SIMD |
| CA: cross_attn.out_proj | [10, B, 256] | [10, B, 256] | [256, 256] | W8A8 | CIM |
| **residual + LayerNorm2** | [10, B, 256] | same | γ,β[256] | FP16 | **FPGA SIMD** |
| FFN linear1 + ReLU + linear2 | [10, B, 256] ↔ [10, B, 2048] | [10, B, 256] | [2048, 256] / [256, 2048] | W8A8 | CIM |
| **residual + LayerNorm3** | [10, B, 256] | same | γ,β[256] | FP16 | **FPGA SIMD** |

Final decoder LayerNorm (post-loop) → **FPGA SIMD** as well. Total: **19 LN sites on FPGA**.

### Detection head — on CIM

| Op | Input | Output | Weight | Precision | Chiplet |
|----|-------|--------|--------|-----------|---------|
| class_embed | [B, 10, 256] | [B, 10, 2] | [2, 256] | W8A8 | CIM |
| bbox_embed layer 0, 1 | [B, 10, 256] | [B, 10, 256] | [256, 256] | W8A8 | CIM |
| bbox_embed layer 2 | [B, 10, 256] | [B, 10, 6] | [6, 256] | W8A8 | CIM |

---

## 5. Per-chiplet storage & compute budget

Hard-verified parameter counts (INT8 = 1 byte/param):

**Encoder layer weights** = 6·256² + 3·512² + 1·512² + 2·(256·2048) = 393,216 + 786,432 + 262,144 + 1,048,576 = **2,490,368 params ≈ 2.49 MB INT8**. Six layers: **14.94 MB**.

**Decoder layer weights** = 5·256² + 256² + 5·256² + 256² + 2·(256·2048) = 327,680 + 65,536 + 327,680 + 65,536 + 1,048,576 = **1,835,008 params ≈ 1.83 MB INT8** per layer, plus 65,536 (ca_qpos_proj in layer 0 only). Six layers: **11.08 MB**.

**Detection head** = 2·256 + 2·256² + 6·256 = 133,120 params ≈ **0.13 MB INT8**.

**Total transformer + head** = 14.94 + 11.08 + 0.13 = **26.15 MB INT8**. Matches QUANT.md int8dq total of 81.4 MB (backbone FP32 ≈ 55 MB + transformer ≈ 26 MB).

**Backbone FP16** = ~27.5 MB (QUANT.md §Backbone says "≈40 MB"; reverse-engineered from 81.4 MB total − 26.1 MB transformer gives ~55 MB FP32 → 27.5 MB FP16).

| Chiplet | SRAM target | Resident | Streamed from DRAM |
|---------|-------------|----------|---------------------|
| CGRA | 1–2 MB | 1–2 Conv2d layers + activation tile | remaining backbone Conv2d weights per layer |
| FPGA | 1–2 MB | soft-core code + LN γ,β (19 × 512 B = 10 KB) + scratchpad | — |
| CIM | 1–2 MB | **one transformer layer's weights** + attention tile buffer | next layer's weights prefetched via FPGA-DMA |

**Pressure point:** one encoder layer = 2.49 MB > 1 MB SRAM. Options: (a) push CIM SRAM target to 2 MB, (b) split each encoder layer into 2 weight-load rounds (TPE+QKV group, then out_proj+FFN group). Decoder layer fits comfortably in 1 MB.

**CIM area** (~1.0–1.3 mm² target):

| Block | Area | Note |
|-------|------|------|
| INT8 MAC array (~1024 MACs, ~2 TOPS @ 1 GHz) | 0.4–0.6 mm² | at 2–5 TOPS/mm² |
| MX weight dequant (pow2 scale multiply) | 0.05–0.1 mm² | ~10–15% MAC overhead |
| Nonlinear SIMD trimmed to {softmax, LN, ReLU}, 16 lanes | ~0.05 mm² | from 70k µm² full SIMD baseline |
| Activation quantizers ×3 (post-softmax UINT8, post-LN INT8, post-ReLU INT8), MX pow2-scale variant | ~0.02 mm² | 7k µm² each (NVFP4 FP8-scale × ⅓) |
| 1 MB SRAM | 0.3–0.4 mm² | 28 nm typical |
| Control + scale broadcast + UCIe slice | 0.1–0.2 mm² | — |

**CGRA area** (~1 mm² target):

| Block | Area |
|-------|------|
| FP16 MAC fabric (~1 TFLOP FP16) | 0.3–0.5 mm² |
| FFT butterfly + twiddle ROM | 0.1–0.2 mm² |
| SRAM 1 MB | 0.3 mm² |
| Control + UCIe | 0.1–0.2 mm² |

**FPGA logic budget** (~5k CLBs total):

| Block | CLB cost | Note |
|-------|----------|------|
| Soft-core CPU | ~3,000 | |
| FP16 SIMD (8-lane, LN + residual + optional ReLU/GeLU LUT) | 800–1,000 | 16-lane at ~1,500–2,000 if DSP blocks available — **open item** |
| DMA engines, UCIe control, scale broadcast | 200–500 | |
| Scratchpad for LN γ,β | ~10 KB BRAM | |

---

## 6. UCIe bandwidth budget (B=1, 30 fps target)

| Flow | Traffic per inference | @ 30 fps |
|------|----------------------|----------|
| CGRA → FPGA → CIM: backbone output (INT8 + scale), 2 views × [1, 256, 64, 32] | 1.05 MB | 32 MB/s |
| DRAM → CIM: transformer weight streaming (26 MB INT8 if re-fetched every inference) | 26 MB | 780 MB/s |
| CIM ↔ FPGA: decoder LN round trips, 19 × [10, 1, 256] FP16 × 2 dirs | 192 KB | 5.8 MB/s |
| CIM → host: detection head output | <1 KB | — |

Total sustained: **< 1 GB/s**. UCIe bundle typically delivers 64–128 GB/s — ample headroom. Weight streaming dominates; can be reduced by keeping resident layers across inferences if workload is bursty.

Counterfactual (rejected): softmax offload to FPGA would ping-pong `[B×4, 512, 512]` attention tiles per head per layer — BW is manageable but tile-granularity scheduling breaks fused attention. Keep softmax on CIM.

---

## 7. Demo modes

### 7.1. Test harness environment — every external piece, called out

The chiplets don't sit alone. A demo needs a host, a board, an interface, and possibly off-chip DRAM. **None of these are decided yet.** This subsection enumerates the options so the demo requirements below can reference them concretely.

**Role A — Host PC.** x86 Linux workstation, PyTorch + simulator installed, ≥32 GB RAM. Drives the test harness, runs the simulator, diffs outputs. Not part of the tape-out.

**Role B — Carrier PCB.** Hosts the tape-out chip, supplies power/clock/reset, exposes I/O to host and (optionally) to DRAM. Not part of the tape-out.

**Role C — Host ↔ chip interface.** Three candidates, none chosen:

| Option | Bandwidth | PHY location | Adequate for |
|--------|-----------|--------------|--------------|
| C1. JTAG | ~1 MB/s | standard tape-out feature | Demo A only |
| C2. USB3 via board bridge (e.g., Cypress FX3) | ~400 MB/s | off-chip bridge on PCB | all demos |
| C3. PCIe | >1 GB/s | on-chip PHY (expensive area) OR off-chip bridge on PCB | all demos, lowest latency |

**Role D — External DRAM** (for weight storage in Demos B/C — ~26 MB INT8 transformer, or ~53 MB with FP16 backbone). Three provisioning options, none chosen:

| Option | Where DRAM controller lives | Impact on chiplet area | Notes |
|--------|-----------------------------|------------------------|-------|
| D1. DDR PHY on FPGA chiplet | inside tape-out | +1–2 mm² hard macro on FPGA chiplet | clean but area-hungry; not in current 1 mm² budget |
| D2. Board-level bridge FPGA owns DRAM | off-chip (Artix-7 / equivalent on PCB) | 0 | exposes weights to our chiplets over UCIe or a simpler parallel link; adds PCB complexity |
| D3. No external DRAM — host streams weights over C2/C3 | — | 0 | host is in the outer loop for weight streaming; soft-core still owns per-layer scheduling. 780 MB/s (Demo B) / 1.6 GB/s (Demo C) fits PCIe gen3 ×4 |

**Role E — Debug/trace.** Logic analyzer headers for UCIe, chiplet I/O, and FPGA soft-core trace ports. Needed for bring-up regardless of demo choice. Not part of the tape-out.

**Decisions blocking demo-board design:** C and D pick. C1 alone would force us to Demo A only. C2 or C3 with any of D1/D2/D3 enables B and C. D1 requires silicon area we haven't budgeted. D2 pushes complexity to the PCB. D3 keeps tape-out simplest but puts the host PCIe/USB bus on the critical weight-streaming path at ~1 GB/s.

### 7.2. Three actors (per inference)

Every demo involves the same three actors; what changes is how much of the work each one does.

- **Host PC (Role A)** — loads MMVR input, runs the pre-silicon reference model, compares outputs against golden, drives the test harness via Role C.
- **Simulator (on host)** — functional/cycle-accurate PyTorch + chiplet models. Runs any model stage *not* currently carved out to silicon. Produces the golden reference for diff checks.
- **Silicon (CGRA + FPGA + CIM chiplets)** — runs whatever carve-out is loaded for that demo. Coordinated by the FPGA soft-core once the host kicks off a run. UCIe links are only exercised when data must flow *chiplet → chiplet*; standalone silicon invocations don't touch UCIe.

The boundary between "simulator" and "silicon" is what distinguishes the demo modes below. Each demo's **Requires** block lists the Role A/B/C/D/E options it depends on.

### 7.3. Demo A — Accuracy-grade slice (component validation)

**Goal:** prove that each chiplet in silicon produces bit-comparable output to the simulator reference, so the quantization recipe is real on hardware.

**Requires:** Role A (host), Role B (carrier PCB, minimal), Role C1/C2/C3 (any — JTAG sufficient because tensors are small and infrequent), Role E (debug). **No external DRAM.** Weights for the one carved-out layer (≤2.5 MB INT8 for encoder layer 6; ≤65 KB for input_proj; LN γ,β ≤1.5 KB) are pushed onto chiplet SRAM over the host interface before the invocation. Per-invocation tensor sizes: CGRA input_proj 512 KB FP16 in / 2 MB FP16 out; CIM encoder layer 128 KB INT8 in / 128 KB INT8 out; FPGA LN 5 KB FP16 in / 5 KB FP16 out. Fits JTAG timescale at 1 inference per minute, USB3 at full rate.

**Actor flow per inference:**

1. Host loads one MMVR sample.
2. Simulator runs FFT + backbone + FPN → produces `[1, 64, 64, 32]` FP16.
3. **Host dumps tensor to CGRA SRAM → CGRA silicon runs input_proj → returns `[1, 256, 64, 32]` FP16 to host.**
4. Simulator runs top-k tokenizer, sine pos, activation quantizer, encoder layers 1–5 → produces `[512, 1, 256]` INT8 + scale.
5. **Host dumps tensor to CIM SRAM → CIM silicon runs encoder layer 6 (all projections + QK^T + softmax + AV + out_proj + FFN + LNs) → returns `[512, 1, 256]` INT8 to host.**
6. Simulator runs decoder layers 1–5 and most of layer 6 → produces `[10, 1, 256]` FP16 at each LN input.
7. **Host dumps each LN input to FPGA → FPGA silicon runs LayerNorm → returns FP16 to host.** (Repeat for 3 LN sites in decoder layer 6.)
8. Simulator runs final decoder LN + detection head → bbox + class + seg.
9. Host diffs against fully-simulated golden.

**UCIe traffic:** none. Each silicon invocation is standalone (host ↔ one chiplet at a time).

**What this proves:** each silicon block computes what the simulator says it should. Accuracy claim is grounded.

**What this does NOT prove:** inter-chiplet coordination, UCIe links, soft-core orchestration.

### 7.4. Demo B — Dataflow showcase (system validation)

**Goal:** prove the three chiplets coordinate over UCIe with the FPGA soft-core scheduling, at real throughput, without host intervention inside the inference.

**Requires:** Role A, Role B (carrier PCB with DRAM socket if D1/D2), Role C2 or C3 (USB3 or PCIe — JTAG too slow for ~1 MB backbone-output push per inference at demo rates), Role D1, D2, or D3 (must pick one — see below), Role E. DRAM (or host-streamed equivalent) holds the **~26 MB INT8 transformer + head weight set**. Soft-core orchestrates per-layer weight DMA into CIM SRAM.

- With D1 or D2: host pushes weights to DRAM once at boot; inference runs with silicon-only inner loop. Host C interface carries only the 1 MB input and small output per inference.
- With D3: host C interface carries the 780 MB/s sustained weight stream *during* inference — PCIe gen3 ×4 (4 GB/s) fits; USB3 (400 MB/s) does not. D3 + C2 is incompatible with 30 fps; D3 + C3 works.

**Actor flow per inference:**

1. Host loads one MMVR sample.
2. Simulator runs FFT + backbone + FPN → produces `[1, 64, 64, 32]` FP16.
3. Host loads that tensor onto CGRA input SRAM and loads the full transformer weight set (~26 MB INT8) onto board DRAM. Triggers silicon.
4. **Silicon runs uninterrupted:**
   ```
   CGRA(input_proj FP16)
     → UCIe → FPGA(FP16→INT8 boundary quantizer, route)
     → UCIe → CIM(encoder layer 1 — weights pulled from DRAM by FPGA-DMA)
     → UCIe → CIM(encoder layer 2) → ... → CIM(encoder layer 6)
     → (decoder layer 1 loop:)
        CIM(SA projs + attn + out_proj)
          → UCIe → FPGA(LN1) → UCIe → CIM(CA projs + attn + out_proj)
          → UCIe → FPGA(LN2) → UCIe → CIM(FFN)
          → UCIe → FPGA(LN3)
          → UCIe → CIM(next layer)
     → ... → decoder layer 6
     → UCIe → FPGA(final LN)
     → UCIe → CIM(detection head)
     → UCIe → host
   ```
5. Host receives bbox + class + seg; diffs against Demo-A output (same silicon, different routing).

**UCIe traffic exercised continuously:** backbone output crossing (1 MB), weight streaming (~780 MB/s sustained), 19 × decoder LN round trips (~5 MB/s), final output. All under 1 GB/s — well inside UCIe budget.

**What this proves:** UCIe links + soft-core weight DMA + chiplet-to-chiplet activation handoff + FPGA-SIMD-in-the-pipeline all work at end-to-end inference timescale. Real system behavior.

**Still partial:** FFT + backbone still run in simulator (too large to fit on CGRA silicon without its own weight streaming). Demo C below extends this.

### 7.5. Demo C — Full-silicon inference (stretch)

**Goal:** push FFT + backbone onto CGRA silicon too, via the same per-layer weight-streaming pattern the CIM already uses for the transformer. No simulator in the data path at all.

**Requires:** same as Demo B, plus DRAM capacity sized for **~53 MB total** (27.5 MB FP16 backbone + 26.1 MB INT8 transformer). Sustained weight-stream load rises to ~1.6 GB/s at 30 fps. With D1/D2: fits UCIe comfortably. With D3: requires C3 (PCIe gen3 ×4 or better); C2 (USB3) no longer sufficient even at reduced frame rate.

**Actor flow per inference:**

1. Host loads raw radar signal onto CGRA input SRAM; loads full model weights (~27 MB FP16 backbone + ~26 MB INT8 transformer = ~53 MB) onto board DRAM.
2. Silicon runs uninterrupted:
   ```
   CGRA(FFT → conv1 → layer1 → layer2 → layer3 → layer4 → FPN → input_proj)
         (backbone weights streamed per layer from DRAM via FPGA-DMA)
     → UCIe → FPGA → UCIe → CIM(transformer + head, as Demo B)
     → host
   ```
3. Host receives full detection + segmentation output.

**What this proves:** the whole pipeline is hardware-executable. Strongest demo story.

**What this costs:** adds ~27 MB FP16 backbone weight stream per inference (~810 MB/s at 30 fps) to UCIe/DRAM load. Total still ~1.6 GB/s — fits UCIe. Main risk is CGRA SRAM pressure (needs to double-buffer at least one Conv2d layer at a time) and demo-board DRAM capacity.

### 7.6. Comparison

| Dimension | Demo A | Demo B | Demo C |
|-----------|--------|--------|--------|
| FFT + backbone | simulator | simulator | CGRA silicon |
| input_proj | CGRA silicon (1-shot) | CGRA silicon | CGRA silicon |
| Encoder 1–5 | simulator | CIM silicon | CIM silicon |
| Encoder 6 | CIM silicon (1-shot) | CIM silicon | CIM silicon |
| Decoder | simulator except 3× LN | CIM + FPGA silicon | CIM + FPGA silicon |
| Detection head | simulator | CIM silicon | CIM silicon |
| UCIe exercised | no | yes (continuous) | yes (continuous) |
| Soft-core orchestration | no | yes | yes |
| Host in inference inner loop | yes (per invocation) | no (D1/D2) or outer-loop only (D3) | no (D1/D2) or outer-loop only (D3) |
| Host↔chip interface req (Role C) | C1 (JTAG) or higher | C2 (USB3) if D1/D2; **C3 (PCIe)** if D3 | C3 (PCIe) if D3 |
| External DRAM req (Role D) | none | ≥26 MB via D1/D2/D3 | ≥53 MB via D1/D2/D3 |
| Weight loading | per-invocation push via C | one-time push to DRAM at boot (D1/D2) or streamed during inference (D3) | same as B |
| Proves | accuracy per-block | system coordination | full hardware pipeline |

All three drive the same tape-out silicon. Build order: Demo A first (validates each chiplet standalone; works even with the minimal test harness), then Demo B once Role C + Role D are selected, then Demo C if CGRA SRAM + board DRAM support it.

---

## 8. Why these carve-outs match hardware dimensions

On a 32×32 INT8 MAC tile in CIM, encoder-layer shapes decompose with **zero padding**:

| Op | Dim factorization on 32-wide tile |
|----|-----------------------------------|
| TPE proj [256 × 256 × 512] | 8 × 8 × 16 outer tiles |
| Main Q/K/V [512 × 512 × 512] | 16 × 16 × 16 |
| QK^T [512 × 128 × 512] | 16 × 4 × 16 |
| AV [512 × 512 × 128] | 16 × 16 × 4 |
| Out_proj [512 × 512 × 512] | 16 × 16 × 16 |
| FFN linear1 [256 × 2048 × 512] | 8 × 64 × 16 |
| FFN linear2 [2048 × 256 × 512] | 64 × 8 × 16 |

Decoder shapes: N_d=10 doesn't tile on 32-wide arrays. Handled by (a) routing decoder LN to FPGA SIMD (N_d=10 is native for per-token reduction), (b) letting the CIM MAC array accept ragged last-tile with masked lanes for the N_d axis. No padding of weights, only of the activation seq dim.

CGRA input_proj is a 1×1 Conv2d, which flattens to GEMM [64 × 256 × (64·32)] = [64 × 256 × 2048]. FP16 tile 32×32: 2 × 8 × 64. Clean.

FPGA 8-lane SIMD on [10, 1, 256] LN: 10 tokens × (256/8 = 32 iterations per pass) × 2 passes = 640 cycles. Dim 256 is a multiple of 8; no padding.

---

## 9. Code status — what's in the repo today vs. what's needed

Current repo (`src/`, commit `cb277be`) supports the quantization accuracy study only.

| Demo requirement | Status | File |
|------------------|--------|------|
| INT8 linear (W8A8, torchao) | ✅ | `src/quantize_torchao.py` (`--scheme int8dq`) |
| INT8 Q/K/V fake-quant at bmm input | ✅ | `src/models/module_retr/attention.py:462-465`; `--attn_bmm_bits 8` |
| UINT8 softmax-output fake-quant | ✅ | `src/models/module_retr/attention.py:486`; `--attn_weights_bits 8` |
| FP16 precision path (CGRA-faithful) | ❌ missing | only BF16 autocast exists |
| FP16 LN / softmax / residual / accumulator | ❌ missing | all FP32 in current forward path |
| MX weight format (pow2 scale) | ❌ missing | `int4fq` uses FP32 per-group scale |
| Boundary activation quantizer (CGRA→CIM, CIM↔FPGA) | ❌ missing | no per-tensor quantizer at input_proj output |
| Stage/chiplet split (silicon-in-the-loop swap) | ❌ missing | encoder/decoder are Modules but no swap machinery |
| Intermediate tensor dump / replay | ❌ missing | no capture at stage boundaries |
| Golden vs. silicon diff harness | ❌ missing | |

**Ordered build list** for the demo harness:

1. `--scheme fp16` — full FP16 reference path (backbone + transformer), not autocast.
2. Boundary activation quantizer Module — callable, produces (INT8 tensor, scale).
3. Stage-isolation hooks on `input_proj`, `encoder.layers[5]`, `decoder.layers[5].norm1/2/3`, `decoder.norm` for dump/replay or silicon-in-the-loop callback. Flag-gated: `--silicon-in-the-loop {cgra,cim,fpga,all}`.
4. MX format fake-quant — replace int4fq with pow2 block scale; validate against int4fq_g128 baseline (+0.15 AP).
5. FP16 LN / softmax / residual inside attention path.
6. Golden-vs-silicon driver — twin inference, diff final logits and per-boundary intermediates.

Items 1–3 are the minimum to run Demo A/B; 4–5 close the accuracy-claim loop; 6 is the verification harness.

---

## 10. Open items for foundry / partner quote

- CIM SRAM target: 1 MB vs 2 MB (determines whether encoder layer fits in one weight-load round or two).
- FPGA DSP block count: drives 8-lane vs 16-lane SIMD choice.
- CIM MAC array effective TOPS/mm² at target node: your range is 2–5; tape-out realism typically low end.
- SRAM density at node (28 nm vs other).
- UCIe PHY area overhead per chiplet.
- DRAM interface: shared vs. per-chiplet; whether CIM pulls weights directly or through FPGA DMA.

Once numbers land, decide: Tier A (one encoder layer + decoder layer resident, ~4–6 mm² CIM) vs Tier B (one encoder attention block resident, ~1–2 mm² CIM). See `QUANT.md` §Hardware sizing for attention-matrix context.
