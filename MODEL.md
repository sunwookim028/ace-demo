<!--
Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# RETR Model Reference

Concise technical reference for the RETR radar detection transformer. Covers architecture,
layer inventory, input/output dimensions, metrics, dataset structure, and quantization targets.

---

## Architecture Overview

```
Radar heatmaps (hor + ver)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  BACKBONE  (backbone.py)                            │
│  ResNet18 + FPN                                     │
│  conv1: nn.Conv2d(4, 64, 7×7)  ← 4-channel input   │
│  input_proj / input_proj_ver: Conv2d(64, 256, 1×1)  │
│  Top-K spatial token selection (detr.py:200–214)    │
└─────────────────────────────────────────────────────┘
        │  [2×topk, B, 256] — hor + ver tokens fused
        ▼
┌─────────────────────────────────────────────────────┐
│  ENCODER  (transformer.py)                          │
│  ConditionalTransformerEncoder, 6 layers            │
│  Self-attention over radar tokens only              │
│  Object queries NOT present here                    │
└─────────────────────────────────────────────────────┘
        │  memory [2×topk, B, 256]
        ▼
┌─────────────────────────────────────────────────────┐
│  DECODER  (transformer.py)                          │
│  ConditionalTransformerDecoder, 6 layers            │
│  tgt: zeros [num_queries=10, B, 256]                │
│  query_embed: nn.Embedding(10, 256) — enters HERE   │
│  Cross-attention: queries ↔ radar memory            │
└─────────────────────────────────────────────────────┘
        │  hs [6, B, 10, 256]  (6 decoder layer outputs)
        ▼
┌─────────────────────────────────────────────────────┐
│  DETECTION HEADS  (detr.py)                         │
│  class_embed: nn.Linear(256, num_classes)           │
│  bbox_embed:  MLP → 3D bbox (hor, ver, image plane) │
│  RadarToImgProjection → image-plane boxes           │
└─────────────────────────────────────────────────────┘
        │  (+ segmentation path if task=SEG)
        ▼
┌─────────────────────────────────────────────────────┐
│  SEGMENTATION HEAD  (segmentation.py)               │
│  MHAttentionMap + MaskHeadSmallConv + Unet          │
│  Input: hs[-1] + FPN backbone features              │
└─────────────────────────────────────────────────────┘
```

### Key architectural facts

- **Object queries** (`nn.Embedding(10, 256)`, `retr.py:68` / `detr.py:57`): decoder-only.
  Initialized via `ref_point_head(query_pos)` at `transformer.py:235`. Never touch the encoder.
- **Custom MultiheadAttention** (`attention.py`): used in decoder self- and cross-attention.
  Has **no in-proj weights** — Q/K/V projections are external `nn.Linear` layers in `transformer.py`.
  Attention dot-product uses `torch.bmm` (not `F.scaled_dot_product_attention`).
- **Encoder** uses standard `nn.MultiheadAttention(d_model×2, nhead)` with fused `in_proj_weight`.
- **No explicit dtype cast** at any stage. `float32` flows uninterrupted from input to all heads.

---

## Layer Inventory and Baseline Precision

**All layers run at `torch.float32` at inference.** No mixed precision anywhere.
`test.py:52` does `RETR(...).to(device)` with no dtype argument.

| Component | Layer type | Representative shape | Baseline dtype |
|-----------|-----------|----------------------|----------------|
| Backbone conv1 (modified) | `nn.Conv2d(4, 64, 7)` | weight: [64, 4, 7, 7] | float32 |
| Backbone ResNet18 remaining Conv2d | `nn.Conv2d` | various | float32 |
| `FrozenBatchNorm2d` (backbone) | custom, frozen stats | buffers: [C] | float32 |
| `input_proj`, `input_proj_ver` | `nn.Conv2d(64, 256, 1)` | weight: [256, 64, 1, 1] | float32 |
| Encoder `self_attn` | `nn.MultiheadAttention(512, nhead)` | in_proj: [1536, 512] | float32 |
| Encoder projection linears (×6 per layer, ×6 layers) | `nn.Linear(256, 256)` | weight: [256, 256] | float32 |
| Encoder FFN | `nn.Linear(256, 2048)`, `nn.Linear(2048, 256)` | — | float32 |
| Decoder self-attn projection linears (×5 per layer, ×6 layers) | `nn.Linear(256, 256)` | weight: [256, 256] | float32 |
| Decoder cross-attn projection linears (×6 per layer, ×6 layers) | `nn.Linear(256, 256)` | weight: [256, 256] | float32 |
| Decoder FFN | `nn.Linear(256, 2048)`, `nn.Linear(2048, 256)` | — | float32 |
| `MultiheadAttention.out_proj` | `NonDynamicallyQuantizableLinear(256, 256)` | weight: [256, 256] | float32 |
| `nn.LayerNorm` (encoder/decoder) | — | [256] | float32 |
| `query_embed` | `nn.Embedding(10, 256)` | [10, 256] | float32 |
| Detection heads (`class_embed`, `bbox_embed` MLP) | `nn.Linear` | — | float32 |
| Segmentation `MHAttentionMap` | `nn.Linear(256, 256)` ×2 | — | float32 |
| Segmentation `MaskHeadSmallConv` | `nn.Conv2d`, `nn.GroupNorm` | — | float32 |

Sinusoidal positional encodings are explicitly `dtype=torch.float32` (`transformer.py:34, 50–52`).

---

## Input / Output Dimensions

### Model inputs

`RETR.forward(hor, ver)` — defined in `src/models/retr.py`

| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `hor` | `[B, 4, 256, 128]` | float32 | Horizontal plane radar heatmap, 4 temporal frames |
| `ver` | `[B, 4, 256, 128]` | float32 | Vertical plane radar heatmap, 4 temporal frames |

Raw heatmap values: radar energy magnitude (very large positive floats, ~10M–80M range).
After pipeline: log-compressed then z-score normalized per environment → values ~N(0, 1).
Normalization constants per environment: `src/data/parameters_mmvr.py`.

### Dataset loading chain

```
segment_4_3/P2/<env>/<session>/<take>/NNNNN_radar.npz
    └── hm_hori: (4, 256, 128) float32  ← 4 frames stacked by create_grouped_dataset.py
    └── hm_vert: (4, 256, 128) float32

mmvr_dataset.py   → log + z-score normalize → torch [4, 1, 256, 128]
det_seg_dataset.py → permute(1,0,2,3)[0]    → [4, 256, 128]
collate_det_seg    → stack batch             → [B, 4, 256, 128]
```

### Labels dict (per sample, from `dataloader.py:collate_det_seg`)

Required by `Metrics.compute(gt, out)` in `utils/detection_process.py`:

| Field | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `iboxes` | `[5, 4]` | float32 | Image-plane GT boxes, xyxy, padded to 5 |
| `hboxes` | `[5, 4]` | float32 | Horizontal radar-plane GT boxes |
| `vboxes` | `[5, 4]` | float32 | Vertical radar-plane GT boxes |
| `masks` | `[5, H/2, W/2]` | float32 | Binary instance masks (SEG task only) |
| `n_sbj` | scalar | int | Actual subject count (0–5, rest is padding) |
| `labels` | `[5]` | float32 | 0 for valid, -1 for padding |
| `env` | scalar | int | Environment index (5–9 for P2) |
| `file_id` | str | — | Source segment identifier |

### Model outputs (per sample, from `RETR.forward`)

| Field | Shape | Description |
|-------|-------|-------------|
| `iboxes` | `[Q, 4]` | Predicted image-plane boxes, xyxy |
| `hboxes` | `[Q, 4]` | Predicted horizontal radar boxes |
| `vboxes` | `[Q, 4]` | Predicted vertical radar boxes |
| `scores` | `[Q]` | Confidence scores |
| `labels` | `[Q]` | Class indices (all 0, single class) |
| `masks`  | `[H_img, W_img]` | Binary instance mask (SEG task only) |

---

## Accuracy Metrics and Baselines

Evaluated by `src/utils/detection_process.py` `Metrics` class using `torchmetrics`.
Radar-plane equivalents (`det_rad`) are computed but not reported in the paper table.

| Metric | Result dict key | Description |
|--------|----------------|-------------|
| **BBox AP** | `det_img['map']` | COCO mAP @ IoU 0.5:0.95, image plane |
| BBox AP50 | `det_img['map_50']` | AP @ IoU=0.50 |
| BBox AP75 | `det_img['map_75']` | AP @ IoU=0.75 |
| **BBox AR1** | `det_img['mar_1']` | Max recall @ 1 detection per image |
| BBox AR10 | `det_img['mar_10']` | Max recall @ 10 detections |
| **Segm IoU** | `seg_img` | Jaccard index, multiclass (2 classes) |

**Pretrained baselines** (ResNet18 backbone, `logs/pretrained_model/`):

| Split | BBox AP | BBox AR1 | Segm IoU | Checkpoint |
|-------|---------|----------|----------|------------|
| **P2S1** | **46.75** | **42.19** | **77.21** | `p2s1_retr_detseg.pth` (156 MB) |
| P2S2 | 12.19 | 19.70 | 59.93 | `p2s2_retr_detseg.pth` (156 MB) |

---

## Dataset (MMVR)

**Source:** Zenodo [https://zenodo.org/records/12611978](https://zenodo.org/records/12611978)
- `P2_00.zip` (20.9 GB), `P2_01.zip` (22.2 GB), `P2_02.zip` (12.0 GB)

### Directory structure

```
MMVR/P2/<env>/<session>/<take>/     ← raw data (one file per frame)
  e.g. MMVR/P2/d5s3/002/00000_radar.npz   hm_hori/hm_vert: (256,128) float32

MMVR/segment_4_3/P2/<env>/<session>/<take>/  ← after preprocessing
  e.g. MMVR/segment_4_3/P2/d5s3/002/00000_radar.npz  hm_hori/hm_vert: (4,256,128) float32
```

### Naming

| Part | Meaning |
|------|---------|
| `d5`–`d9` | Environment index (d5–d9 = P2 protocol, 5 distinct environments) |
| `s1`–`s6` | Subject/session index within that environment |
| `000`–`009` | Take index — independent continuous recording sequence |

Example: `d5s3/002` = Environment 5, Session 3, Take 002.

### P2S1 split

**Static artifact** in `src/utils/data_split.npz` — no generation script committed.

| Split | Takes | Frames (est.) |
|-------|-------|---------------|
| Train | 219 (80%) | ~100,000 |
| Val   | 27 (10%) | ~12,000 |
| **Test**  | **27 (10%)** | **~12,000** |
| **Total** | **273** | **~124,000** |

P2S1 test takes (27 total, spanning all 3 P2 zip files):

| Zip | Test takes |
|-----|-----------|
| P2_00 (d5, d6) | `d5s3/002` `d5s4/007` `d5s5/003` `d5s6/006` `d6s2/004` `d6s2/006` `d6s2/007` `d6s3/006` `d6s4/009` |
| P2_01 (d7, d8) | `d7s1/002` `d7s1/008` `d7s3/002` `d7s3/004` `d7s4/009` `d7s5/006` `d8s2/001` `d8s4/000` `d8s5/007` `d8s5/009` `d8s6/002` `d8s6/008` |
| P2_02 (d9)     | `d9s1/000` `d9s1/003` `d9s2/009` `d9s4/005` `d9s6/004` `d9s6/009` |

### Preprocessing

```bash
cd src/data
python create_grouped_dataset.py --num_frames 4 --overlap 3 \
  --dataset_dir ../../MMVR --output ../../MMVR
```

Stride = `num_frames − overlap = 1` (one segment per raw frame). Segment N uses raw frames
`[max(0, N−3), max(0, N−2), max(0, N−1), N]` with edge padding.

### Evaluation

```bash
cd src
python test.py \
  --root ./MMVR/segment_4_3 --split P2S1 --task DETSEG \
  --pretrained_path ../logs/pretrained_model/p2s1_retr_detseg.pth
```

---

## Quantization Targets

### Covered by torchao `quantize_(model, scheme, filter_fn=...)`

All `nn.Linear` and `nn.Conv2d` in:
- Backbone: Conv2d layers (ResNet18 + input_proj)
- Encoder: `self_attn.in_proj_weight`, 6 projection linears per layer, FFN
- Decoder: 11 projection linears per layer (`sa_*_proj`, `ca_*_proj`), FFN, `out_proj`

### Requires forward pre-hooks (torchao does not cover)

Attention activation quantization — Q/K/V tensors entering `torch.bmm` inside
`attention.py:multi_head_attention_forward` (lines 447, 467).
Workaround: register `register_forward_pre_hook` on each `MultiheadAttention` module
to fake-quantize Q/K/V before the dot-product. No modification to trusted source files.

### Not quantized (out of scope)

Detection/segmentation heads (small parameter count, directly output-sensitive),
`nn.LayerNorm`, `nn.GroupNorm`, `nn.Embedding`.

### Planned granularities

| Scheme | torchao API | Granularity |
|--------|------------|-------------|
| INT8 weight-only | `int8_weight_only()` | Per output channel (default) |
| INT4 weight-only, group=128 | `int4_weight_only(group_size=128)` | Per group of 128 input elements |
| INT4 weight-only, group=64 | `int4_weight_only(group_size=64)` | Per group of 64 input elements |
| INT4 weight-only, group=32 | `int4_weight_only(group_size=32)` | Per group of 32 input elements |
| INT8 dynamic activation | `int8_dynamic_activation_int8_weight()` | Per-token activation + per-channel weight |

Component targets: `backbone`, `encoder`, `decoder`, `transformer` (enc+dec), `ffn`, `proj`, `all`.
