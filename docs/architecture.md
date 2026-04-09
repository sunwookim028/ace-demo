<!--
Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# RETR Architecture

Deep-dive reference: model pipeline, layer dimensions, baseline precisions, data loading chain, and annotated code walkthroughs.

---

## Pipeline overview

```
Input: rf_hor, rf_ver — [B, 4, 256, 128] float32 each
  (4 stacked radar frames, horizontal and vertical heatmap planes)

   ┌─────────────────────────────────────────────────────────┐
   │  BACKBONE  (models/module_retr/backbone.py)             │
   │  ResNet18 + FPN                                         │
   │  conv1: Conv2d(4→64, 7×7, stride=2) ← 4-channel input  │
   │  FPN output: 4 levels, 64 channels each                 │
   │  input_proj / input_proj_ver: Conv2d(64→256, 1×1)       │
   │  Top-K selection (detr.py:200–214): topk=256 tokens     │
   └────────────────────────┬────────────────────────────────┘
                            │ memory: [512, B, 256]
                            │ (2×topk tokens: hor + ver fused)
   ┌────────────────────────▼────────────────────────────────┐
   │  ENCODER  (models/module_retr/transformer.py)           │
   │  ConditionalTransformerEncoder — 6 layers               │
   │  Self-attention on radar tokens (no object queries)     │
   │  Each layer: MHA(512-dim) → FFN(256→2048→256)           │
   └────────────────────────┬────────────────────────────────┘
                            │ memory: [512, B, 256]
   ┌────────────────────────▼────────────────────────────────┐
   │  DECODER  (models/module_retr/transformer.py)           │
   │  ConditionalTransformerDecoder — 6 layers               │
   │  tgt: zeros [10, B, 256]                                │
   │  query_embed: Embedding(10, 256) → positional input     │
   │  Each layer: SA(256) → CA(512-dim, vdim=256) → FFN      │
   └────────────────────────┬────────────────────────────────┘
                            │ hs: [6, B, 10, 256]
   ┌────────────────────────▼────────────────────────────────┐
   │  DETECTION HEADS  (models/module_retr/detr.py)          │
   │  class_embed: Linear(256→2)                             │
   │  bbox_embed: MLP(256→256→256→6) → 3D bounding boxes     │
   │  RadarToImgProjection → image-plane boxes               │
   └────────────────────────┬────────────────────────────────┘
                            │ (SEG task only)
   ┌────────────────────────▼────────────────────────────────┐
   │  SEGMENTATION HEAD  (models/module_retr/segmentation.py)│
   │  MHAttentionMap(hs[-1], memory) → attention masks       │
   │  MaskHeadSmallConv + Unet → per-instance mask logits    │
   │  Output: masks [B, H_img, W_img], masks_inst [B, N, H, W]│
   └─────────────────────────────────────────────────────────┘

Output: list of B per-image dicts with keys:
  iboxes [Q,4], hboxes [Q,4], vboxes [Q,4], scores [Q], labels [Q]
  masks [H,W], masks_person [N,H,W]   (SEG task only)
```

**Note on object queries:** The 10 learnable object queries enter via `query_embed` at the decoder, not the encoder. The encoder sees only the 512 radar tokens (hor + ver). This is a key difference from vanilla DETR where encoder and decoder see the same sequence.

---

## Layer inventory and baseline precision

All layers run at `torch.float32` at inference — no mixed-precision hardcoding in the upstream model. `test.py:52` calls `RETR(...).to(device)` with no `dtype` argument.

**Backbone** (`models/module_retr/backbone.py`)

| Layer | Type | Weight shape | Precision |
|-------|------|-------------|-----------|
| `backbone.0.body.body.conv1` | `Conv2d(4, 64, 7×7, stride=2)` | [64, 4, 7, 7] | float32 |
| ResNet18 conv2–conv5 | `Conv2d` | various | float32 |
| `FrozenBatchNorm2d` | frozen stats, no gradient | buffers [C] | float32 |
| FPN `inner_blocks.*.0` | `Conv2d(C_in, 256, 1×1)` | [256, C_in, 1, 1] | float32 |
| FPN `layer_blocks.*.0` | `Conv2d(256, 256, 3×3)` | [256, 256, 3, 3] | float32 |
| `input_proj` | `Conv2d(64, 256, 1×1)` | [256, 64, 1, 1] | float32 |
| `input_proj_ver` | `Conv2d(64, 256, 1×1)` | [256, 64, 1, 1] | float32 |

**Encoder** (×6 layers, `ConditionalTransformerEncoderLayer`)

Each encoder layer's self-attention input is formed by `with_pos_concat`, which concatenates content and positional embeddings to produce 512-dim queries, keys, and values before projection.

| Layer | Type | Weight shape | Precision |
|-------|------|-------------|-----------|
| `self_attn_q_proj` | `Linear(512, 512)` | [512, 512] | float32 |
| `self_attn_k_proj` | `Linear(512, 512)` | [512, 512] | float32 |
| `self_attn_v_proj` | `Linear(512, 512)` | [512, 512] | float32 |
| `self_attn_out_proj` | `Linear(512, 512)` | [512, 512] | float32 |
| `ca_qcontent_proj` | `Linear(256, 256)` | [256, 256] | float32 |
| `ca_kcontent_proj` | `Linear(256, 256)` | [256, 256] | float32 |
| `ca_v_proj` | `Linear(256, 256)` | [256, 256] | float32 |
| `ca_qpos_sine_proj` | `Linear(256, 256)` | [256, 256] | float32 |
| `ca_kpos_proj` | `Linear(256, 256)` | [256, 256] | float32 |
| `ca_vpos_proj` | `Linear(256, 256)` | [256, 256] | float32 |
| `linear1` (FFN) | `Linear(256, 2048)` | [2048, 256] | float32 |
| `linear2` (FFN) | `Linear(2048, 256)` | [256, 2048] | float32 |
| `norm1`, `norm2` | `LayerNorm(256)` | — | float32 |

The 4 `self_attn_*_proj` layers replace the original fused `nn.MultiheadAttention.in_proj_weight [1536, 512]`. Checkpoint keys are remapped at load time by `quantize_torchao.migrate_encoder_mha_state_dict()`. See [experiments.md](experiments.md#checkpoint-migration) for details.

**Decoder** (×6 layers, `ConditionalTransformerDecoderLayer`)

| Layer | Type | Weight shape | Precision |
|-------|------|-------------|-----------|
| `sa_qcontent_proj` | `Linear(256, 256)` | [256, 256] | float32 |
| `sa_qpos_proj` | `Linear(256, 256)` | [256, 256] | float32 |
| `sa_kcontent_proj` | `Linear(256, 256)` | [256, 256] | float32 |
| `sa_kpos_proj` | `Linear(256, 256)` | [256, 256] | float32 |
| `sa_v_proj` | `Linear(256, 256)` | [256, 256] | float32 |
| `self_attn.out_proj` | `NonDynamicallyQuantizableLinear(256, 256)` | [256, 256] | float32 |
| `ca_qcontent_proj` | `Linear(256, 256)` | [256, 256] | float32 |
| `ca_qpos_proj` | `Linear(256, 256)` | [256, 256] | float32 |
| `ca_kcontent_proj` | `Linear(256, 256)` | [256, 256] | float32 |
| `ca_kpos_proj` | `Linear(256, 256)` | [256, 256] | float32 |
| `ca_v_proj` | `Linear(256, 256)` | [256, 256] | float32 |
| `cross_attn.out_proj` | `NonDynamicallyQuantizableLinear(256, 256)` | [256, 256] | float32 |
| `linear1` (FFN) | `Linear(256, 2048)` | [2048, 256] | float32 |
| `linear2` (FFN) | `Linear(2048, 256)` | [256, 2048] | float32 |
| `norm1`, `norm2`, `norm3` | `LayerNorm(256)` | — | float32 |

Plus decoder-level: `query_scale` MLP (256→256→256), `ref_point_head` MLP (256→256→3).

**Heads**

| Layer | Type | Weight shape | Precision |
|-------|------|-------------|-----------|
| `query_embed` | `Embedding(10, 256)` | [10, 256] | float32 |
| `class_embed` | `Linear(256, 2)` | [2, 256] | float32 |
| `bbox_embed` (MLP 3-layer) | `Linear(256, 256)` ×2 + `Linear(256, 6)` | — | float32 |
| `box_affine_transformer` | `Linear(10, 64)`, `Linear(64, 128)`, `Linear(128, 4)` | — | float32 |
| Seg `MHAttentionMap` | `Linear(256, 256)` ×2 | — | float32 |
| Seg `MaskHeadSmallConv` | `Conv2d` + `GroupNorm` layers | various | float32 |
| Seg `Unet` | `Conv2d` + `ConvTranspose2d` | various | float32 |

**Total model: 156 MB float32.** After INT8 weight-only quantization of all eligible linears: ~81 MB (~1.9× compression).

---

## Input / output dimensions

### Model inputs

`RETR.forward(hor, ver)` — `src/models/retr.py`

| Argument | Shape | Dtype | Description |
|----------|-------|-------|-------------|
| `hor` | `[B, 4, 256, 128]` | float32 | Horizontal-plane radar heatmap stack |
| `ver` | `[B, 4, 256, 128]` | float32 | Vertical-plane radar heatmap stack |

Raw heatmap values are radar energy magnitude (large positives, ~10M–80M). The data loader applies log-compression then z-score normalization per environment, yielding values approximately N(0, 1). Normalization statistics live in `src/data/parameters_mmvr.py`.

### Internal tensor dimensions (key checkpoints)

| Stage | Tensor | Shape |
|-------|--------|-------|
| After backbone FPN | per-level features | [B, 64, H_l, W_l] |
| After input_proj (hor) | projected tokens | [B, 256, H_l, W_l] |
| After top-K selection | combined tokens | [512, B, 256] |
| After encoder | memory | [512, B, 256] |
| Decoder queries (init) | zeros + query_embed | [10, B, 256] |
| After decoder (all layers) | hs | [6, B, 10, 256] |
| Class logits (last layer) | — | [B, 10, 2] |
| Box predictions (last layer) | — | [B, 10, 6] |
| Segmentation masks | — | [B, H_img, W_img] |

### Model output

`RETR.forward` returns a list of `B` per-image dicts (post-processed, thresholded at `thresh_mask=0.5`):

| Key | Shape | Description |
|-----|-------|-------------|
| `iboxes` | `[Q, 4]` | Image-plane boxes, xyxy format |
| `hboxes` | `[Q, 4]` | Horizontal radar-plane boxes |
| `vboxes` | `[Q, 4]` | Vertical radar-plane boxes |
| `scores` | `[Q]` | Confidence scores (sigmoid of class logit) |
| `labels` | `[Q]` | Class indices (always 0, single class) |
| `masks` | `[240, 320]` | Merged binary segmentation mask |
| `masks_person` | `[Q, 240, 320]` | Per-instance binary masks |

Q ≤ 10 (number of queries passing the threshold).

---

## Data loading chain

```
MMVR/segment_4_3/P2/<env>/<session>/<take>/NNNNN_radar.npz
    └── hm_hori: (4, 256, 128) float32   ← 4 consecutive frames
    └── hm_vert: (4, 256, 128) float32

  ┌── src/data/mmvr_dataset.py (MMVR.__getitem__)
  │     1. np.load() radar, bbox, mask files
  │     2. log-compress:  x = np.log(x + 1e-10)        (if log_scale=True)
  │     3. z-score:       x = (x - mean_env) / std_env  (per environment)
  │     4. NaN → env NaN value
  │     5. return torch.tensor, shape [4, 1, 256, 128]
  │
  ├── src/data/det_seg_dataset.py (MMVRDetSeg.__getitem__)
  │     1. call MMVR.__getitem__
  │     2. permute: [4, 1, 256, 128] → squeeze → [4, 256, 128]
  │     3. build label dict: iboxes, hboxes, vboxes, masks, n_sbj, labels, env, file_id
  │        (boxes padded to 5 subjects; labels[i]=-1 for padding)
  │
  └── src/data/dataloader.py (collate_det_seg)
        1. torch.stack hm_hori → [B, 4, 256, 128]
        2. torch.stack hm_vert → [B, 4, 256, 128]
        3. labels: list of B dicts (NOT stacked — variable subject count)
        4. return {"hm_hori": ..., "hm_vert": ..., "labels": [...]}
```

The split (train/val/test take lists) is stored as a static artifact in `src/utils/data_split.npz`, loaded by `det_seg_dataset.py` via a hardcoded path `./utils/data_split.npz` relative to CWD. Always run evaluation scripts from the `src/` directory (handled automatically by `eval_ptq.py:os.chdir`).

**P2S1 test split stats:**
- 27 takes, environments d5–d9
- 23,074 preprocessed segments (frames)
- 2,885 batches at batch_size=8; ~720 at batch_size=32

---

## Code walkthrough: forward pass

The full forward pass through `eval_ptq.py` for one batch:

```python
# eval_ptq.py:161–170
batch = move_to_device(batch, device)
rf_hor = batch["hm_hori"].detach()    # [B, 4, 256, 128]
rf_ver = batch["hm_vert"].detach()    # [B, 4, 256, 128]

with torch.no_grad():
    out = model(rf_hor, rf_ver)       # list of B dicts
```

Inside `RETR.forward` (`src/models/retr.py:155–285`):

```python
# retr.py:159 — wraps DETRsegm (segmentation.py:127)
out = self.model(hor, ver)
#  ↳ segmentation.py:167–178: backbone forward
#       backbone_hor = self.detr.backbone(hor)   # ResNet18 + FPN
#       backbone_ver = self.detr.backbone(ver)
#       src_hor = self.detr.input_proj(backbone_hor[-1])  # [B, 256, H, W]
#       src_ver = self.detr.input_proj_ver(backbone_ver[-1])
#  ↳ segmentation.py:181: decoder
#       hs, reference = self.detr.decoder(tgt, memory, ...)
#         ↳ transformer.py:238–257: 6 decoder layers
#           each layer: SA → CA → FFN → LayerNorm residuals
#  ↳ segmentation.py:189–222: segmentation path
#       bbox_mask = self.bbox_attention(hs[-1], memory)
#       seg_masks = self.mask_head(src_proj, bbox_mask, fpn_features)

# retr.py:162–218: compute detection predictions
#   class_embed → class logits [B, 10, 2]
#   bbox_embed  → 6D normalized boxes [B, 10, 6]
#   RadarToImgProjection → image-plane xyxy boxes
#   box_affine_transformer → refined image-plane boxes

# retr.py:260–285: postprocess and threshold (no gradient context needed)
#   For each image: filter by scores > thresh_mask, build output dict
```

### Encoder self-attention detail

`transformer.py:154–171`, `ConditionalTransformerEncoderLayer.forward_post`:

```python
_, _, n_model = src.shape                   # n_model = 256

# with_pos_concat builds 512-dim Q/K/V by concatenating content + sinusoidal pos
q, k, v = self.with_pos_concat(src, src, src, pos, pos, pos)
#   q, k, v: [512, B, 512]  (2×topk tokens, 512-dim each)

# Explicit projection via F.multi_head_attention_forward
src2 = F.multi_head_attention_forward(
    q, k, v,
    embed_dim_to_check=512, num_heads=4,
    in_proj_weight=None,
    in_proj_bias=torch.cat([q_proj.bias, k_proj.bias, v_proj.bias]),
    ...
    use_separate_proj_weight=True,
    q_proj_weight=self.self_attn_q_proj.weight,   # [512, 512]
    k_proj_weight=self.self_attn_k_proj.weight,   # [512, 512]
    v_proj_weight=self.self_attn_v_proj.weight,   # [512, 512]
    out_proj_weight=self.self_attn_out_proj.weight # [512, 512]
)[0][:, :, :n_model]                             # → [512, B, 256]
```

The `[:, :, :n_model]` slice discards the positional half of the 512-dim output, keeping only the 256-dim content representation.

### Decoder cross-attention detail

`transformer.py:351–396`, `ConditionalTransformerDecoderLayer.forward_post`:

```python
# Cross-attention uses custom MultiheadAttention (attention.py)
# No fused in_proj_weight — projections are all external nn.Linear modules
# Q from queries (content + sine pos), K from memory (content + pos), V from memory
tgt2 = self.cross_attn(
    query=q,    # [10, B, 512]  (content + pos concatenated)
    key=k,      # [512, B, 512]
    value=v,    # [512, B, 256] (no pos for value)
    ...
)
# attention.py uses torch.bmm for the dot-product (not F.sdpa)
# out_dim=self.vdim=256 slices the output to 256-dim
```

---

## Code walkthrough: quantization path

`eval_ptq.py:120–134`, with `--scheme int8wo --component all`:

```python
# 1. Load checkpoint and migrate encoder MHA keys
model = RETR(task="SEG").to(device)
params = torch.load(pretrained_path, map_location=device)
params = migrate_encoder_mha_state_dict(params)
#   Remaps:  *.self_attn.in_proj_weight [1536, 512]
#         →  *.self_attn_q/k/v_proj.weight [512, 512] each
#   Remaps:  *.self_attn.out_proj.weight → *.self_attn_out_proj.weight
model.load_state_dict(params)
model.eval()

# 2. Apply quantization (in-place, no model copy)
apply_ptq(model, scheme="int8wo", component="all")
#   quantize_torchao.py:
#   quant_config = Int8WeightOnlyConfig()
#   filter_fn = _is_all  (matches nn.Linear in encoder + decoder)
#   quantize_(model, quant_config, filter_fn=filter_fn)
#   → 161 nn.Linear weights become AffineQuantizedTensor
#   → weight storage: int8 data [out, in] + float32 scale [out]
#   → dequantized to float32 before each matmul at runtime

# 3. Forward is unchanged — all activations remain float32
out = model(rf_hor, rf_ver)

# 4. model_size_mb() correctly sums AffineQuantizedTensor storage
#   FP32: 156 MB → INT8wo all: ~81 MB
```

### Why backbone Conv2d is excluded

torchao 0.17.0 has a shape bug in per-channel `Int8WeightOnlyConfig` on 1×1 `Conv2d` — it attempts `scale.view([256, 64, 1, 1])` but scale has only 256 elements (per output channel), which fails for layers where `in_channels == 1`. All backbone layers are `Conv2d`, so `_is_backbone`, `_is_all`, etc. exclude them via `isinstance(mod, nn.Linear)` only. Backbone quantization is effectively a no-op until this upstream bug is resolved.
