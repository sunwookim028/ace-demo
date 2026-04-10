<!--
Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# PTQ Experiments Guide

Reference for all post-training quantization experiments: what each configuration measures, how to run the scripts, and how to read results.

---

## Background

All experiments use `src/eval_ptq.py`, which wraps the upstream `test.py` evaluation loop with:
- Quantization via torchao (`--scheme`, `--component`)
- Optional decoder attention activation quantization (`--attn_act_bits`)
- JSON result output to `experiments/results/`
- Accurate quantized model size via `model_size_mb()`

No model weights are modified during training (PTQ only). No QAT. The pretrained checkpoint `logs/pretrained_model/p2s1_retr_detseg.pth` is shared across all runs.

---

## Quantization schemes

| `--scheme` | Method | What changes at runtime |
|-----------|--------|------------------------|
| *(omit)* | FP32 baseline | Nothing — full float32 |
| `bf16` | `torch.autocast(bfloat16)` | Eligible ops cast to bf16 on-the-fly; weights stay float32 |
| `int8wo` | INT8 weight-only (torchao) | Weights stored int8; **dequantized to float32 before matmul**; activations and math fully float32 |
| `int4fq_g128` | INT4 weight-only, group=128 | Symmetric per-group INT4 fake-quant: weights rounded to INT4 precision, dequantized to float32 before matmul |
| `int4fq_g64` | INT4 weight-only, group=64 | Finer grouping → lower rounding error, larger scale overhead |
| `int4fq_g32` | INT4 weight-only, group=32 | Finest grouping in sweep |
| `int8dq` | INT8 dynamic weight+activation (torchao) | Weights int8 (per-channel); activations quantized per-token at runtime **before** matmul; actual INT8 arithmetic |

**Key point for int8wo / int4fq:** Computation is still float32. The hardware benefit is reduced memory bandwidth (smaller weight tensors fetched from DRAM). INT4 fake-quant uses symmetric per-group quantization: scale = max(|W_group|) / 7, rounded to [-8, 7], dequantized back to float32 — identical rounding noise to hardware INT4 weight-only.

**Key point for int8dq:** Both weights and activations are quantized. This targets compute-bound paths and enables actual INT8 GEMM on supported hardware (CUDA, CPU with AVX-VNNI). It is the most aggressive scheme and most likely to impact accuracy.

> **Note on INT4 implementation:** torchao 0.17.0's `Int4WeightOnlyConfig` requires the `mslk` library (not publicly available) for CPU execution. INT4 is implemented directly as per-group symmetric fake-quant in `quantize_torchao._apply_int4_weight_fake_quant`. The accuracy measurement is identical to hardware INT4 weight-only — same rounding noise, same dequant-to-FP32 compute path.

---

## Component filters

`--component` controls which `nn.Linear` modules are quantized. All filters exclude `nn.Conv2d` (backbone) due to a torchao 0.17.0 shape bug on 1×1 convolutions.

| `--component` | Modules matched | Count (approx) |
|--------------|----------------|----------------|
| `backbone` | `nn.Linear` in backbone/input_proj paths | 0 (all Conv2d, no Linear) |
| `encoder` | All Linear in `model.detr.encoder.*` | 72 |
| `decoder` | All Linear in `model.detr.decoder.*` | 89 |
| `transformer` | encoder + decoder | 161 |
| `ffn` | `linear1`, `linear2` in encoder + decoder | 24 |
| `proj` | All transformer Linear except FFN | 137 |
| `all` | backbone + transformer (currently same as `transformer`) | 161 |

The encoder's 4 `self_attn_*_proj` layers (each `Linear(512, 512)`) are the largest individual weights in the model at 1 MB each. All 6 encoder layers contribute 6 × 4 × 1 MB = 24 MB of these alone.

---

## Attention activation quantization

`--attn_act_bits 8` registers forward pre-hooks on all decoder `MultiheadAttention` instances. Before the Q·K^T dot-product (implemented with `torch.bmm`), the hook symmetric-per-tensor fake-quantizes Q, K, and V to INT8 range and dequantizes back to float32. This is a **straight-through estimator** — it measures accuracy impact without hardware INT8 matmul speed.

This is the only way to simulate quantization inside the custom decoder attention, since `torch.bmm` is not intercepted by torchao's weight quantization.

```python
# Example: INT8 weight-only + INT8 attention activations, decoder only
conda run -n retr-quant --cwd src python eval_ptq.py \
    ... --scheme int8wo --component decoder --attn_act_bits 8
```

---

## Full experiment catalogue

### Quick sweep (`experiments/run_sweep_quick.sh`)

4 experiments. Intended for pipeline validation and first accuracy signals before the full sweep. Uses `--device cpu --batch_size 8`.

| # | `--scheme` | `--component` | Purpose |
|---|-----------|--------------|---------|
| 1 | *(none)* | — | FP32 baseline |
| 2 | `int8wo` | `all` | Most common production quantization |
| 3 | `int8wo` | `transformer` | Encoder + decoder only (no backbone) |
| 4 | `int8wo` | `decoder` | Decoder-only ablation |

```bash
bash experiments/run_sweep_quick.sh [--dry-run]
```

### Full sweep (`experiments/run_sweep.sh`)

22 experiments. Uses `--device cuda --batch_size 32`. Covers the full design space for the paper.

| Group | `--scheme` | `--component` | Count |
|-------|-----------|--------------|-------|
| Baseline | *(none)* | — | 1 |
| BF16 | `bf16` | `all` | 1 |
| INT8 weight-only per-component | `int8wo` | backbone, encoder, decoder, transformer, ffn, proj, all | 7 |
| INT4 group size sweep | `int4fq_g{128,64,32}` | `all` | 3 |
| INT4 per-component | `int4fq_g128` | backbone, encoder, decoder, transformer, ffn, proj | 6 |
| INT8 dynamic act+weight | `int8dq` | encoder, decoder, transformer, all | 4 |
| INT8 weight + INT8 attn act | `int8wo` | decoder, all | 2 (with `--attn_act_bits 8`) |
| INT4 + INT8 attn act | `int4fq_g128` | `all` | 1 (with `--attn_act_bits 8`) |

```bash
bash experiments/run_sweep.sh [--dry-run]
```

---

## Running a single experiment

```bash
# From repo root
conda run -n retr-quant --cwd src python eval_ptq.py \
    --root ../MMVR/segment_4_3 \
    --split P2S1 \
    --task DETSEG \
    --pretrained_path ../logs/pretrained_model/p2s1_retr_detseg.pth \
    --batch_size 32 \
    --worker 2 \
    --device cpu \          # or cuda
    --scheme int8wo \       # omit for FP32 baseline
    --component all         # default
```

Override the output filename:

```bash
... --run_name my_experiment_v2
# → experiments/results/my_experiment_v2.json
```

Default output filename: `{split}_{scheme}_{component}[_attn{bits}].json`
e.g. `P2S1_int8wo_all.json`, `P2S1_fp32_all.json`, `P2S1_int4wo_g128_decoder_attn8.json`

---

## Reading result JSON files

Each JSON in `experiments/results/` has the following structure:

```json
{
  "run_name":            "P2S1_int8wo_all",
  "split":               "P2S1",
  "task":                "DETSEG",
  "scheme":              "int8wo",
  "component":           "all",
  "attn_act_bits":       null,
  "pretrained_path":     "../logs/pretrained_model/p2s1_retr_detseg.pth",

  "bbox_ap":             46.75,   ← primary detection metric (mAP @ IoU 0.5:0.95)
  "bbox_ap50":           72.13,   ← AP @ IoU=0.50
  "bbox_ap75":           51.20,   ← AP @ IoU=0.75
  "bbox_ar1":            42.19,   ← recall @ 1 detection/image
  "bbox_ar10":           48.50,   ← recall @ 10 detections/image

  "seg_iou":             77.21,   ← segmentation Jaccard index (null for DET task)

  "model_size_fp32_mb":  156.13,  ← float32 parameter storage
  "model_size_quant_mb":  81.40,  ← quantized storage (int_data + scale tensors)

  "latency_mean_s":      0.04210, ← per-sample wall-clock time (seconds)
  "latency_std_s":       0.00180,
  "n_batches":           2885
}
```

**Baseline reference values (P2S1, FP32):**

| Environment | `bbox_ap` | `bbox_ar1` | `seg_iou` |
|------------|-----------|-----------|----------|
| GPU (original paper, `retr` env) | 46.75 | 42.19 | 77.21 |
| CPU (`retr-quant` env, PyTorch 2.4) | **42.78** | **39.79** | **74.41** |

> Note: CPU floating-point arithmetic differs from GPU. All `eval_ptq.py` runs use CPU results as the baseline for quantization comparisons.

**Model size reference:**

| Scheme | Component | `model_size_quant_mb` | Compression |
|--------|----------|----------------------|-------------|
| FP32 | — | 156.1 MB | 1.0× |
| `int8wo` | `all` | ~81 MB | ~1.9× |
| `int4wo_g128` | `all` | ~45 MB | ~3.5× |

---

## Checkpoint migration

The pretrained checkpoint was saved with the original fused encoder self-attention (`nn.MultiheadAttention.in_proj_weight [1536, 512]`). This fork refactors the encoder layers to use explicit Q/K/V projections — see [architecture.md](architecture.md) — so the checkpoint keys must be remapped on load.

`quantize_torchao.migrate_encoder_mha_state_dict(state_dict)` handles this automatically. It is called in `eval_ptq.py` before `model.load_state_dict()` and is idempotent (safe to call on an already-migrated dict).

**Mapping:**

| Old key | New key | Slice |
|---------|---------|-------|
| `*.self_attn.in_proj_weight [1536, 512]` | `*.self_attn_q_proj.weight [512, 512]` | `[0:512, :]` |
| | `*.self_attn_k_proj.weight [512, 512]` | `[512:1024, :]` |
| | `*.self_attn_v_proj.weight [512, 512]` | `[1024:1536, :]` |
| `*.self_attn.in_proj_bias [1536]` | `*.self_attn_q/k/v_proj.bias [512]` | split by thirds |
| `*.self_attn.out_proj.weight [512, 512]` | `*.self_attn_out_proj.weight` | (copy) |
| `*.self_attn.out_proj.bias [512]` | `*.self_attn_out_proj.bias` | (copy) |

---

## Numerical equivalence test

Before any core-model change, save a float32 reference; after the change, verify outputs are identical within floating-point tolerance:

```bash
# Step 1 — run BEFORE modifying transformer.py
conda run -n retr python src/tests/test_model_refactor.py --save

# Step 2 — run AFTER the refactor + weight migration
conda run -n retr python src/tests/test_model_refactor.py --verify
# Expected: PASS — refactored model is numerically equivalent to original.
```

The test compares per-output-key tensors with `atol=rtol=1e-4`. The current refactor passes with `max_abs=0.00e+00` (exact match in float32, CPU, PyTorch 2.4).

---

## Timing reference

Measured on AMD Ryzen AI MAX+ 395 (16 cores), CPU-only, `--batch_size 8`:

| Stage | Time |
|-------|------|
| Model load + migrate + quantize | ~5 s |
| Full P2S1 test (23,074 samples, bs=8) | ~17 min |
| 4-run quick sweep | ~70 min total |
| 22-run full sweep (GPU recommended) | ~8 hr |
