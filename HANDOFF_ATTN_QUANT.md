<!--
Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Handoff: Attention Matmul Quantization

**For a fresh agent starting this work. Read this file in full before touching any code.**

---

## Goal

All existing PTQ experiments quantize `nn.Linear` weights (and optionally activations entering linear layers). The attention dot-product matmuls — QK^T and AV — are untouched and run in FP32.

This task implements **per-tensor symmetric INT8 fake-quantization of attention matmul inputs** (Q, K, V tensors going into QK^T and AV), measures the AP/IoU impact, and logs results to `experiments/results/`.

**Scope:** accuracy measurement only (fake-quant / straight-through estimator). No performance optimization. The results directly inform hardware design: whether the custom INT8 MAC array can handle attention matmuls without accuracy loss.

---

## Context: what is already done

Read memory files at `/home/sk3463/.claude/projects/-home-sk3463-ace-demo/memory/` for full project context. Key points:

- **Baseline:** AP = 42.78, Seg IoU = 74.41 (P2S1 CPU, FP32)
- **INT8 linear weight+act (int8dq):** AP = 42.66, Δ = −0.12 — essentially lossless
- **All prior experiments leave attention matmuls in FP32**
- env: `conda activate retr-quant` (Python 3.11, PyTorch 2.11, torchao 0.17.0)
- all experiments run as: `cd src && python eval_ptq.py --root ../MMVR/segment_4_3 --split P2S1 --task DETSEG --pretrained_path ../logs/pretrained_model/p2s1_retr_detseg.pth --batch_size 8 --worker 2 --device cpu`

---

## Architecture: where attention matmuls live

### Decoder attention — already partially handled

`src/models/module_retr/attention.py` defines `MultiheadAttention` (CustomMHA). Its `forward` calls `multi_head_attention_forward()` (local function, same file, line 226), which contains the two bmm calls:

```python
# attention.py line 448
attn_output_weights = torch.bmm(q, k.transpose(1, 2))   # QK^T
# ...softmax...
# attention.py line 468
attn_output = torch.bmm(attn_output_weights, v)          # AV
```

`src/quantize_torchao.py` already has `register_attention_act_quant_hooks()` which registers forward pre-hooks on `CustomMHA` instances, fake-quantizing `query`, `key`, `value` args before they enter forward. This covers **decoder self-attention and decoder cross-attention Q/K/V inputs**.

**Gap:** the hook fires before `F.multi_head_attention_forward` projects Q/K/V through in_proj. The Q/K/V tensors that actually enter the bmm are the projected, scaled, head-split versions computed *inside* `multi_head_attention_forward`. The hook quantizes the inputs to the projection, not the inputs to the bmm.

### Encoder self-attention — not hookable via module

`src/models/module_retr/transformer.py`, class `ConditionalTransformerEncoderLayer`, method `forward_post` (line ~167) and `forward_pre` (line ~208):

```python
q, k, v = self.with_pos_concat(src, src, src, pos, pos, pos)
src2 = F.multi_head_attention_forward(
    q, k, v,
    ...
    use_separate_proj_weight=True,
    q_proj_weight=self.self_attn_q_proj.weight,   # nn.Linear — IS torchao-quantized
    k_proj_weight=self.self_attn_k_proj.weight,
    v_proj_weight=self.self_attn_v_proj.weight,
    ...
)
```

The Q/K/V projections (`self_attn_q_proj`, `self_attn_k_proj`, `self_attn_v_proj`) are `nn.Linear` and are covered by torchao. But `F.multi_head_attention_forward` is a free function — there is no module to attach a hook to. The projected, head-split Q/K/V that enter the bmm are local variables inside the function.

---

## Implementation plan

### 1. Fix the existing decoder hook (quantize at bmm input, not projection input)

The correct hook point for the decoder is *inside* `multi_head_attention_forward`, on the head-split q, k, v locals just before `torch.bmm`. The cleanest way without modifying `attention.py` is to **monkey-patch `torch.bmm`** locally within the evaluation scope, or to **modify `multi_head_attention_forward` to accept optional fake-quant callbacks**.

**Recommended approach:** add an optional `qkv_fake_quant` argument to the local `multi_head_attention_forward` in `attention.py`. When set, it is applied to q, k, v just before `torch.bmm`. This is a minimal, surgical change to one trusted file.

```python
# attention.py: near line 448, after head-split, before bmm
if qkv_fake_quant is not None:
    q = qkv_fake_quant(q)
    k = qkv_fake_quant(k)
    v = qkv_fake_quant(v)
attn_output_weights = torch.bmm(q, k.transpose(1, 2))
```

Propagate the parameter up through `MultiheadAttention.forward` (which calls `multi_head_attention_forward`).

### 2. Add encoder attention matmul fake-quant

For the encoder, `ConditionalTransformerEncoderLayer` calls `F.multi_head_attention_forward` (PyTorch built-in, not the local one). Verify this in `transformer.py` imports. If it is calling the local function, pass `qkv_fake_quant` through. If it truly calls `torch.nn.functional.multi_head_attention_forward`, replace those calls with the local version and pass the callback.

Check `transformer.py` imports at the top for clarity before deciding.

### 3. New API in `quantize_torchao.py`

Add a new function `register_attention_bmm_quant_hooks(model, bits=8)` — name chosen to distinguish from the existing `register_attention_act_quant_hooks` which quantizes projection inputs.

```python
def register_attention_bmm_quant_hooks(model, bits=8):
    """
    Fake-quantize Q, K, V at the bmm input (post-projection, post-head-split).
    Covers both encoder (ConditionalTransformerEncoderLayer) and decoder (CustomMHA).
    Returns list of handles/state for cleanup.
    """
```

The "hooks" here may be implemented as patches to `qkv_fake_quant` args rather than PyTorch hook handles — return whatever is needed for cleanup.

### 4. New `--attn_bmm_bits` flag in `eval_ptq.py`

Mirror the existing `--attn_act_bits` flag:

```python
parser.add_argument("--attn_bmm_bits", default=None, type=int, choices=[8],
    help="Fake-quantize Q/K/V at bmm input (post-projection) to INT{bits}. "
         "Covers encoder and decoder attention matmuls.")
```

Add `attn_bmm_bits` to `build_run_name` suffix (`_bmm8`) and to the JSON result dict.

### 5. Canonical quantization method to use

Symmetric per-tensor INT8, same as `_make_fake_quant_int8` already in `quantize_torchao.py`:

```
scale = max(|x|) / 127
x_q = clamp(round(x / scale), -128, 127)
x_fq = x_q * scale
```

This is the **straight-through estimator (STE)** — standard for PTQ accuracy measurement. References: PyTorch `torch.ao.quantization.FakeQuantize`; FQ-ViT (Lin et al., ICCV 2021); Q-DETR (Xu et al., CVPR 2023).

Per-tensor is appropriate here (not per-token or per-channel) because Q, K, V are already normalized via LayerNorm and position encoding — outliers are bounded.

---

## Experiments to run (spawn as parallel subagents)

Each experiment takes ~17 min on CPU. Run in parallel where hardware allows.

| Run name | `--scheme` | `--component` | `--attn_bmm_bits` | Question answered |
|----------|-----------|--------------|-------------------|-------------------|
| `P2S1_fp32_all_bmm8` | *(none)* | — | 8 | Attention matmul quantization cost in isolation |
| `P2S1_int8wo_all_bmm8` | `int8wo` | `all` | 8 | W8 + A8-attn combined |
| `P2S1_int8dq_all_bmm8` | `int8dq` | `all` | 8 | Full W8A8 everywhere |
| `P2S1_int4fq_g128_all_bmm8` | `int4fq_g128` | `all` | 8 | W4 + A8-attn |

The first run is the most informative — isolates the cost of INT8 attention matmuls with FP32 linear layers.

---

## Verification before running experiments

1. Write a unit test: create a small random [4, 512, 512] bmm, apply fake-quant, verify `max_abs_diff < 0.1 * max(|input|)` — confirms quantization is within expected range.
2. Run FP32 baseline through the new code path with `attn_bmm_bits=None` — output must match existing `P2S1_fp32_all.json` exactly (AP = 42.78). If not, the code path change introduced a regression.

---

## Files to touch

| File | Change |
|------|--------|
| `src/models/module_retr/attention.py` | Add `qkv_fake_quant=None` param to `multi_head_attention_forward`; apply at bmm input |
| `src/models/module_retr/transformer.py` | Pass `qkv_fake_quant` through encoder `F.multi_head_attention_forward` calls (if using local fn) |
| `src/quantize_torchao.py` | Add `register_attention_bmm_quant_hooks()` |
| `src/eval_ptq.py` | Add `--attn_bmm_bits` flag, wire into pipeline |
| `experiments/results/` | New JSON files (auto-generated by eval_ptq.py) |
| `README.md` | Fill in pending rows in attention matmul quantization table |

Do **not** modify: `src/models/retr.py`, `src/data/`, `src/utils/`, any upstream model files beyond `attention.py` and `transformer.py`.

---

## Definition of done

- `python eval_ptq.py ... --attn_bmm_bits 8` runs without error for all 4 experiment configs
- FP32 baseline with `attn_bmm_bits=None` reproduces AP = 42.78 exactly
- 4 result JSONs written to `experiments/results/`
- README results table updated with actual numbers
- Git commit with all changes
