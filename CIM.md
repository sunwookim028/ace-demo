<!-- Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL) -->
<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# CIM Chiplet

## Role in Automotive Perception

CIM is the arithmetic engine for the transformer stack. It executes all linear projections, attention matrix multiplications (QKᵀ, AV), and element-wise operations (softmax, LayerNorm, activation functions). In the radar perception pipeline it processes every encoder and decoder layer of the transformer — the part of the model that reasons over geometry-enriched tokens and produces object detections.

CIM has no knowledge of model architecture, PE scheme, or attention type. It receives weight tiles and activation tiles, executes MAC operations, and returns results. All scheduling decisions are made by FPGA.

## Evolvability

**What reprogramming covers (via FPGA orchestration):**
CIM's evolvability is largely inherited from FPGA. Because FPGA controls what tiles CIM receives and in what order, CIM transparently handles larger sequence lengths, different hidden dimensions, new weight-load groupings, and new layer counts — without any change to CIM itself.

**What CIM absorbs natively:**
MLP-based PE (PETR-style) requires additional linear projections over every token per layer. These are structurally identical to any other linear layer — CIM runs them on the same MAC array with additional weight traffic managed by FPGA.

**What requires a chiplet upgrade:**
- Sub-INT8 precision (INT4, FP4, FP8 tensor operations) requires native MAC array support not achievable by reprogramming
- Larger batch sizes or hidden dimensions that exceed the physical SRAM scratchpad require a new die with more on-chip memory
- A conv unit (needed for DWConv CPE) cannot be added to CIM; that operation belongs on CGRA
