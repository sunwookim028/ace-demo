<!-- Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL) -->
<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# CGRA Chiplet

## Role in Automotive Perception

CGRA handles all spatially-structured compute that precedes or feeds the transformer token stream. Today this is the radar sensor backbone: CNN feature extraction over the raw radar heatmap, producing the sparse top-K tokens that the transformer encoder consumes. Near-term it expands to multi-modal fusion: processing camera feature maps alongside radar tokens to produce a jointly-enriched representation.

CGRA's domain is operations with spatial locality — convolutions, pooling, bilinear interpolation over dense feature maps. These do not map efficiently to CIM's GEMM array or to FPGA's memory-orchestration fabric.

## Evolvability

**Backbone architecture.** Radar backbone CNNs will grow denser and deeper as sensor resolution increases and accuracy requirements tighten. CGRA bitstream reprogramming absorbs new backbone architectures — deeper ResNets, hybrid CNN+attention stems — without changing the downstream transformer chiplets.

**Multi-modal fusion.** Deformable cross-attention (CRN, RCBEVDet-class models) requires gathering K=4 bilinearly-interpolated samples from a large camera feature map per query. This is spatially-structured compute over a dense buffer — the same class of operation as the backbone. CGRA handles it; FPGA dispatches gather descriptors and receives fused tokens. The transformer on CIM sees a standard token buffer regardless of whether fusion occurred.

**DWConv CPE.** TransRAD-style models insert a depthwise conv between transformer layers as conditional positional encoding. This requires spatial 2D conv over the token sequence — a natural CGRA operation, not achievable on CIM or in the FPGA DMA path.

**What requires a chiplet upgrade:**
If the fusion feature map grows substantially (e.g., multi-camera rigs, high-resolution BEV grids) the CGRA's internal memory bandwidth and compute density become the bottleneck. A chiplet upgrade with larger on-chip buffer or higher MAC density would be the response — the interface to FPGA and CIM remains the same (token buffers over UCIe).
