<!-- Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL) -->
<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# FPGA Chiplet

## Role in Automotive Perception

FPGA is the memory orchestration and data flow control layer. It sits between DRAM and the two compute chiplets (CIM, CGRA) over UCIe, and owns all decisions about how data moves: weight streaming to CIM, activation tile sequencing for FlashAttention, PE injection into the inter-layer token stream, and deformable gather descriptor dispatch to CGRA.

FPGA performs no heavy arithmetic. Its value is that all orchestration logic is bitstream-programmable — model changes that affect data flow, tiling strategy, or PE scheme are absorbed here without touching CIM or CGRA silicon.

## Evolvability

**Attention kernel type.** Switching between full MHSA, deformable gather, and retentive/decay attention changes the FSM and DMA descriptor program on FPGA. CIM sees the same tile format regardless of which kernel is active.

**Sequence length N.** Larger N (denser radar, higher-resolution backbone) requires more FlashAttention tile passes. FPGA reprogram adjusts loop bounds; CIM tile shape is unchanged.

**PE scheme.** Between transformer layers, FPGA injects positional encodings into the token stream in-flight during DMA — additive sinusoidal, learned-lookup, and TPE scalar variants are all table swaps. Enabling the iterative decoder feedback path (anchor output → sigmoid → PE reinjection before next decoder layer) is also a bitstream change.

**Hidden dimension and layer structure.** Larger d or new layer counts change weight DMA descriptor chains and tile dimensions. FPGA reprogram; no CIM change.

**Quantization scale management.** Per-layer scale factors and dequantization steps for INT8/INT4 are managed in the DMA path. New quantization schemes adjust the scale broadcast logic.

**What requires more than reprogramming:**
Operations that are compute-heavy rather than memory-orchestration belong on CIM or CGRA. MLP-based PE (PETR-style linear projections over every token) goes to CIM. Bilinear interpolation for deformable fusion gather goes to CGRA. FPGA dispatches descriptors and receives results but does not execute the arithmetic.
