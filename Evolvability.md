# Evolvability: 4D Imaging Radar Transformer Accelerator

This document motivates the evolvability requirements at application and algorithm levels, then maps them to the three-chiplet architecture. Each chiplet's coverage is detailed in its own document.

→ [FPGA.md](FPGA.md) — memory orchestration, tiling strategy, PE injection
→ [CGRA.md](CGRA.md) — spatial compute: backbone, fusion, DWConv
→ [CIM.md](CIM.md) — arithmetic engine: GEMM, BMM, FlashAttention

---

## 1. Application-Level Motivations

**4D imaging radar** measures range, azimuth, elevation, and radial velocity per point. It is weather-robust, directly velocity-aware, and lower cost than LiDAR. Market: ~$2B today → >$10B within a decade, driven by ADAS deployment and AV sensing.

The product is not a single model running on a fixed sensor. Three forces drive the need for hardware evolvability:

**OEM sensor diversity.** Radar chipsets vary by manufacturer in azimuth/elevation resolution, point density, and FOV. The geometry encoding (PE) must be tunable per deployment — RETR's TPE scalars (α) are one instance of this; future OEM integrations will differ. Silicon that hardcodes PE is obsolete at the next design-in.

**ADAS capability ladder.** L2 products ship with radar-only 3D detection (RETR-class). L3/L4 requires radar+camera fusion for full semantic scene understanding (CRN, RCBEVDet-class). This is a staged product roadmap over the same silicon lifetime (5–7 year automotive cycles), not a one-time design decision.

**Accuracy requirements tighten over time.** Euro NCAP and NHTSA upgrade detection benchmarks on a ~2-year cycle. Models that meet today's requirements may need architectural upgrades (denser backbones, iterative decoder refinement, larger N) within the product's service life. The hardware must absorb these without a respin.

---

## 2. Algorithm-Level Motivations

Three axes dominate algorithm evolution for 4D radar perception:

### Axis A — Geometry / Positional Encoding

Radar's non-Cartesian polar observation geometry (range × azimuth × elevation, non-uniform density) means PE is not a solved problem. Every new model redesigns it:

| PE Scheme | Model | What changes vs. RETR |
|---|---|---|
| Tunable polar (TPE) | RETR | Two learnable scalars α on sinusoidal; injected into Q/K each encoder layer |
| Fourier XYZ | 3DETR | Fixed Fourier basis on 3D coords; injected once at encoder input |
| Dynamic anchor | DAB-DETR, DINO | PE recomputed from decoder output each layer; feedback loop inside decoder |
| MLP-based 3D | PETR, CMT | MLP over frustum coords runs on every token every layer; compute-heavy |
| DWConv CPE | TransRAD | 3×3 depthwise conv between transformer layers; requires 2D spatial token layout |

The direction of travel: PE is becoming more compute-intensive and more tightly coupled to the token stream, not less.

### Axis B — Doppler Velocity Encoding

Radar is uniquely velocity-aware: a single frame yields both position and velocity per point. Camera and LiDAR models compensate for missing velocity with temporal buffering. No published model exploits this at the PE level — every surveyed model appends Doppler as a feature channel at the backbone stage. The architectural opportunity is encoding velocity as a positional dimension alongside range/azimuth/elevation, or as an attention-gating modality that directly modulates token relevance. This is an open research gap that the hardware must accommodate: velocity-aware PE is structurally another table-swap on the FPGA injection path, the same mechanism as TPE or learned-lookup PE.

### Axis C — Multi-Modal Fusion

Radar alone cannot classify objects reliably. The product evolution is:

```
Radar-only detection (today)
  → Radar + camera BEV fusion (near-term, CRN / RCBEVDet)
  → Temporal fusion with Doppler consistency (mid-term)
```

Fusion introduces deformable cross-attention: each radar query samples K=4 learned offset locations from a 20 MB camera feature map via bilinear interpolation. This is structurally a second backbone stage (spatial compute over a dense feature map), not a transformer operation. It sits on CGRA.

### Supporting evidence: model survey

The table below covers the relevant model space from radar-native to camera-fusion to LiDAR analogues. Hardware design targets the Tier 1 and Tier 2 models; Tier 3 are reference only.

**Tier 1 — Radar-native**

| Model | Attention | PE | Decoder | Seq N |
|---|---|---|---|---|
| RETR | Full MHSA | TPE (polar, per-layer) | Parallel, 6L | 512 |
| TransRAD | Retentive MaSA (decay) | DWConv CPE | — | BEV grid |
| RadarFormer | Vector attn (Hadamard) | None (coords as feat) | MLP only | 128 |
| RPFA-Net | SA per pillar | Implicit coord feat | None (RPN) | ~20–100/pillar |
| RCBEVDet | DMSA (Gaussian decay) | Learned BEV | BEV head | BEV grid |
| CRN | Deformable CA | Pre-LN; ref pt coords | None | BEV queries |

**Tier 2 — LiDAR/point-cloud analogues**

| Model | Attention | PE | Decoder | Seq N |
|---|---|---|---|---|
| 3DETR | Std MHSA | Fourier 3D XYZ | Parallel, 8L | 2048 |
| TransFusion | SA + CA (LiDAR→Cam) | Heatmap center init | 2-layer seq | ~16K BEV |
| SparseFusion | SA + deformable CA | Learned MLP PE | 1L per modality | 200+200 sparse |
| CenterFormer | Multi-scale + temporal CA | Center coords | Center-query | 500–1K |

**Tier 3 — Background reference**

DETR family (architectural ancestors), BEVFormer/PETR (camera BEV, N=6K–40K, only relevant for fusion), ViT-B/Swin-T (vision baselines showing Pre-LN + GELU as modern default).

**Pre-LN vs. Post-LN placement**

Relevant for fused LN+linear kernel optimization: Pre-LN feeds directly into projection weights (natural fusion); Post-LN sits after the residual add (harder to fuse). Pre-LN is the modern default and is becoming dominant in radar and fusion models.

| Placement | Models |
|---|---|
| Pre-LN | RETR, ViT-B, Swin-T, 3DETR, CRN |
| Post-LN | DETR, Deformable DETR, DINO-DETR, BEVFormer, PETR |

**Decoder parallelism and hardware implications**

Whether decoder layers can overlap with weight prefetch depends on whether each layer has a data dependency on the previous layer's output beyond the token buffer.

| Style | Models | Decoder layers pipeline? | HW implication |
|---|---|---|---|
| Parallel fixed queries | RETR, 3DETR, PETR, DN-DETR | Yes — no inter-layer dependency | Decoder layers overlap with prefetch |
| Iterative refinement | Deformable DETR, DAB-DETR, DINO, CMT | No — box coords update PE for next layer | FPGA feedback path is on critical latency path |
| Two-stage modality sequential | TransFusion | No — camera CA waits for LiDAR CA | Inter-chiplet round-trip per decoder layer |
| Temporal recurrent | BEVFormer TSA | No — requires prior-frame BEV buffer | External frame buffer; not single-inference |

For iterative decoder support, the FPGA feedback register path (sigmoid + PE reinjection) is a latency cost per decoder layer, not just area overhead.

---

## 3. Hardware Evolvability Response

The three-chiplet split assigns each evolvability axis to the unit best suited to absorb it without requiring a full respin:

| Evolvability Axis | Covered By | Mechanism |
|---|---|---|
| Attention kernel type (MHSA → deformable → retentive) | FPGA | FSM/descriptor reprogram |
| Sequence length N | FPGA | Tile loop bound reprogram |
| PE scheme (sinusoidal → learned → TPE scalar) | FPGA | PE table swap + injection enable/disable |
| Iterative decoder (anchor feedback) | FPGA | Feedback path enable; sigmoid + PE table |
| Hidden dim d, FFN width | FPGA + CIM | DMA descriptor reprogram; CIM sees larger tiles |
| Radar backbone evolution (denser CNN, larger N) | CGRA | Bitstream reprogram |
| Multi-modal fusion (deformable gather + bilinear interp) | CGRA | New bitstream configuration |
| DWConv CPE (TransRAD) | CGRA | Conv unit, same reconfigurable fabric |
| MLP-based PE (PETR-style) | CIM | Additional linear layers; same MAC datapath |
| Quantization scheme (INT8 → INT4/FP4) | CIM + FPGA | CIM MAC precision config; FPGA scale/dequant in DMA |

**What requires new silicon:** a fundamentally different MAC precision that the CIM array cannot express (e.g., FP8 tensor cores), or SRAM demand that exceeds the physical scratchpad (B≥4 at d=256, or d≥512 at B=1).

---

## 4. Key Architectural Constraints

**FlashAttention is mandatory for RETR today.** The encoder QKᵀ buffer at [4, 512, 512] INT32 = 4 MB — 4× the CIM SRAM budget. Tiling with B_r=B_c=64 fits in ~290 KB. This is handled entirely by FPGA orchestration; CIM sees only tiles. See CIM.md §Attention buffer sizes.

**Geometry encoding is mid-transformer, not pre-transformer.** TPE runs at the start of each of the 6 encoder layers, not once before the stack. PE injection in the FPGA DMA path therefore occurs at every inter-layer boundary, not just at stack entry.

**Multi-modal fusion is another backbone stage.** Deformable CA gather requires compute against a 20 MB camera feature map. This is spatial compute (CGRA domain), not a side-effect of memory copying (FPGA domain). FPGA dispatches gather descriptors; CGRA returns fused tokens.
