<!--
Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->
# RETR — FFT + Backbone + Tokenizer (CGRA chiplet)

  WHAT IS RETR?  (one paragraph)
  ══════════════════════════════════════════════════════════
    RETR = Radar dEtection TRansformer (Yataka et al., NeurIPS
    2024, arXiv:2411.10293).  It extends DETR to **multi-view**
    indoor radar: two orthogonal MIMO-FMCW radars produce a
    *horizontal* heatmap (azimuth × depth) and a *vertical*
    heatmap (elevation × depth) that share the **depth** axis.
    RETR fuses the two views by (1) extracting per-view CNN
    features with a shared-weight ResNet18 backbone, (2)
    tokenizing each feature map via a parameter-free top-K
    magnitude selector, (3) running a DETR-style transformer
    encoder + decoder that produces N=10 "object queries"
    describing 3D BBoxes in the radar coordinate system, and
    (4) projecting those 3D BBoxes to the 2D image plane via a
    learnable radar-to-camera transform + pinhole projection
    (detection + optional segmentation).

    This file covers stages (1)–(2), which together form the
    CGRA chiplet.  Stages (3)–(4) are in TRANSFORMER.md (CIM
    chiplet) and the downstream detection/segmentation heads.

  SEMANTIC ROLE OF THIS CHIPLET
  ══════════════════════════════════════════════════════════
    Input  : per-view radar heatmap [C_in=4 temporal frames,
             H_in=256 depth bins, W_in=128 angular bins] FP16,
             one for horizontal and one for vertical view.
             (In the real HW path, FFT is also done here on a
             pre-FFT ADC signal; in the PyTorch eval path the
             MMVR dataset ships post-FFT heatmaps directly.)
    Output : N_e = V × topk = 512 INT8 radar tokens at hidden
             dim d=256, plus a matching sequence of tunable
             positional encodings (TPE, α=0.6 depth ratio),
             ready to be self-attended by the CIM transformer.

    The tokenizer's L2-norm + top-K is a *parameter-free*
    selector: it picks the K=256 spatial locations per view
    whose d=256-dim feature vector has the largest L2 norm.
    Selected indices vary frame-to-frame, but the operation is
    differentiable (gradient flows through the gathered
    features to the backbone).  Because the selector has no
    weights, no weight traffic is needed for tokenization — a
    key reason this chiplet can live on a small (~1 mm²) CGRA.

  BACKBONE BUILDING BLOCKS  (glossary)
  ══════════════════════════════════════════════════════════
    BasicBlock  (ResNet18 unit; torchvision):
       x ─► conv3×3 ─► BN ─► ReLU ─► conv3×3 ─► BN ─┐
        └─── 1×1 conv (only if stride≠1 or C changes) ┴─► ⊕ ─► ReLU
      Two BasicBlocks per ResNet stage (layer1–4).  The first
      block of layer2/3/4 has stride=2 (halves H,W, doubles C).
      BN is folded into the preceding conv at inference via
      FrozenBatchNorm2d (src/models/module_retr/backbone.py),
      so BN = 0 MACs on hardware.

    FPN  (torchvision BackboneWithFPN, level 0 only):
      1×1 lateral reduces C_1=64 → 64, then 3×3 "smooth" conv.
      RETR only keeps level-0 (the highest-resolution, coarsest
      feature); other pyramid levels are discarded before
      tokenization.  input_proj (1×1 conv) then lifts 64 → d=256
      per view.

    Top-K selector  (paper §4.2):
      H = {H_hor, H_ver} ∈ R^{C×P}, with P_hor = P_ver = WD/s²
      before selection.  RETR picks
         H_hor = Selector(Z_hor) ∈ R^{C×K},  K=256
      by L2-norm magnitude ranking over the spatial grid,
      shrinking the encoder input from P=(W+H)D/s² down to
      P=2K=512 tokens.  This is the "Top-K Feature Selection"
      in paper §4.2 and the dominant reason RETR keeps
      encoder complexity tractable.

    TPE  (paper §4.3, "Tunable Positional Encoding"):
      Depth-prioritized sine/cosine PE with an α-tunable split
      of the d=256 positional dim between the *depth* axis
      (αd, with α=0.6 default → 154 dims) and the *angular*
      axis (azimuth for hor view / elevation for ver view, the
      remaining 102 dims).  It is **not summed** into the token
      content; instead it is carried alongside the tokens and
      concatenated with content at every encoder/decoder layer
      (Conditional-DETR style).  See TRANSFORMER.md for details.
      Implemented by DepthPrioritizedPositionEmbeddingSine in
      src/models/module_retr/position_encoding.py.


  PIPELINE AT A GLANCE  (what this chiplet does, top-to-bottom)
  ══════════════════════════════════════════════════════════════

      2 radar heatmaps  hor + ver  [4, 256, 128]  FP16
      (pre-FFT done on-chip just before, see FFT block)

                                │
                                ▼

                ┌────────────────────────────┐
                │ ResNet18 backbone          │
                │   shared weights, run      │
                │   once per view            │
                │   → feature map            │
                │     [64, 64, 32]           │
                └────────────┬───────────────┘

                             │
                             ▼

                ┌────────────────────────────┐
                │ FPN level-0 + input_proj   │
                │   1×1 lateral + 3×3 smooth │
                │   lift 64 → 256 channels   │
                │   [256, 64, 32] per view   │
                └────────────┬───────────────┘

                             │
                             ▼

                ┌────────────────────────────┐
                │ Top-K token selection      │
                │   pick 256 strongest cells │
                │   per view by L2 norm      │
                │   (no learned weights)     │
                └────────────┬───────────────┘

                             │  2 × 256 tokens
                             ▼

                ┌────────────────────────────┐
                │ concat hor+ver, INT8-quant │
                │   emit TPE alongside       │
                │   (depth-prioritized,α=0.6)│
                └────────────┬───────────────┘

                             │
                             ▼

                   512 radar tokens  INT8
                   + TPE  →  CIM transformer

  LEGEND  (used in the detailed diagram below)
  ══════════════════════════════════════════════════════════════
    ╔═══╗        scope of one HW chiplet (CGRA / CIM / FPGA)
    ──►          DMA transfer (DRAM ↔ chiplet SRAM)
    [a,b,c]      tensor shape
    FP16 / INT8  numeric precision at that point
    BN           batch-norm (folded into conv at inference)
    FPN          feature pyramid network (we use level 0 only)
    TPE          Tunable Positional Encoding (§4.3 of the paper)


  SYMBOL TABLE
  ══════════════════════════════════════════════════════════
    B      = 1      batch
    V      = 2      views (horizontal, vertical)
    C_in   = 4      temporal frames (channel-packed)
    H_in   = 256    input height
    W_in   = 128    input width
    C_FPN  = 64     FPN level-0 output channels
    d      = 256    input_proj / token channels
    topk   = 256    tokens selected per view  (sqrt_topk = 16)
    N_e    = 512    encoder tokens  (V × topk)
  ══════════════════════════════════════════════════════════


  DRAM + UCIe (weights)               CGRA (FP16 MAC + SIMD)
  ══════════════════════              ════════════════════════════════════

                                      ┌─ INPUT ────────────────────────────┐
                                      │  radar ADC  [V,C_in,H_in,W_in]    │
                                      │    complex, pre-FFT boundary       │
                                      │  (MMVR ships post-FFT .npz; raw   │
                                      │   ADC synthetic for HW eval)       │
                                      └────────────────────────────────────┘
                                                │
                                      ╔═════════╪═════════════════════════╗
                                      ║  FFT  (per view, no weights)      ║
                                      ║  ┌─────────────────────────────┐  ║
   twiddle ROM  ~0.1–0.2 mm²          ║  │ 2D FFT [4,256,128] cplx     │  ║
   (on-chip, not DMA'd) ─────────────►║  │  → |·| → log → normalize    │  ║
                                      ║  │ output: [4,256,128] FP16    │  ║
                                      ║  └─────────────────────────────┘  ║
                                      ╚════════════════╪══════════════════╝
                                                │
                                      ╔═════════╪═════════════════════════╗
                                      ║  BACKBONE ×V views  (ResNet18)    ║
                                      ║                                   ║
   DMA conv1                          ║  conv1 7×7 s=2                    ║
     [64,4,7,7] FP16 → SRAM ────────► ║  ┌─────────────────────────────┐  ║
                                      ║  │ [4,256,128] → [64,128,64]   │  ║
                                      ║  │ 4 horiz strips (h=32)       │  ║
                                      ║  │ 128 KB in + 128 KB out tile │  ║
                                      ║  │ BN fold + ReLU + maxpool    │  ║
                                      ║  │  → [64,64,32]  (128 KB)     │  ║
                                      ║  └─────────────────────────────┘  ║
                                      ║                │                  ║
   DMA layer1                         ║  layer1 ×2 BasicBlock             ║
     4× [64,64,3,3] → SRAM ─────────► ║  │ [64,64,32] → [64,64,32]    │  ║
                                      ║                │                  ║
   DMA layer2                         ║  layer2 ×2 BasicBlock (s=2)       ║
     [64→128, 128→128]×2 + shortcut ► ║  │ → [128,32,16]  (128 KB)     │  ║
                                      ║                │                  ║
   DMA layer3                         ║  layer3 ×2 BasicBlock (s=2)       ║
     [128→256, 256→256]×2 + shortcut► ║  │ → [256,16,8]   (64 KB)      │  ║
                                      ║                │                  ║
   DMA layer4                         ║  layer4 ×2 BasicBlock (s=2)       ║
     [256→512, 512→512]×2 + shortcut► ║  │ → [512,8,4]    (32 KB)      │  ║
                                      ║                                   ║
   DMA FPN L0 + input_proj            ║  FPN lateral 1×1 + output 3×3     ║
     [64,64,1,1] + [64,64,3,3] +      ║  │  → [64,64,32]  (128 KB)     │  ║
     [256,64,1,1] → SRAM ───────────► ║  input_proj 1×1                   ║
                                      ║  │  → [256,64,32]  (512 KB)    │  ║
                                      ║  │  resident for tokenizer     │  ║
                                      ╚════════════════╪══════════════════╝
                                                │
                                      ╔═════════╪═════════════════════════╗
                                      ║  TOKENIZER  (no weights)          ║
                                      ║  ┌─────────────────────────────┐  ║
                                      ║  │ L2-norm over channels       │  ║
                                      ║  │   [256,64,32] → [2048]      │  ║
                                      ║  │ top-256 argsort  (SIMD,     │  ║
                                      ║  │   or offload 4 KB → FPGA)   │  ║
                                      ║  │ gather → [256,16,16]        │  ║
                                      ║  │ permute → [256,1,256] /view │  ║
                                      ║  │ cat(hor,ver) → [512,1,256]  │  ║
                                      ║  │ emit TPE alongside tokens   │  ║
                                      ║  │   (α=0.6 depth-prio sine/cos,│ ║
                                      ║  │    NOT summed — concatenated │ ║
                                      ║  │    per-layer in encoder)    │  ║
                                      ║  │ INT8 quantize               │  ║
                                      ║  └─────────────────────────────┘  ║
                                      ╚════════════════╪══════════════════╝
                                                │
                                      ┌─ OUTPUT ───────────────────────────┐
                                      │  radar tokens [N_e, 1, d]  INT8    │
                                      │  → CIM (transformer encoder)       │
                                      └────────────────────────────────────┘


  SRAM BUDGET (1 MB target)
  ══════════════════════════════════════════════════════════
    conv1 strip tile   128 KB in + 128 KB out   always resident
    layer2–4 fmaps     ≤128 KB                  no tiling needed
    FPN L0 + input_proj output                  512 KB (resident
                                                for tokenizer)
    Token buffer [512,1,256] INT8               128 KB
    Layer weight working set                    ≤128 KB / layer
    (backbone 27.5 MB FP16 > 1 MB → per-layer streaming;
     1–2 layers resident at a time)

  WEIGHT STREAMING BUDGET (30 fps)
  ══════════════════════════════════════════════════════════
    Backbone + FPN FP16   ~27.5 MB   (55 MB FP32 ÷ 2)
    input_proj            ~32 KB     (negligible)
    Total per inference   ~27.5 MB  →  ~825 MB/s sustained
    UCIe capacity         64–128 GB/s  (>>headroom)
    (assumes weights reused across V views per layer; doubles
     to ~1.65 GB/s if weights are restreamed per view)

  UCIe BANDWIDTH (B=1, 30 fps)
  ══════════════════════════════════════════════════════════
    DRAM→CGRA   backbone weights FP16            ~825 MB/s
    CGRA→CIM    tokens [512,1,256] INT8           ~3.8 MB/s
    CGRA↔FPGA   top-K sort offload (opt., 5 KB)  ~0.15 MB/s
    Total sustained  < 1 GB/s  (UCIe 64–128 GB/s >> headroom)

  COMPUTE (B=1, per view)
  ══════════════════════════════════════════════════════════
    conv1 7×7 s=2                                 ~103M MACs
    layer1 ×2 BasicBlock                          ~302M
    layer2 ×2 BasicBlock + shortcut               ~268M
    layer3 ×2 BasicBlock + shortcut               ~268M
    layer4 ×2 BasicBlock + shortcut               ~268M
    FPN L0 lateral 1×1 + output 3×3               ~84M
    input_proj 1×1                                ~34M
    Per view subtotal                             ~1.33G MACs
    Both views  (V=2)                             ~2.65G MACs
    FFT (radix-2 2D, per view)                    ~4M ops
    BN/ReLU/maxpool/L2-norm                       MAC-free

  SIMD / ACTIVATION BUDGET
  ══════════════════════════════════════════════════════════
    BN folded into conv weights                   0 cycles
    ReLU (fused with BN output)                   element-wise
    maxpool 3×3 s=2 on [64,128,64]                ~64K ops
    L2-norm reduce + sqrt  [256,64,32] /view      ~65K reductions
    top-256 argsort over 2048 scores /view        2048 × log2(2048)
                                                  (or 4 KB → FPGA)
    Tokenizer total                               < 0.1 ms @ 1 GHz

  CGRA AREA ESTIMATE
  ══════════════════════════════════════════════════════════
    FP16 MAC fabric  (~1 TFLOP FP16)             0.3–0.5 mm²
    FFT butterfly + twiddle ROM                   0.1–0.2 mm²
    1 MB SRAM                                     0.3 mm²
    Control + UCIe slice                          0.1–0.2 mm²
    Total target                                 ~1.0 mm²

  TOKENIZER PLACEMENT: CGRA (not FPGA)
  ══════════════════════════════════════════════════════════
    1. FPN output (512 KB) already resident in CGRA scratchpad —
       routing to FPGA adds 2× inter-chip traffic on the critical
       path (backbone→encoder handoff).
    2. FPGA bandwidth is reserved for CIM weight prefetch; sharing
       it with tokenizer risks CIM stalls.
    If CGRA lacks a sort unit: CGRA computes L2-norms, ships
    2,048 FP16 scores (4 KB) to FPGA, FPGA returns 512 indices
    (1 KB), CGRA gathers.  Total extra traffic: 5 KB.

  FFT NOTE
  ══════════════════════════════════════════════════════════
    FFT is not implemented in the PyTorch workload.  MMVR ships
    pre-computed radar heatmaps (*_radar.npz, keys hm_hori /
    hm_vert, shape [4,256,128] float32); raw ADC was never
    released.  The model's true input boundary is FFT output, so
    results are reproducible against the fixed .npz files.

    Synthetic pre-FFT data for CGRA benchmarking:
      signal  = torch.randn(4,256,128, dtype=complex64).half()
      heatmap = torch.fft.fft2(signal).abs()
      heatmap = torch.log(heatmap + 1e-10)
      heatmap = (heatmap - mean_env) / std_env

    Hardware plan: FP16 twiddles + input, output [4,256,128] FP16
    fed directly into conv1.  Twiddles live in on-chip ROM.

  RECIPE — FP32 baseline with FP16 backbone  (primary HW recipe)
  ══════════════════════════════════════════════════════════
    Backbone Conv2d in FP16 (--fp16_half_backbone).  Conv2d is
    never int8-quantized — torchao int8dq targets nn.Linear only.
    Transformer runs at its own recipe (see TRANSFORMER.md).

    cd src && CUDA_VISIBLE_DEVICES=0 python eval_ptq.py \
      --root ../MMVR/segment_4_3 --split P2S1 --task DETSEG \
      --pretrained_path ../logs/pretrained_model/p2s1_retr_detseg.pth \
      --batch_size 16 --worker 2 --device cuda \
      --scheme fp32 --use_autocast_fp16 --fp16_half_backbone
