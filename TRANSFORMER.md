<!--
Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->
# RETR — Transformer Encoder + Decoder (CIM chiplet)

  WHAT DOES THIS CHIPLET DO?  (semantic overview)
  ══════════════════════════════════════════════════════════
    RETR's transformer is a Conditional-DETR-style [Meng 2021]
    encoder-decoder specialized for multi-view radar (paper
    arXiv:2411.10293 §4.1).  It takes the 512 INT8 radar tokens
    produced by the CGRA tokenizer (see BACKBONE.md) and emits
    N=10 "object queries" that predict 3D BBoxes in the radar
    coordinate system.  The split of responsibilities is:

      ENCODER (×L_enc=6) = Cross-View Radar Feature Association
      ─────────────────────────────────────────────────────────
        Self-attends jointly over the concatenated
        [hor-tokens (256), ver-tokens (256)] sequence of length
        N_e=512.  Because both views share a depth axis, the
        encoder's job is to **let horizontal tokens and vertical
        tokens re-weight each other's features based on depth
        proximity** — i.e., a horizontal cell at depth z should
        attend strongly to the vertical cell at the same z.
        The TPE (α=0.6) encodes this bias directly into the
        attention score (see "Why 2d-wide projections" below).
        Output: encoder "memory" H^{L_enc} ∈ R^{512×d}.

      DECODER (×L_dec=6) = Object-Query ↔ Radar Binding
      ─────────────────────────────────────────────────────────
        Three sub-layers per decoder layer:
          (a) SELF-ATTN over the N=10 object queries — lets
              queries talk to each other so they don't collapse
              onto the same object (DETR's NMS-free mechanism).
          (b) CROSS-ATTN from each object query to the encoder
              memory — each query gathers the radar evidence
              relevant to one potential detection.
          (c) FFN — per-query MLP.
        After L_dec layers, a 6-dim bbox FFN head regresses
        {cx, cy, cz, w, h, d} in radar coords per query; a class
        head predicts object/no-object.

    Post-transformer (outside this chiplet):
      3D BBoxes → learnable R ∈ SO(3) radar→camera transform →
      pinhole projection → 2D image-plane BBoxes (+ optional
      segmentation head built on the vertical-view backbone
      features).  Training uses a **tri-plane loss** (L1+GIoU on
      horizontal radar plane + vertical radar plane + 2D image
      plane) with Hungarian matching.

  SEMANTIC KEY DETAILS  (glossary)
  ══════════════════════════════════════════════════════════
    TPE — Tunable Positional Encoding  (paper §4.3):
      Depth-prioritized sine/cosine PE with α∈[0,1] that splits
      the d=256 positional dim between depth (αd dims) and
      angular (1−α)d dims.  Paper default α=0.6 ⇒ 154 depth
      dims + 102 angular dims.  Applied via **concatenation**
      with content (Conditional-DETR style), *at every encoder
      and decoder layer* — not just once at the input.  The
      concat form eliminates content×pos cross-terms in the
      QᵀK inner product, leaving only c_que·c_key + p_que·p_key,
      and the TPE is tuned so p_que·p_key gives a much higher
      score when the two tokens share depth than when they
      don't.  This is the **"depth-prioritized feature
      similarity"** that is RETR's core inductive bias.
      Impl: DepthPrioritizedPositionEmbeddingSine (ratio=0.6)
      in src/models/module_retr/position_encoding.py, emitted
      by the tokenizer, consumed by `pos` argument of every
      encoder/decoder layer.

    Why 2d-wide Q/K projections?
      Because concat(content d + pos d) = 2d.  That is why the
      encoder self-attn projections and the decoder
      cross-attn Q/K projections in this file are [2d, 2d] —
      they act on the concatenated 2d-wide vectors, not on
      summed d-wide ones as in vanilla DETR.  Decoder self-attn
      stays d-wide because decoder-SA doesn't use TPE (the
      query-position embedding q_pos is summed there, DETR-style,
      not concatenated — see ConditionalTransformerDecoderLayer
      in src/models/module_retr/transformer.py).

    Object queries (tgt) and reference points:
      N=10 learnable position embeddings query_embed ∈ R^{10×d}
      (nn.Embedding).  Decoder input tgt is initialized to
      zeros (content) and updated layer-by-layer; query_embed
      is the fixed per-query positional embedding (q_pos).
      The decoder head `ref_point_head: MLP(d→3)` maps each
      query_embed to a 3D *reference point* in sigmoid space
      (cx, cy, cz); `gen_sineembed_for_3d_position` turns it
      into a sine embedding that is concatenated with q_content
      at every cross-attn layer — this is the RETR-specific
      3D-position-aware query conditioning.

    Cross-view concatenation order:
      The encoder input is src_tokens = cat([hor_topk_flatten,
      ver_topk_flatten], dim=0) ∈ R^{512×B×d}, and pos =
      cat([hor_pos, ver_pos], dim=0).  Self-attention over this
      joint sequence is what implements "cross-view feature
      association" — the horizontal and vertical streams are
      never separately attended; they fuse from encoder layer 0.

    Why 3 DMA rounds per decoder layer (instead of 2)?
      A decoder layer has 3 sub-blocks (SA, CA, FFN).  SA uses
      5 [d,d] projs (sa_qcontent/qpos/kcontent/kpos/v) + out_proj
      ≈ 6×[d,d].  CA uses 5 [d,d] projs + out_proj ≈ 6×[d,d].
      FFN = [ff,d]+[d,ff] ≈ 2×4×[d,d].  Each sub-block's working
      set is ≤1 MB, but the sum (1.83 MB) exceeds the 1 MB SRAM,
      hence three distinct DMA rounds.


  PIPELINE AT A GLANCE  (what this chiplet does, top-to-bottom)
  ══════════════════════════════════════════════════════════════

      512 radar tokens (hor+ver)          10 object queries
      from CGRA tokenizer ──────┐         (learnable, start as zeros)
                                │                 │
                                ▼                 │
                ┌────────────────────────────┐    │
                │ ENCODER ×6                 │    │
                │   self-attention across    │    │
                │   hor+ver tokens           │    │
                │   (fuses the two views     │    │
                │    via depth-similar TPE)  │    │
                └────────────┬───────────────┘    │
                             │ memory             │
                             ▼                    ▼
                        ┌────────────────────────────┐
                        │ DECODER ×6                 │
                        │   (a) queries talk to each │
                        │       other  (self-attn)   │
                        │   (b) each query pulls     │
                        │       radar evidence       │
                        │       from memory (cross)  │
                        │   (c) per-query FFN        │
                        └────────────┬───────────────┘
                                     │ 10 refined queries
                                     ▼
                        ┌────────────────────────────┐
                        │ 3D BBox head (FFN)         │
                        │   → R (radar→camera) → 2D  │
                        │     image-plane BBox       │
                        └────────────────────────────┘

  LEGEND  (used in the detailed diagram below)
  ══════════════════════════════════════════════════════════════
    SA / CA        self-attention / cross-attention
    "memory"       output of the last encoder layer (512 × 256)
    TPE            Tunable Positional Encoding (depth-prioritized,
                   concatenated with content at every layer)
    object query   a learnable slot that will either predict one
                   object or say "no object" (DETR-style)
    INT8 / FP32    precision of MAC inputs / softmax accumulators


  SYMBOL TABLE
  ══════════════════════════════════════════════════════════
    d      = 256    hidden dim
    H      = 4      attention heads (encoder & decoder)
    ff     = 2048   FFN intermediate dim
    N_e    = 512    encoder tokens  (2 × top-k=256, hor+ver)
    N_d    = 10     decoder object queries
    L_enc  = 6      encoder layers
    L_dec  = 6      decoder layers
    h_e    = 128    encoder head dim  (2d / H)
    h_d_sa = 64     decoder self-attn head dim  (d / H)
    h_d_ca = 128    decoder cross-attn Q,K head dim  (2d / H)
    h_d_v  = 64     decoder cross-attn V head dim  (d / H)
  ══════════════════════════════════════════════════════════


  FPGA (orchestration + DMA)          CIM (INT8 MAC + SIMD)
  ══════════════════════════          ════════════════════════════════════════

                                      ┌─ INPUT ────────────────────────────┐
                                      │  radar tokens  [N_e, 1, d]  INT8   │
                                      │    (from CGRA tokenizer; see       │
                                      │     BACKBONE.md)                   │
                                      │  queries tgt   [N_d, 1, d]  zeros  │
                                      └────────────────────────────────────┘
                                                │
                                      ╔═════════╪═════════════════════════╗
                                      ║  ENCODER ×L_enc                   ║
                                      ║                                   ║
   DMA enc layer weights (round 1)    ║  Phase 1A — content/pos concat +  ║
   concat-projs ×6 + q/k/v_proj:      ║               Q/K/V proj           ║
     6×[d,d] + 3×[2d,2d] → SRAM       ║  ┌─────────────────────────────┐  ║
   (Conditional-DETR style: 6 per-    ║  │ 6 pre-concat projs (d→d):   │  ║
    layer [d,d] projs transform       ║  │   qcontent / kcontent / v   │  ║
    content and TPE into the 2d-wide  ║  │   qpos_sine / kpos / vpos   │  ║
    concat input expected by Q/K/V)   ║  │ concat: [N_e,2d] per Q,K,V  │  ║
   ─────────────────────────────────► ║  │ Q/K/V proj: [N_e,2d]×3      │  ║
                                      ║  │   (weights [2d,2d])          │  ║
                                      ║  │ split:  [H,N_e,h_e] INT8    │  ║
                                      ║  └─────────────────────────────┘  ║
                                      ║                │                  ║
                                      ║  Phase 2 — Flash Attention        ║
   DMA K,V tiles from DRAM            ║  ┌─────────────────────────────┐  ║
   [H, B_c=64, h_e] INT8 each ──────► ║  │ outer: N_e/64 = 8 Q-tiles   │  ║
   (each inner iter)                  ║  │ inner: N_e/64 = 8 KV-tiles  │  ║
                                      ║  │ = 64 passes/layer           │  ║
                                      ║  │   × L_enc = 384 total       │  ║
                                      ║  │                             │  ║
                                      ║  │ per tile pass:              │  ║
                                      ║  │  QKᵀ [H,64,h_e]×[H,h_e,64]  │  ║
                                      ║  │      INT8×INT8→INT32  (MAC) │  ║
                                      ║  │  scale 1/√h_e + softmax     │  ║
                                      ║  │      online FP32  (SIMD)    │  ║
                                      ║  │  AV  [H,64,64]×[H,64,h_e]   │  ║
                                      ║  │      INT8×INT8→INT32  (MAC) │  ║
                                      ║  │  O accum FP32  (SIMD)       │  ║
                                      ║  │                             │  ║
                                      ║  │  SRAM per tile: ~290 KB     │  ║
                                      ║  └─────────────────────────────┘  ║
                                      ║                │                  ║
   DMA enc layer weights (round 2)    ║  Phase 3 — Post-attention         ║
   out_proj + FFN:                    ║  ┌─────────────────────────────┐  ║
     [2d,2d] + [ff,d] + [d,ff]        ║  │ out_proj [N_e,2d]→[N_e,d]   │  ║
   ─────────────────────────────────► ║  │ slice+residual  (SIMD)      │  ║
                                      ║  │ LayerNorm1      (SIMD)      │  ║
                                      ║  │ linear1 [N_e,d]→[N_e,ff]    │  ║
                                      ║  │ ReLU            (SIMD)      │  ║
                                      ║  │ linear2 [N_e,ff]→[N_e,d]    │  ║
                                      ║  │ residual+LN2    (SIMD)      │  ║
                                      ║  └─────────────────────────────┘  ║
                                      ║                                   ║
   prefetch next enc layer weights    ║  output: memory [N_e, 1, d]       ║
   while CIM runs Phase 2 ──────────► ║  (cached; reused ×L_dec)          ║
                                      ╚═══════════════════════════════════╝
                                                │ memory [N_e,1,d]
                                      ╔═════════╪══════════════════════════╗
                                      ║  DECODER ×L_dec                   ║
                                      ║                           tgt      ║
                                      ║                       [N_d,1,d]   ║
                                      ║                           │        ║
                                      ║  Self-Attention (SA)      │        ║
   DMA dec layer weights (round 1)    ║  ┌─────────────────────── ▼ ────┐  ║
   SA projections ×5 + out_proj:      ║  │ 5 proj: [N_d,d]→[N_d,d]×5   │  ║
     6×[d,d] → SRAM                   ║  │ Q=qc+qp, K=kc+kp            │  ║
   ─────────────────────────────────► ║  │ split: [H,N_d,h_d_sa] INT8   │  ║
                                      ║  │ QKᵀ  [H,N_d,N_d]  (MAC)     │  ║
                                      ║  │ softmax            (SIMD)    │  ║
                                      ║  │ AV   [H,N_d,h_d_sa](MAC)    │  ║
                                      ║  │ out_proj+residual+LN1(SIMD) │  ║
                                      ║  └─────────────────────────────┘  ║
                                      ║                │                  ║
                                      ║  Cross-Attention (CA)             ║
   DMA dec layer weights (round 2)    ║  ┌─────────────────────────────┐  ║
   CA projections ×5 + out_proj:      ║  │ Q: [N_d,d]→cat→[N_d,2d]    │  ║
     6×[d,d] → SRAM                   ║  │ K: [N_e,d]→cat→[N_e,2d]    │  ║
   ─────────────────────────────────► ║  │ V: [N_e,d]                  │  ║
   K,V SRAM-cached across ×L_dec      ║  │  (K,V from cached memory;   │  ║
   no re-DMA per layer ─────────────► ║  │   no re-DMA per dec layer)  │  ║
                                      ║  │ split:                       │  ║
                                      ║  │  Q [H,N_d,h_d_ca]  INT8     │  ║
                                      ║  │  K [H,N_e,h_d_ca]  INT8     │  ║
                                      ║  │  V [H,N_e,h_d_v]   INT8     │  ║
                                      ║  │ QKᵀ [H,N_d,N_e]    (MAC)   │  ║
                                      ║  │ softmax             (SIMD)  │  ║
                                      ║  │ AV  [H,N_d,h_d_v]  (MAC)   │  ║
                                      ║  │ out_proj+residual+LN2(SIMD)│  ║
                                      ║  └─────────────────────────────┘  ║
                                      ║                │                  ║
   DMA dec layer weights (round 3)    ║  FFN                              ║
   FFN: [ff,d] + [d,ff] → SRAM        ║  ┌─────────────────────────────┐  ║
   ─────────────────────────────────► ║  │ linear1 [N_d,d]→[N_d,ff]   │  ║
                                      ║  │ ReLU            (SIMD)      │  ║
                                      ║  │ linear2 [N_d,ff]→[N_d,d]   │  ║
                                      ║  │ residual+LN3    (SIMD)      │  ║
                                      ║  └─────────────────────────────┘  ║
                                      ║                                   ║
   prefetch next dec layer weights    ║  tgt out: [N_d, 1, d]             ║
   while CIM computes ──────────────► ║  (fed back as tgt next layer)     ║
                                      ╚═══════════════════════════════════╝
                                                │
                                      ┌─ OUTPUT ───────────────────────────┐
                                      │  tgt  [N_d, 1, d]                  │
                                      │  → detection / segmentation heads  │
                                      └────────────────────────────────────┘


  SRAM BUDGET (1 MB target)
  ══════════════════════════════════════════════════════════
    Flash attn tile (enc Phase 2)      ~290 KB   always resident
    Enc layer weights  2.49 MB > 1 MB  → 2 DMA rounds  (Ph1A, Ph3)
    Dec layer weights  1.83 MB > 1 MB  → 3 DMA rounds  (SA, CA, FFN)
    Dec CA K,V buffers [H,N_e,h_d_ca]  256 KB
                       [H,N_e,h_d_v]   128 KB   cached across L_dec

  WEIGHT STREAMING BUDGET (30 fps)
  ══════════════════════════════════════════════════════════
    Enc  2.49 MB × L_enc = 14.94 MB INT8
    Dec  1.83 MB × L_dec = 10.98 MB INT8
    Total per inference   ~26 MB  →  ~780 MB/s sustained
    UCIe capacity         64–128 GB/s  (>>headroom)

  UCIe BANDWIDTH (B=1, 30 fps)
  ══════════════════════════════════════════════════════════
    CGRA→CIM  backbone features INT8  2 views×[1,256,64,32]  ~32 MB/s
    DRAM→CIM  transformer weights                            ~780 MB/s
    Total sustained  < 1 GB/s  (UCIe 64–128 GB/s >> headroom)

  COMPUTE (B=1)
  ══════════════════════════════════════════════════════════
    Enc concat-projs ×6×6    6×6×(512×256²)       = 1,207M MACs
      (qcontent/kcontent/v/qpos_sine/kpos/vpos [d,d] per layer)
    Enc Q/K/V proj ×6        6×3×(512×512²)       = 2,415M MACs
      (acting on the content⊕TPE concat, [2d,2d] each)
    Enc attn BMM QKᵀ+AV ×6  6×2×(4×512×128×512)  = 1,610M MACs
    Enc out_proj + FFN ×6    6×(512×512²+2×512×256×2048) = 4,026M MACs
    Encoder total                                  ≈ 9,258M ≈ 9.3G
    Decoder SA/CA/FFN ×6                           ≈   735M
    Transformer total                              ≈ 9,993M ≈ 10G MACs
    BMMs = 16% of total;  linear projections = 84%

  SIMD BUDGET (16 lanes, 1 GHz)
  ══════════════════════════════════════════════════════════
    Enc softmax [4,512,512] ×6   12,288 rows × ⌈512/16⌉ × 3 passes = 1,179,648 cy  (84%)
    Enc LayerNorm ×2 ×6           6,144 rows × 16 × 2 passes        =   196,608 cy
    Dec CA softmax [4,10,512] ×6    240 rows × 32 × 3 passes         =    23,040 cy
    Dec LN ×3 ×6                  negligible                         =    ~6,000 cy
    Total  ~1,405,296 cycles  ≈  1.41 ms @ 1 GHz
    Bottleneck: encoder softmax (512-wide rows × 12,288 = 84% of SIMD time)

  FLASH ATTN ACCUMULATOR NOTE
  ══════════════════════════════════════════════════════════
    O, m, l held FP32 — online rescale e^(m_old−m_new)×O loses all precision
    in FP16; FP32 accumulators are the published norm (INT-FlashAttention,
    FlashAttention-3, SageAttention).
    Alt: integer-only softmax (I-BERT / ITA) — 2nd-order poly + power-of-2
    shift on INT32 logits, keeps m/l in INT32, no FP32 datapath; requires
    polynomial/LUT unit added to SIMD coprocessor.

  B > 1 SCALING
  ══════════════════════════════════════════════════════════
    Enc token buffer   256B KB    exhausts 1 MB SRAM at B=4
    Flash attn tile    ~290 KB    unchanged (per-head, per-sample); passes × B
    Dec CA map         80B KB     fits SRAM up to B≈12
    Total MACs / SIMD  10B G / 1.41B ms  (linear with B)
    B=2 feasible (~802 KB total); B≥4 needs ≥2 MB SRAM or FP4 tokens (64 KB/sample)

  CIM AREA ESTIMATE
  ══════════════════════════════════════════════════════════
    INT8 MAC array (~1024 MACs, ~2 TOPS @ 1 GHz)   0.4–0.6 mm²
    SIMD coprocessor (softmax, LN, ReLU), 16 lanes   ~0.07 mm²
    MX dequant + activation quantizers ×3           0.07–0.12 mm²
    1 MB SRAM                                       0.3–0.4 mm²
    Control + scale broadcast + UCIe slice          0.1–0.2 mm²
    Total target                                    ~1.0–1.4 mm²

  RECIPE 1 — INT8dq W8A8 + INT8 attention  (primary HW recipe)
  ══════════════════════════════════════════════════════════
    All 161 nn.Linear + attention BMMs: INT8×INT8→INT32  (torchao oneDNN)
    Softmax, LN, ReLU: FP16 on SIMD coprocessor
    Verified P2S1, n=7,942:  BBox AP 0.5011 (+0.0047 vs FP32 0.4964)
                             Seg IoU 0.7539
                             Model  59.62 MB  (2.62× vs 156.1 MB FP32)

    cd src && CUDA_VISIBLE_DEVICES=0 python eval_ptq.py \
      --root ../MMVR/segment_4_3 --split P2S1 --task DETSEG \
      --pretrained_path ../logs/pretrained_model/p2s1_retr_detseg.pth \
      --batch_size 16 --worker 2 --device cuda \
      --scheme int8dq --component transformer \
      --use_autocast_fp16 --fp16_half_backbone \
      --fp16_ln all --fp16_softmax all --ln_input_bits 4 --attn_int8

