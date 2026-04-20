"""DATAFLOW.py — functional simulator of the RETR P2S1 ASIC pipeline.

End-to-end inference.  Executable; canonical recipe:

    python DATAFLOW.py --lin-scale-fmt e8m0 --attn-scale-fmt blockk128

Numerically equivalent to eval_ptq.py run with:

    --scheme int8dq --component transformer --attn_int8
    --use_autocast_fp16 --fp16_half_backbone
    --fp16_ln all --fp16_softmax all --ln_input_bits 4

╔══════════════════════════════════════════════════════════════════════════╗
║ Dataflow typing discipline                                               ║
╠══════════════════════════════════════════════════════════════════════════╣
║ All activations flow in **FP16** from radar input to box outputs.        ║
║ Linear-layer activations carry an E8M0 (power-of-2) scale — one INT8    ║
║ exponent per token; no FP16 reciprocal at the QNT unit (LZC + ceil).    ║
║ Attention BMMs use block-K128 tiling: Q and K/V are E8M0-quantized per  ║
║ 128-wide K-slice; online softmax accumulates running (max, sum) across   ║
║ slices on the FP SIMD unit — avoiding [512,512] score materialization.   ║
║ The ONLY FP32 path is the online-softmax row-sum accumulator on SIMD.    ║
║ INT8 MAC accumulators are INT32; DQT narrows to FP16 per tile.           ║
╚══════════════════════════════════════════════════════════════════════════╝

Hardware map  (canonical: --lin-scale-fmt e8m0  --attn-scale-fmt blockk128)
    CGRA — FP16 SIMD engine: backbone (ResNet18+FPN), input_proj 1×1 conv,
           class/bbox heads, box_affine_transformer, box geometry (sigmoid,
           cxcywh↔xyxy), sine/cos positional embeddings, top-K selection.
    CIM  — 8 MB on-chip SRAM per chiplet.  Units on each chiplet:
           · INT8 MAC array (W8A8 systolic): every transformer Linear and
             every attention BMM tile.  INT32 accumulators, DQT per tile.
           · FP SIMD unit: online softmax (running max+sum over block-K128
             K-slices), LN (after MXQ), residual adds, bias, ReLU, sigmoid.
           · Quant/Dequant units: E8M0 QNT (log2+ceil → INT8 exponent) for
             linear activations; per-tile E8M0 QNT for attention K-slices;
             DQT = exponent shift (+ FP16 weight-scale mul for linears only).

Unit tags used in code comments:
    QNT : FP16 → INT8 E8M0 (scale = 2^⌈log₂(absmax/127)⌉; LZC+ceil, no FP recip)
    MAC : INT8 × INT8 → INT32 systolic matmul
    DQT : linear: INT32 × 2^e_a × FP16 w_scale → FP16 (exponent shift + 1 FP mul)
          attn  : INT32 × 2^(e_a+e_b) → FP16 (INT8 exponent-add + shift; no FP mul)
    BLKK: block-K128 tiled BMM — per-slice QNT+MAC+DQT, FP16 partial accumulate
    MXQ : FP16 → FP4 E2M1 grid on blocks of 32 (fake-quant; LN input path)
    LN  : F.layer_norm with FP16 affine (preceded by MXQ; runs on FP SIMD)
    SFM : online FP16 softmax across block-K128 K-slices; FP32 running row-sum
    FPU : FP16 elementwise unit (adds, muls, ReLU, sigmoid, sin/cos)

Every Linear (weight-container: Int8DynLinear) and every attention BMM is
expanded INLINE in main() as the four-stage HW sequence:
    [producer]→QNT  →  MAC  →  DQT  →  FPU (bias add / residual / relu)
QNT is FUSED WITH THE PRODUCING UNIT (LN, softmax, BMM-out, ReLU, concat,
sineembed) so the activation is stored as (INT8 + INT8 E8M0 exponent) on the
long-haul edge between producer and CIM consumer — 9 bits/token overhead vs
16 bits for FP16 scale, halving inter-unit bandwidth.  The FP16 tensor at
the producer remains local/transient for any FP16 consumer (residual add).  Activations that are
constant across layers (pos_embed across all 6 enc + 6 dec layers,
query_embed across all 6 dec layers, memory across all 6 dec layers) are
QNT'd ONCE outside the loop and reused — replacing tens of redundant QNTs.
Every LayerNorm (weight-container: FP16BlockLN) is expanded inline as:
    MXQ(input, block=32) →  F.layer_norm(FP16 weight+bias) → FP16 → QNT.

╔══════════════════════════════════════════════════════════════════════════╗
║ Implied hardware requirements (surveyed from every call in main())       ║
╠══════════════════════════════════════════════════════════════════════════╣
║ QNT (FP16 → INT8 E8M0, per-token or per-col — canonical)                 ║
║   • absmax reduce tree along the quantized axis                          ║
║       widths needed: 64, 128, 256, 512, 2048 lanes                       ║
║       tree depth  ⌈log₂ D⌉  (max 11 levels at D=2048)                    ║
║   • E8M0 scale: leading-zero-count (LZC) + round-up on absmax/127 →     ║
║       INT8 exponent; no FP16 reciprocal unit required                    ║
║   • D-lane INT8 right-shift for x >> e (replaces FP16 multiplier row)   ║
║   • 2 passes per row (absmax then shift-quantize) → row shadow buffer    ║
║       max row buffer = 4 KB  (D=2048, FFN hidden)                        ║
║   • per-row exponent SRAM (INT8, 1 B/row) carried into MAC/DQT:          ║
║       max = 512 B  (N=512 encoder tokens) — half the FP16 scale budget   ║
║   • QNT-at-production: ~12 QNTs/encoder layer, ~14 QNTs/decoder layer    ║
║     plus ~3 one-shot QNTs amortized across all layers (pos, query, mem)  ║
║                                                                          ║
║ MAC  (INT8 × INT8 → INT32 systolic matmul)                               ║
║   • 8 × 8 → 32 bit multiply-accumulate cells                             ║
║   • minimum tile ≥ 32 × 32 (torch._int_mm pads M,N to 32 and K to %8)    ║
║   • largest MAC calls per forward:                                       ║
║        encoder SA scores  [4,512,128]·[4,128,512] → 134 M MACs / layer   ║
║        encoder SA context [4,512,512]·[4,512,128] → 134 M MACs / layer   ║
║        encoder FFN1       [512B,256]·[256,2048]   → 268 M MACs / layer   ║
║        encoder FFN2       [512B,2048]·[2048,256]  → 268 M MACs / layer   ║
║        decoder CA scores  [4,10,128]·[4,128,512]  →   2.6 M MACs / layer ║
║   • per-layer (encoder) total ≈ 1.1 G MACs INT8                          ║
║   • whole forward (6 enc + 6 dec + heads) ≈ 7 G MACs INT8                ║
║                                                                          ║
║ DQT  (INT32 → FP16 at the MAC tile output — two variants in canonical)   ║
║   Linear (dqt_e8m0_lin): INT32 × 2^e_a × FP16 w_scale → FP16           ║
║   • INT32 → FP32 widen; shift exponent field by e_a (pure bit-op);      ║
║     one FP16×FP16 multiply for weight scale; narrow to FP16.            ║
║   • One INT8 exponent + one FP multiplier per output row.                ║
║   Attention (dqt_e8m0_attn, per block-K128 tile): INT32 × 2^(e_a+e_b)  ║
║   • e_a + e_b in INT8 (one INT8 add per output tile); shift into FP32   ║
║     exponent field; narrow to FP16.  No FP multiplier at all.           ║
║   • FP16 partial sums accumulated across K-slices on the SIMD unit.     ║
║                                                                          ║
║ MXQ  (FP16 → FP4 E2M1 on blocks of 32, power-of-2 scale)                 ║
║   • 32-lane absmax per block                                             ║
║   • log₂ + ⌈·⌉ table (LUT, not dataflow tensor)                          ║
║   • 7-boundary bucketize + sign · posval · scale multiply                ║
║   • used only on the LN input path (1 MXQ per LayerNorm call)            ║
║                                                                          ║
║ BLKK (block-K128 tiled BMM — block_k_bmm_e8m0)                           ║
║   Applied to all attention score BMMs (Q·Kᵀ) and context BMMs (A·V).    ║
║   K split into 128-element slices; for each slice:                       ║
║     1. E8M0 QNT on rows (a) and cols (b) → INT8 + INT8 exponent pair    ║
║     2. INT8 MAC → INT32 tile accumulator on the MAC array                ║
║     3. Tile DQT: 2^(e_a+e_b) shift → FP16 partial                       ║
║     4. FP16 partial accumulated into output tensor on the SIMD unit      ║
║   On hardware: softmax is fused into the K-slice loop (online algorithm) ║
║   eliminating the full [B·H, N, N] score matrix from SRAM.              ║
║   Peak in-flight tile per BMM: [4, 512, 128] FP16 = 512 KB.             ║
║                                                                          ║
║ SFM  (online FP16 softmax across block-K128 K-slices — on FP SIMD unit)  ║
║   Canonical: encoder SA rows K=512 → 4 slices of K=128; dec SA K=10.    ║
║   Online algorithm per row (fused with BLKK K-slice loop on hardware):  ║
║     m_new = max(m_old, max(scores_slice))         FP16 running max       ║
║     correction = exp(m_old − m_new)               FP16 SIMD             ║
║     s_new = correction·s_old + Σexp(x − m_new)    FP32 accumulator      ║
║     out_partial += correction·out_partial + exp(x−m_new)·V_slice        ║
║   Final: out = out_partial / s_new → FP16                                ║
║   FP32 accumulator for s (row-sum) — the ONLY FP32 path                  ║
║   Running state per head-row: m (FP16, 2 B) + s (FP32, 4 B) = 6 B/row   ║
║     encoder: 4 heads × 512 rows × 6 B = 12 KB on SIMD (negligible)     ║
║                                                                          ║
║ LN   (F.layer_norm on MXQ'd input, FP16 affine)                          ║
║   • mean / variance reduce across D=256                                  ║
║   • 1 reciprocal-sqrt unit                                               ║
║   • per-dim FP16 affine multiply-add                                     ║
║                                                                          ║
║ FPU  (FP16 vector elementwise)                                           ║
║   • add/sub/mul/fma, ReLU, sigmoid, sin/cos, reciprocal, abs, exp        ║
║   • vector width ≥ D=256 lanes (feeds residual + LN + ffn adds)          ║
║                                                                          ║
║ CGRA (FP16 general-purpose engine — backbone, heads, geometry, top-K)    ║
║   • Conv2d 7×7/s2 + BN + ReLU, 4× residual stages (ResNet18)             ║
║   • 1×1 Conv2d projection  (64→256)                                      ║
║   • L2-norm reduce + top-K selection over H·W tokens                     ║
║   • 2-D and 3-D sine positional embeddings (sin / cos LUTs)              ║
║   • class_embed Linear, bbox_embed 3-layer MLP, box_affine_transformer   ║
║   • all tensors flow in FP16                                             ║
║                                                                          ║
║ Data footprint  (per sample, B=1, V=2 views, in=(4,256,128) per view)    ║
║   Per-stage weight & peak activation tiles across the CGRA→CIM pipeline. ║
║                                                                          ║
║ ── Stage 1 — FFT  (CGRA; parameter-free, per view) ───────────────────   ║
║   weights   twiddle ROM (on-chip, not DMA'd)            ≈   1 KB         ║
║   act i/o   [4, 256, 128] FP16                         256 KB / view     ║
║                                                                          ║
║ ── Stage 2 — Backbone CNN  (CGRA FP16; shared across V=2) ────────────   ║
║   weights (ResNet18 + FPN-L0 + input_proj×2, FP16)                       ║
║     conv1  [64,4,7,7]                                       25 KB        ║
║     layer1 2×BasicBlock(64→64)                             288 KB        ║
║     layer2 2×BasicBlock(64→128, s=2)                      1.00 MB        ║
║     layer3 2×BasicBlock(128→256, s=2)                     4.00 MB        ║
║     layer4 2×BasicBlock(256→512, s=2)                    16.00 MB        ║
║     FPN 4×lateral 1×1 + 4×smooth 3×3                       408 KB        ║
║     input_proj ×2  (64→256, 1×1)                            64 KB        ║
║     TOTAL backbone + FPN + input_proj               ≈  21.8 MB FP16      ║
║   peak activation tile (per view, FP16)                                  ║
║     input_proj output   [256, 64, 32]            1,024 KB (resident)     ║
║     2-D sine pos_embed  [256, 64, 32]            1,024 KB / view         ║
║                                                                          ║
║ ── Stage 3 — Tokenizer  (CGRA; parameter-free) ───────────────────────   ║
║   weights   none  (L2-norm + top-K selector + QNT)                       ║
║   act       L2 scores [2048] FP16  4 KB;  top-K idx  ≤ 0.5 KB            ║
║             gathered feats [256, 16, 16] FP16   128 KB / view            ║
║             concat src, pos [512, B, 256] FP16  256 KB each (transient)  ║
║             after QNT → INT8 [512, 256] 128 KB + FP16 scale 1 KB  (×2)   ║
║                                                                          ║
║ ── Stage 4 — Encoder ×6  (CIM INT8 W8A8; 1 layer resident) ───────────   ║
║   weights per layer  (INT8 w + FP16 w_scale + FP16 bias + LN affine)     ║
║     6× content/pos proj  (256→256)                       ≈  393 KB       ║
║     SA Q/K/V proj ×3     (512→512)                       ≈  786 KB       ║
║     SA out_proj          (512→512)                       ≈  262 KB       ║
║     FFN linear1/2        (256↔2048)                      ≈ 1024 KB       ║
║     scales + biases + LN affines                         ≈   20 KB       ║
║     per layer                                            ≈ 2.43 MB       ║
║     × 6 layers                                           ≈ 14.6 MB       ║
║   peak transient activation tiles  (block-K128 online-SFM fusion)        ║
║     FFN hidden       [512, 2048]    FP16    2,048 KB  ← global peak      ║
║     Q/K/V 2d-wide    [512, B, 512]  FP16×3  1,536 KB                     ║
║     SA context       [4, 512, 128]  FP16      512 KB                     ║
║     SA score tile    [4, 512, 128]  FP16      512 KB  (block-K128 slice) ║
║     online SFM state [4, 512, 2]   FP16/FP32   12 KB  (m+s per head-row)║
║     src residual     [512, B, 256]  FP16      256 KB                     ║
║     (full [4,512,512] score matrix never materialised in SRAM)           ║
║                                                                          ║
║ ── Stage 5 — Decoder ×6  (CIM INT8 W8A8; 1 layer resident) ───────────   ║
║   weights per layer                                                      ║
║     SA projs ×5 + out_proj   (256→256 ×6)               ≈  393 KB        ║
║     CA projs ×5 + out_proj   (256→256 ×6)               ≈  393 KB        ║
║     FFN linear1/2            (256↔2048)                 ≈ 1024 KB        ║
║     norms + scales + biases + amortized ref_point_head/query_scale       ║
║     per layer                                           ≈ 1.88 MB        ║
║     × 6 layers                                          ≈ 11.3 MB        ║
║   peak transient activation tiles                                        ║
║     CA K             [4, 512, 128]  FP16      512 KB  ← decoder peak     ║
║     memory (cached)  [512, 256]     INT8      128 KB  (reused ×6 dec)    ║
║     FFN hidden       [10, 2048]     FP16       40 KB                     ║
║     CA scores        [4, 10, 512]   FP16       20 KB                     ║
║                                                                          ║
║ ── Heads + box_affine  (CGRA FP16; post-decoder) ─────────────────────   ║
║   class_embed + bbox_embed MLP + box_affine_transformer  ≈ 0.3 MB FP16   ║
║                                                                          ║
║ ── Totals ───────────────────────────────────────────────────────────    ║
║   backbone + FPN + input_proj   (CGRA, FP16)          ≈ 21.8 MB          ║
║   6 encoder layers              (CIM,  INT8)          ≈ 14.6 MB          ║
║   6 decoder layers              (CIM,  INT8)          ≈ 11.3 MB          ║
║   heads + box_affine            (CGRA, FP16)          ≈  0.3 MB          ║
║   TOTAL weights                                       ≈ 48   MB          ║
║                                                                          ║
║   SRAM sizing guidance  (8 MB per CIM chiplet — canonical)               ║
║     weight SRAM    ≈ 2.5 MB  one encoder layer (heaviest) resident       ║
║     activation     ≈ 2.0 MB  FFN hidden [512,2048] FP16 (global peak)    ║
║     score tile     ≈ 0.5 MB  SA score K-slice [4,512,128] FP16           ║
║     online SFM     ≈  12 KB  running (m,s) per head-row on SIMD          ║
║     exponent SRAM  ≈   0.5 KB  INT8 E8M0 exponents (N=512 rows)          ║
║     total peak     ≈ 5.1 MB  → 8 MB chiplet has ~2.9 MB headroom         ║
║     (full [4,512,512] score matrix is never in SRAM with online SFM)     ║
║   (MAC accumulators are per-cell registers, not SRAM — INT32 covers all  ║
║    cases including FFN K=2048 which peaks at ~26 bits accumulation.)     ║
║                                                                          ║
║ Shape constraints enforced by int8_mac()                                 ║
║   • M ≥ 32, N ≥ 32, K ≡ 0 (mod 8)                                        ║
║     zero-pads tails when needed; a real systolic array maps this to      ║
║     its native tile size (≥ 32) and K-lane multiple.                     ║
║                                                                          ║
║ Only one FP32 datapath: the SFM row-sum accumulator.                     ║
║ DQT's INT32·FP16·FP16 widen-to-FP32-and-narrow is unit-internal and      ║
║ does not appear on the dataflow-level budget.                            ║
╚══════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════╗
║ QNT / DQT recipe catalogue  (exact algorithms used in main())           ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║ ── QNT: activation quantization (canonical: qnt_act_e8m0) ────────────  ║
║   E8M0  e     = ⌈log₂(absmax(x, dim) / 127)⌉  → INT8 exponent          ║
║         scale = 2^e  (hardware: LZC + round-up; no FP reciprocal)       ║
║         x_i8  = round(clip(x / scale, −127, 127))  → INT8               ║
║   FP16 fallback (qnt_act):                                               ║
║         scale = absmax(x, dim) / 127  (FP16, requires FP reciprocal)    ║
║         x_i8  = round(clip(x / scale, −127, 127))  → INT8               ║
║   Axis conventions                                                       ║
║     per-token  (dim=−1): applied to Q, softmax-attn, LN outputs,        ║
║                           ReLU, concat, BMM-out, boundary inputs.        ║
║                           exponent/scale shape [N, 1] or [B,H,T,1].    ║
║     per-col    (dim=−2): applied to K^T and V (transposed for BMMs).    ║
║                           exponent/scale shape [B,H,1,T].                ║
║   Weight QNT (static, at model-load time, in Int8DynLinear.from_fp):    ║
║     scale = absmax(W, dim=1, keepdim) / 127   per-output-channel FP16   ║
║     w_i8  = round(clip(W / scale, −127, 127)) stored INT8 [out, in]     ║
║                                                                          ║
║ ── DQT: dequantize at MAC output (canonical: two variants) ───────────  ║
║   Linear (dqt_e8m0_lin):                                                ║
║     out = (acc_i32.float() × 2^e_a × w_scale_fp16).to(FP16)            ║
║     e_a is INT8 → shift in FP32 exponent field; one FP16×FP16 mul for  ║
║     weight scale.  Scale pairing:                                        ║
║       a_exp (INT8) [N,1] per-token × w_scale FP16 [1,out] per-out-ch    ║
║   Attention (dqt_e8m0_attn, per block-K128 tile):                       ║
║     out = (acc_i32.float() × 2^(e_a+e_b)).to(FP16)                     ║
║     one INT8 add (e_a+e_b), one exponent-field shift — no FP mul.       ║
║     Scale pairing (per K-slice tile):                                    ║
║       QK: e_q [B,H,T,1] per-token × e_kT [B,H,1,K128] per-col          ║
║       AV: e_a [B,H,T,1] per-token × e_v  [B,H,1,D]    per-col          ║
║   Bias add and residual happen in FP16 SIMD immediately after DQT.      ║
║                                                                          ║
║ ── MXQ: FP4 E2M1 block quantization (mx_fp4) ─────────────────────────  ║
║   Applied exclusively on the LN input path (one call per LayerNorm).    ║
║   Block size = 32 elements.                                              ║
║   Scale  = 2^⌈log₂(absmax(block))⌉  (power-of-two, per block)           ║
║   Grid   = {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}  (E2M1, 7 boundaries)  ║
║   Output is FP16 (fake-quant); F.layer_norm reads the rounded values.   ║
║   LN affine weights and biases remain FP16.                              ║
║                                                                          ║
║ ── Amortised QNTs (quantized once, reused across many layers) ─────────  ║
║   pos_embed   : QNT once before enc loop → reused by all 6 enc layers   ║
║                 (content_proj/k/v/q + pos_proj/k/v/q calls) and all     ║
║                  6 dec CA layers (ca_kpos_proj).  12+ reuses total.      ║
║   query_embed : QNT once before dec loop → reused by dec SA pos-proj     ║
║                 (sa_qpos, sa_kpos every layer) and ca_qpos in layer 0   ║
║                 plus both ref_point_head linear calls.  13+ reuses.     ║
║   memory      : INT8 from enc LN2 layer-6 → reused by dec CA content-   ║
║                 proj (ca_kcontent, ca_v) all 6 dec layers. 12 reuses.   ║
║                 No re-QNT; the scale s_memory is carried alongside.     ║
║                                                                          ║
║ ── QNT-at-production policy ───────────────────────────────────────────  ║
║   Every activation consumed by a CIM MAC is quantized at the producing  ║
║   unit, not at the CIM input.  The long-haul edge carries               ║
║   (INT8 tensor, INT8 E8M0 exponent) — 9 bits/token vs 16 for FP16.      ║
║   The FP16 tensor is local/transient (residual buffer) only.             ║
║   Producing units and their QNT call sites:                              ║
║     Encoder boundary (CGRA→CIM) : src concat, pos_embed                 ║
║     Encoder LN1 output           : QNT → FFN linear1                    ║
║     Encoder ReLU output          : QNT → FFN linear2                    ║
║     Encoder LN2 output           : QNT → next layer content/pos projs   ║
║     Encoder SA concat (2D wide)  : QNT q_cat, k_cat, v_cat separately   ║
║     Encoder SA Q/kT/V heads      : QNT per-token / per-col              ║
║     Encoder SA softmax output    : QNT → AV BMM                         ║
║     Encoder SA BMM-out           : QNT → out_proj                       ║
║     Decoder SA Q/kT/V heads      : QNT per-token / per-col              ║
║     Decoder SA softmax output    : QNT → AV BMM                         ║
║     Decoder SA BMM-out           : QNT → sa_out_proj                    ║
║     Decoder LN1 output           : QNT → ca_qcontent                    ║
║     Decoder CA Q/kT/V heads      : QNT per-token / per-col              ║
║     Decoder CA softmax output    : QNT → AV BMM                         ║
║     Decoder CA BMM-out           : QNT → ca_out_proj                    ║
║     Decoder LN2 output           : QNT → FFN linear1                    ║
║     Decoder ReLU output          : QNT → FFN linear2                    ║
║     Decoder LN3 output           : QNT → next iter SA / query_scale     ║
║     Decoder sineembed output     : QNT → ca_qpos_sine                   ║
║     Decoder ref_point_head ReLU  : QNT → rp_head linear2                ║
╚══════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════╗
║ BACKBONE + TOKENIZER  (CGRA chiplet, FP16 native)                       ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  Input: hor/ver  FP16 [B, 4, 256, 128]  (post-FFT heatmaps, 2 views)   ║
║                                                                          ║
║  Stage 1 — FFT  (parameter-free; twiddle ROM ≈ 0.2 KB on-chip)          ║
║    [pre-computed upstream; hor/ver arrive as FP16 heatmaps]              ║
║                                                                          ║
║  Stage 2 — ResNet18 stem + stages  (BN folded; 0 extra MACs)            ║
║    conv1  7×7 s=2, 4→64 ch + BN-fold + ReLU                             ║
║      FP16 [B,4,256,128] → [B,64,128,64]                                 ║
║      CGRA tiling: 4 horizontal strips h=32 (128 KB in + 128 KB out)     ║
║    maxpool 3×3 s=2  →  FP16 [B,64,64,32]                                ║
║    layer1  2×BasicBlock(64→64,  s=1)  → FP16 [B, 64, 64, 32]   c2      ║
║    layer2  2×BasicBlock(64→128, s=2)  → FP16 [B,128, 32, 16]   c3      ║
║    layer3  2×BasicBlock(128→256,s=2)  → FP16 [B,256, 16,  8]   c4      ║
║    layer4  2×BasicBlock(256→512,s=2)  → FP16 [B,512,  8,  4]   c5      ║
║                                                                          ║
║  Stage 2b — FPN top-down  (lateral 1×1 + nearest-upsample + 3×3)       ║
║    inner_blocks[3](c5) → p5  FP16 [B, 64,  8,  4]                       ║
║    inner_blocks[2](c4) + up(p5) → p4  FP16 [B, 64, 16,  8]             ║
║    inner_blocks[1](c3) + up(p4) → p3  FP16 [B, 64, 32, 16]             ║
║    inner_blocks[0](c2) + up(p3) → p2  FP16 [B, 64, 64, 32]             ║
║    layer_blocks[0](p2) → f0  FP16 [B, 64, 64, 32]   ← det head + seg  ║
║    layer_blocks[1](p3) → f1  FP16 [B, 64, 32, 16]   ← seg-L1           ║
║    layer_blocks[2](p4) → f2  FP16 [B, 64, 16,  8]   ← seg-L2           ║
║                                                                          ║
║  Stage 3a — input_proj  1×1 Conv2d 64→256  (CGRA FP16)                  ║
║    f0 → FP16 [B, 256, 64, 32]  (two projections: hor + ver)             ║
║                                                                          ║
║  Stage 3b — Tokenizer  (CGRA, parameter-free)                           ║
║    L2-norm over 256 channels  → score map FP16 [B, 64, 32]  (2048/view) ║
║    top-256 spatial locs by L2 magnitude → idx FP16 [B, 256]             ║
║    gather feat + PE at top-K indices    → FP16 [B, 256, 16, 16]         ║
║    CGRA: sort 2048 scores or offload 4 KB to FPGA for top-K             ║
║                                                                          ║
║  CGRA→CIM boundary  (E8M0 QNT applied here, not inside backbone)        ║
║    concat [hor‖ver] tokens → src FP16 [512, B, 256]                     ║
║    QNT(src)       → INT8 [512·B, 256] + e_src INT8 [512·B, 1]           ║
║    QNT(pos_embed) → INT8 [512·B, 256] + e_pos INT8 [512·B, 1]           ║
║    (pos_embed constant → one QNT amortised across all 6 enc + 6 dec)    ║
╚══════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════╗
║ TRANSFORMER  (CIM chiplet, 8 MB SRAM, INT8 W8A8 MAC + FP SIMD)         ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  Flash Attention 2-level tiling  (flash_attn_e8m0_tiled)                ║
║    Outer loop: ⌈N_q / B_r⌉ Q-tiles    (B_r = 128 canonical)             ║
║    Inner loop: ⌈N_k / B_c⌉ KV-tiles   (B_c = 128 canonical)             ║
║                                                                          ║
║    Per (Q-tile × KV-tile) CIM pass:                                      ║
║      QNT  q_tile  per-token  (dim=-1) → INT8 + e_q    (once/outer)      ║
║      QNT  kT_tile per-col    (dim=-2) → INT8 + e_kT                     ║
║      MAC  q_i8 × kT_i8  → INT32 [H, B_r, B_c]  score tile              ║
║      DQT  2^(e_q + e_kT) → FP16  (INT8 add + shift; no FP mul)         ║
║      SFM  online: m_new=max(m_old,s_max); corr=exp(m_old−m_new)         ║
║             p=exp(scores−m_new); l_new=corr·l_old+sum(p)  [FP32 SIMD]  ║
║      QNT  p per-token      (dim=-1) → INT8 + e_a                        ║
║      QNT  v_tile per-col   (dim=-2) → INT8 + e_v                        ║
║      MAC  a_i8 × v_i8  → INT32 [H, B_r, h_v]  context tile             ║
║      DQT  2^(e_a + e_v) → FP16  (INT8 add + shift)                     ║
║      ACC  O_tile = corr · O_tile + ctx_fp16.float()   [FP32 SIMD]       ║
║    End inner loop: O_tile /= l_tile  (FP32 → FP16 at Q-tile boundary)  ║
║                                                                          ║
║    SRAM per Q-tile (B_r=B_c=128, H=4, h_qk=h_v=128):                   ║
║      q_tile INT8 resident  [4,128,128]   64 KB                          ║
║      kT/v tiles FP16 in    2×[4,128,128] 256 KB                         ║
║      O_tile FP32 resident  [4,128,128]  256 KB                          ║
║      m, l FP32 (1 scalar/head-row) [4,128,1]   2 KB                    ║
║      scores FP16 transient [4,128,128]  128 KB                          ║
║      p FP32 transient→INT8 [4,128,128]  256 KB peak                     ║
║      Total per Q-tile ≈ 966 KB → fits in 8 MB chiplet SRAM              ║
║                                                                          ║
║  Encoder SA  (6 layers; DMA round 1: load weights, round 2: write out)  ║
║    Q/K/V  FP16 [H=4, N_e=512, H_CA=128]  pre-scaled                    ║
║    flash_attn_e8m0_tiled(q,k,v):                                         ║
║      ⌈512/128⌉=4 Q-tiles × ⌈512/128⌉=4 KV-tiles = 16 CIM passes        ║
║    ctx  FP16 [H=4, 512, 128] → reshape [512, B, 512] → out_proj        ║
║                                                                          ║
║  Decoder SA  (6 layers; tiny N_d=10, single tile)                       ║
║    Q/K/V  FP16 [H=4, N_d=10, H_SA=64]                                  ║
║    flash_attn_e8m0_tiled(q,k,v):                                         ║
║      1 Q-tile × 1 KV-tile = 1 CIM pass  (N_d=10 << B_r=128)            ║
║                                                                          ║
║  Decoder CA  (6 layers; DMA rounds 1-2: enc mem load, round 3: queries) ║
║    q_ca  FP16 [H=4, N_d=10,  H_CA=128]  (decoder queries, pre-scaled)  ║
║    k_ca  FP16 [H=4, N_e=512, H_CA=128]  (encoder memory)               ║
║    v_ca  FP16 [H=4, N_e=512, H_SA=64]   (encoder memory)               ║
║    flash_attn_e8m0_tiled(q_ca, k_ca, v_ca):                              ║
║      1 Q-tile × ⌈512/128⌉=4 KV-tiles = 4 CIM passes                    ║
║    ctx_ca FP16 [H=4, 10, 64] → reshape [10, B, 256] → ca_out_proj      ║
║                                                                          ║
║  E8M0 QNT/DQT at every BMM (canonical blockk128 / flash path):          ║
║    per-token Q  (dim=-1) → INT8 + e_q  INT8 [H, N_q, 1]                ║
║    per-col  K^T (dim=-2) → INT8 + e_kT INT8 [H, 1, N_k]                ║
║    per-token attn (dim=-1) → INT8 + e_a INT8 [H, N_q, 1]               ║
║    per-col  V   (dim=-2) → INT8 + e_v  INT8 [H, 1, h_v]                ║
║    DQT: exponent add (INT8) + exponent-field shift — no FP multiplier   ║
║                                                                          ║
║  Only FP32 datapath: online-softmax running (m, l) on SIMD              ║
║    encoder: 4×512 rows × (4+4 bytes) = 16 KB SIMD state / layer        ║
║    decoder CA: 4×10 rows × 8 bytes  = 0.3 KB SIMD state / layer        ║
╚══════════════════════════════════════════════════════════════════════════╝

Run (canonical recipe):
    python DATAFLOW.py --lin-scale-fmt e8m0 --attn-scale-fmt blockk128 --cuda --batch 1
    python DATAFLOW.py --lin-scale-fmt e8m0 --attn-scale-fmt blockk128 --eval  # full MMVR eval
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.roi_heads import maskrcnn_inference

# ──────────────────────────────────────────────────────────────────────────────
# Import RETR from the sibling src/ tree.
# ──────────────────────────────────────────────────────────────────────────────

_REPO = Path(__file__).resolve().parent
_SRC = _REPO / "src"
sys.path.insert(0, str(_SRC))
os.chdir(_SRC)

from models import RETR                                            # noqa: E402
from models.module_retr import box_ops                             # noqa: E402
from models.module_retr.misc import (                              # noqa: E402
    nested_tensor_from_tensor_list,
    inverse_sigmoid,
)
from quantize_torchao import migrate_encoder_mha_state_dict        # noqa: E402
from data.dataloader import collate_det_seg, get_dataloader        # noqa: E402
from data.det_seg_dataset import MMVRDetSeg                        # noqa: E402
from utils.common import move_to_device                            # noqa: E402
from utils.detection_process import Metrics                        # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Symbol table (P2S1)
# ──────────────────────────────────────────────────────────────────────────────
D, H, FF = 256, 4, 2048       # hidden dim / heads / FFN dim
N_E, N_D = 512, 10            # encoder tokens (256 hor + 256 ver) / decoder queries
L_ENC, L_DEC = 6, 6
H_SA = D // H                 # 64  — decoder SA & half-head in encoder TPE concat
H_CA = 2 * D // H             # 128 — encoder SA head, decoder CA head (2d-wide)


# ═════════════════════════════════════════════════════════════════════════════
# HW PRIMITIVES
# ═════════════════════════════════════════════════════════════════════════════

# ── QNT ──────────────────────────────────────────────────────────────────────

def qnt_act(x_fp16: torch.Tensor, dim: int):
    """QNT unit.  FP16 → INT8 symmetric quantization along `dim`.

    scale  = absmax(x, dim=dim, keepdim) / 127       (FP16, per-row or per-col)
    x_i8   = round(clip(x / scale, -127, 127))       (INT8)

    Returns (x_i8, scale_fp16).  `dim=-1` gives per-token (used for act / lhs),
    `dim=-2` gives per-col (used for rhs of matmul along its K axis).
    """
    assert x_fp16.dtype == torch.float16, f"QNT expects FP16, got {x_fp16.dtype}"
    scale = x_fp16.abs().amax(dim=dim, keepdim=True).clamp(min=1e-6) / 127.0  # FP16
    x_i8  = (x_fp16 / scale).clamp(-127, 127).round().to(torch.int8)          # INT8
    return x_i8, scale


# ── MAC ──────────────────────────────────────────────────────────────────────

_INT8_MM_MIN_MN = 32          # torch._int_mm CUDA: requires M>16, N>16; K must be % 8


def int8_mac(a_i8: torch.Tensor, b_i8: torch.Tensor) -> torch.Tensor:
    """MAC unit.  INT8 × INT8 → INT32 systolic matmul.

    Accepts 2D [M,K]·[K,N] or 3D [B,M,K]·[B,K,N].  CPU path uses INT32 matmul
    (exact integer math).  CUDA path uses torch._int_mm per batch with a shape
    pad (the kernel requires M>16, N>16, K%8==0 — padded tail is sliced off).
    """
    squeezed = a_i8.dim() == 2
    if squeezed:
        a_i8, b_i8 = a_i8.unsqueeze(0), b_i8.unsqueeze(0)
    B, M, K = a_i8.shape
    _, _, N = b_i8.shape
    M_eff = max(M, _INT8_MM_MIN_MN)
    N_eff = max(N, _INT8_MM_MIN_MN)
    K_eff = ((K + 7) // 8) * 8
    if K_eff > K:
        a_i8 = F.pad(a_i8, (0, K_eff - K))
        b_i8 = F.pad(b_i8, (0, 0, 0, K_eff - K))
    if M_eff > M:
        a_i8 = F.pad(a_i8, (0, 0, 0, M_eff - M))
    if N_eff > N:
        b_i8 = F.pad(b_i8, (0, N_eff - N))
    if a_i8.is_cuda:
        out = torch.stack([torch._int_mm(a_i8[i].contiguous(), b_i8[i].contiguous())
                           for i in range(B)])                                # INT32
    else:
        out = torch.matmul(a_i8.to(torch.int32), b_i8.to(torch.int32))        # INT32 (exact)
    out = out[:, :M, :N]
    return out.squeeze(0) if squeezed else out


# ── DQT ──────────────────────────────────────────────────────────────────────

def dqt_fp16(acc_i32: torch.Tensor, s_a_fp16: torch.Tensor, s_b_fp16: torch.Tensor) -> torch.Tensor:
    """DQT unit.  INT32 × (FP16 act_scale · FP16 weight_scale) → FP16.

    The post-MAC dequant is the single HW stage where an INT32 accumulator meets
    two FP16 scales.  We widen to FP32 inside this unit (not in the dataflow) so
    the multiply doesn't overflow FP16 for large INT32 magnitudes, and cast back
    to FP16 at the unit boundary.
    """
    assert acc_i32.dtype == torch.int32
    return (acc_i32.float() * (s_a_fp16.float() * s_b_fp16.float())).to(torch.float16)


# ── E8M0 scale variants ───────────────────────────────────────────────────────

def qnt_act_e8m0(x_fp16: torch.Tensor, dim: int):
    """E8M0 QNT.  Same granularity as qnt_act but scale = 2^ceil(log2(absmax/127)).

    Returns (x_i8, e) where e is an INT8 tensor of exponents (one per row/col).
    Hardware: the log2+ceil is a leading-zero-count + round, much cheaper than
    the FP16 reciprocal required by qnt_act.
    """
    assert x_fp16.dtype == torch.float16, f"E8M0 QNT expects FP16, got {x_fp16.dtype}"
    absmax = x_fp16.abs().amax(dim=dim, keepdim=True).clamp(min=1e-6)
    e      = torch.ceil(torch.log2(absmax / 127.0)).clamp(-127, 127).to(torch.int8)
    scale  = torch.pow(2.0, e.float()).to(torch.float16)
    x_i8   = (x_fp16 / scale).clamp(-127, 127).round().to(torch.int8)
    return x_i8, e


def dqt_e8m0_lin(acc_i32: torch.Tensor, e_a: torch.Tensor, w_scale_fp16: torch.Tensor) -> torch.Tensor:
    """DQT for linear layers: E8M0 activation scale × FP16 weight scale → FP16.

    Hardware: the act scale is a bit-shift in the FP32 exponent field; only the
    weight-scale multiply needs a general FP multiplier.
    """
    assert acc_i32.dtype == torch.int32
    return (acc_i32.float() * (torch.pow(2.0, e_a.float()) * w_scale_fp16.float())).to(torch.float16)


def dqt_e8m0_attn(acc_i32: torch.Tensor, e_a: torch.Tensor, e_b: torch.Tensor) -> torch.Tensor:
    """DQT for attention BMMs: both scales are E8M0 exponents.

    output = acc_i32 × 2^(e_a + e_b) → FP16.
    Hardware: one INT8 add in the exponent fields per output tile + a shift —
    no FP multiplier at all in the DQT path.
    """
    assert acc_i32.dtype == torch.int32
    exp = (e_a.float() + e_b.float()).clamp(-126, 126)   # keep inside FP32 range
    return (acc_i32.float() * torch.pow(2.0, exp)).to(torch.float16)


def block_k_bmm_e8m0(a_fp16: torch.Tensor, b_fp16: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """Block-K E8M0 INT8 matmul — models tile-level DQT (SageAttention2 / NVFP4 style).

    K is tiled into slices of `block_size`.  For each K-slice:
      • per-token E8M0 QNT on a  (one exponent per row,  reducing over K-slice)
      • per-col   E8M0 QNT on b  (one exponent per col,  reducing over K-slice)
      • INT8 × INT8 → INT32 MAC for the slice
      • DQT: 2^(e_a + e_b) scalar broadcast — one exponent-add per tile
      • accumulate partial into FP16 output

    Hardware cost vs baseline outer-product DQT:
      baseline : M × N FP16 multiplies per forward (e.g. 512×512 = 262 K)
      block-K  : (K/B) × 2 INT8 adds + shifts amortised over M × N tile
                 ≈ (K/B) × (M/T_m) × (T_m + T_n) scalar ops instead.

    Accepts 2-D [M,K]·[K,N] or 3-D [B,M,K]·[B,K,N].  b is NOT pre-transposed.
    """
    assert a_fp16.dtype == torch.float16 and b_fp16.dtype == torch.float16
    squeezed = a_fp16.dim() == 2
    if squeezed:
        a_fp16 = a_fp16.unsqueeze(0)
        b_fp16 = b_fp16.unsqueeze(0)
    BT, M, K = a_fp16.shape
    _,  _, N = b_fp16.shape
    out = torch.zeros(BT, M, N, dtype=torch.float16, device=a_fp16.device)
    for start in range(0, K, block_size):
        end   = min(start + block_size, K)
        a_blk = a_fp16[:, :, start:end].contiguous()           # [BT, M, bs]
        b_blk = b_fp16[:, start:end, :].contiguous()           # [BT, bs, N]
        ea = torch.ceil(torch.log2(
            a_blk.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 127.0
        )).clamp(-127, 127)                                     # [BT, M, 1] FP32 exp
        sa = torch.pow(2.0, ea).to(torch.float16)
        ai8 = (a_blk / sa).clamp(-127, 127).round().to(torch.int8)
        eb = torch.ceil(torch.log2(
            b_blk.abs().amax(dim=-2, keepdim=True).clamp(min=1e-6) / 127.0
        )).clamp(-127, 127)                                     # [BT, 1, N] FP32 exp
        sb = torch.pow(2.0, eb).to(torch.float16)
        bi8 = (b_blk / sb).clamp(-127, 127).round().to(torch.int8)
        acc  = int8_mac(ai8, bi8)                               # INT32 [BT, M, N]
        out  = out + (acc.float() * sa.float() * sb.float()).to(torch.float16)
    return out.squeeze(0) if squeezed else out


def flash_attn_e8m0_tiled(
    q_fp16: torch.Tensor,    # [H, N_q, h_qk]  FP16  (pre-scaled by 1/√h_qk)
    k_fp16: torch.Tensor,    # [H, N_k, h_qk]  FP16  (NOT transposed)
    v_fp16: torch.Tensor,    # [H, N_k, h_v]   FP16
    blk_q:  int = 128,       # Q-tile rows  (outer loop)
    blk_k:  int = 128,       # KV-tile rows (inner loop)
) -> torch.Tensor:           # [H, N_q, h_v]  FP16
    """Two-level tiled Flash Attention with E8M0 INT8 tiles and online softmax.

    Outer loop: ⌈N_q/blk_q⌉ Q-tiles.
    Inner loop: ⌈N_k/blk_k⌉ KV-tiles.

    Per (Q-tile × KV-tile) inner iteration — maps to one CIM tile pass:
      QNT  q_tile  [H,B_r,h_qk] per-token  (dim=-1) → INT8 + e_q INT8     (done once/outer iter)
      QNT  kT_tile [H,h_qk,B_c] per-col    (dim=-2) → INT8 + e_kT INT8
      MAC  q_i8 × kT_i8                             → INT32 [H,B_r,B_c]   (score tile)
      DQT  2^(e_q + e_kT) → FP16                    (INT8 add + shift; no FP mul)
      SFM  online update  m_new, correction, l_new  (m/l FP32 on SIMD unit)
      QNT  p = exp(scores − m_new) [H,B_r,B_c] per-token → INT8 + e_a INT8
      QNT  v_tile [H,B_c,h_v]     per-col (dim=-2)      → INT8 + e_v INT8
      MAC  a_i8 × v_i8                               → INT32 [H,B_r,h_v]  (context partial)
      DQT  2^(e_a + e_v) → FP16                     (INT8 add + shift)
      ACC  O_tile = O_tile * correction + ctx_fp16.float()   (FP32 SIMD)

    After inner loop: O_tile /= l_tile   (FP32 → FP16 at Q-tile boundary)

    SRAM per inner iteration (canonical B_r=B_c=128, H=4, h_qk=h_v=128):
      q_tile (INT8, resident across inner loop)   [4,128,128]   64 KB
      kT/v tiles (FP16 → INT8 each inner iter)    [4,128,128] + [4,128,128]  256 KB FP16 in
      O_tile (FP32, resident across inner loop)   [4,128,128]  256 KB
      m, l   (FP32, 1 scalar/head-row)            [4,128,1]      2 KB
      scores (FP16 transient)                     [4,128,128]  128 KB
      p      (FP32 transient → INT8)              [4,128,128]  256 KB peak
      Total per Q-tile                            ≈ 966 KB → fits in 8 MB
    """
    H, N_q, h_qk = q_fp16.shape
    _, N_k, h_v  = v_fp16.shape

    O_out = torch.zeros(H, N_q, h_v, dtype=torch.float32, device=q_fp16.device)

    for qs in range(0, N_q, blk_q):
        qe   = min(qs + blk_q, N_q)
        B_r  = qe - qs
        q_tile = q_fp16[:, qs:qe, :].contiguous()                        # FP16 [H, B_r, h_qk]
        O_tile = torch.zeros(H, B_r, h_v,  dtype=torch.float32, device=q_fp16.device)
        m_tile = torch.full( (H, B_r, 1),  float('-inf'), dtype=torch.float32, device=q_fp16.device)
        l_tile = torch.zeros(H, B_r, 1,    dtype=torch.float32, device=q_fp16.device)

        # QNT: quantize Q tile once; reused across all KV tiles
        q_i8, e_q = qnt_act_e8m0(q_tile, dim=-1)                         # INT8 [H,B_r,h_qk], e_q INT8 [H,B_r,1]

        for ks in range(0, N_k, blk_k):
            ke   = min(ks + blk_k, N_k)
            B_c  = ke - ks
            k_tile = k_fp16[:, ks:ke, :].contiguous()                    # FP16 [H, B_c, h_qk]
            v_tile = v_fp16[:, ks:ke, :].contiguous()                    # FP16 [H, B_c, h_v]

            # ── Score BMM ────────────────────────────────────────────────
            kT_tile = k_tile.transpose(1, 2).contiguous()                 # FP16 [H, h_qk, B_c]
            kT_i8, e_kT = qnt_act_e8m0(kT_tile, dim=-2)                  # QNT @ kT (per-col)
            acc_s = int8_mac(q_i8, kT_i8)                                 # INT32 [H, B_r, B_c]
            exp_s = (e_q.float() + e_kT.float()).clamp(-126, 126)         # FP32 exponent-add
            scores = (acc_s.float() * torch.pow(2.0, exp_s)).to(torch.float16)  # FP16 (scale=1; q pre-scaled)

            # ── Online softmax (SFM on SIMD, FP32 running stats) ─────────
            s_max  = scores.float().amax(dim=-1, keepdim=True)            # FP32 [H,B_r,1] per-row max
            m_new  = torch.maximum(m_tile, s_max)                         # FP32 running max update
            correction = torch.exp(m_tile - m_new)                       # FP32 rescale factor
            p      = torch.exp(scores.float() - m_new)                   # FP32 [H,B_r,B_c] unnorm probs
            l_tile = correction * l_tile + p.sum(dim=-1, keepdim=True)   # FP32 running normalizer
            m_tile = m_new

            # ── Context BMM ───────────────────────────────────────────────
            p_fp16 = p.to(torch.float16)                                  # FP16 [H, B_r, B_c]
            a_i8,  e_a = qnt_act_e8m0(p_fp16,  dim=-1)                   # QNT @ p  (per-token)
            v_i8,  e_v = qnt_act_e8m0(v_tile,  dim=-2)                   # QNT @ v  (per-col, [H,B_c,h_v])
            acc_o = int8_mac(a_i8, v_i8)                                  # INT32 [H, B_r, h_v]
            exp_o = (e_a.float() + e_v.float()).clamp(-126, 126)
            ctx_fp16 = (acc_o.float() * torch.pow(2.0, exp_o)).to(torch.float16)  # FP16

            # ── FP32 accumulate with online rescale ───────────────────────
            O_tile = correction * O_tile + ctx_fp16.float()

        # Normalize and write Q-tile output
        O_out[:, qs:qe, :] = (O_tile / l_tile)                           # FP32 normalize

    return O_out.to(torch.float16)                                        # [H, N_q, h_v] FP16


# ── Scale dispatch (module-level; set by main() before calling dataflow_forward) ─
# _QNTL / _DQTL : linear-layer activation QNT and DQT
# _QNTA / _DQTA : attention Q/K/V/softmax QNT and DQT (when _FLASH_ATTN is None)
# _FLASH_ATTN   : if not None, replaces the QNT+MAC+DQT+SFM triple for every
#                 attention (Q,K,V)->ctx computation with flash_attn_e8m0_tiled
# _BLK_Q/_BLK_K : Q-tile and KV-tile sizes for flash_attn_e8m0_tiled

_QNTL:       callable = qnt_act
_DQTL:       callable = dqt_fp16
_QNTA:       callable = qnt_act
_DQTA:       callable = dqt_fp16
_FLASH_ATTN: object   = None   # set to flash_attn_e8m0_tiled lambda for blockk*
_BLK_Q:      int      = 128
_BLK_K:      int      = 128


# ── MXQ (FP4 E2M1 block quant) ───────────────────────────────────────────────

_FP4_POS_VALS   = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_FP4_BOUNDARIES = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
_FP4_MAX = 6.0


def mx_fp4(x_fp16: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """MXQ unit.  Fake-quantize FP16 → FP4 E2M1 grid on blocks of `block_size`
    along the last dim.  Per-block scale is a power-of-two.  Output FP16.

    Exponent extraction (log2 / ceil) runs in FP32 internally (LUT math, not a
    dataflow tensor).
    """
    assert x_fp16.dtype == torch.float16
    orig_shape = x_fp16.shape
    x32 = x_fp16.float()                                           # unit-internal widen
    last = orig_shape[-1]
    pad = (block_size - last % block_size) % block_size
    if pad:
        x32 = F.pad(x32, (0, pad))
    blocks = x32.reshape(*x32.shape[:-1], -1, block_size)
    max_abs = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale   = torch.pow(2.0, torch.ceil(torch.log2(max_abs / _FP4_MAX)))
    scaled  = (blocks / scale).clamp(-_FP4_MAX, _FP4_MAX)
    idx     = torch.bucketize(scaled.abs().contiguous(), _FP4_BOUNDARIES.to(x_fp16.device))
    dq      = (scaled.sign() * _FP4_POS_VALS.to(x_fp16.device)[idx] * scale).reshape(
        *orig_shape[:-1], last + pad
    )
    if pad:
        dq = dq[..., :last]
    return dq.to(torch.float16)                                    # unit-output narrow


# ── LN (MXQ + F.layer_norm) ──────────────────────────────────────────────────

def fp16_ln(x_fp16: torch.Tensor, weight_fp16: torch.Tensor, bias_fp16: torch.Tensor,
            dim: int, block_size: int = 32) -> torch.Tensor:
    """LN unit.  MXQ(x, block_size) → F.layer_norm with FP16 affine.  FP16 in/out."""
    assert x_fp16.dtype == torch.float16
    x_mxq = mx_fp4(x_fp16, block_size=block_size)                  # MXQ FP16 → FP16 (FP4 grid)
    return F.layer_norm(x_mxq, (dim,), weight_fp16, bias_fp16)     # FPU FP16


# ── SFM (softmax with FP32 row-sum accumulator) ──────────────────────────────

def fp16_softmax_fp32_accum(x_fp16: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """SFM unit.  The single FP32 exception in the pipeline.

    Pipeline:
        xmax  = max(x, dim)                     FP16 reduction
        e     = exp(x - xmax)                   FP16 elementwise
        s_32  = sum(e, dim) — accumulated in FP32
        inv_s = (1/s_32).to(FP16)               reciprocal in FP32, narrow at unit boundary
        y     = e * inv_s                       FP16 multiply

    This matches the HW softmax FPU which uses a wider accumulator for stability
    but produces FP16 probabilities.
    """
    assert x_fp16.dtype == torch.float16
    xmax  = x_fp16.amax(dim=dim, keepdim=True)                     # FP16
    e     = (x_fp16 - xmax).exp()                                  # FP16
    s_32  = e.float().sum(dim=dim, keepdim=True)                   # FP32 accumulator
    inv_s = s_32.reciprocal().to(torch.float16)                    # FP16
    return e * inv_s                                               # FP16


# ── Sine/cos 3D positional embedding (used per decoder layer) ────────────────

def sineembed_3d(pos_fp16: torch.Tensor) -> torch.Tensor:
    """Sine/cos positional embedding for a 3D object center [*, 3] → [*, 256].
    All FP16. The constant dim_t LUT is FP16; values that overflow saturate and
    give zero contribution (same behavior on HW)."""
    assert pos_fp16.dtype == torch.float16
    scale, dim = 2 * math.pi, 86
    dim_t = 10000 ** (3 * (torch.arange(dim, dtype=pos_fp16.dtype, device=pos_fp16.device) // 3) / dim)
    outs = []
    for i in range(3):
        p = (pos_fp16[..., i] * scale).unsqueeze(-1) / dim_t        # FP16
        outs.append(torch.stack((p[..., 0::2].sin(), p[..., 1::2].cos()), dim=-1).flatten(-2))
    return torch.cat(outs, dim=-1)[..., :D]                         # FP16 [*, 256]


# ═════════════════════════════════════════════════════════════════════════════
# WEIGHT STORAGE CONTAINERS (used by load_state_dict)
# ═════════════════════════════════════════════════════════════════════════════

class Int8DynLinear(nn.Module):
    """INT8 W8A8 Linear.  Stores:
        w_i8     [out, in]   INT8     symmetric per-output-channel quantized weight
        w_scale  [out, 1]    FP16     per-output-channel FP16 scale
        bias     [out]       FP16

    forward() wraps the HW sequence:
        x(FP16) → QNT(per-token) → MAC(INT32) → DQT(→FP16) → + bias → FP16
    Used in main() directly; every call site is commented with FP16 shapes.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("w_i8",    torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("w_scale", torch.ones(out_features, 1, dtype=torch.float16))
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16)) if bias else None

    @classmethod
    def from_linear(cls, lin: nn.Linear) -> "Int8DynLinear":
        m = cls(lin.in_features, lin.out_features, bias=lin.bias is not None).to(lin.weight.device)
        w = lin.weight.data.float()
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127.0
        m.w_i8.copy_((w / scale).clamp(-127, 127).round().to(torch.int8))
        m.w_scale.copy_(scale.to(torch.float16))
        if lin.bias is not None:
            m.bias.data.copy_(lin.bias.data.to(torch.float16))
        return m

    def forward(self, x_fp16: torch.Tensor) -> torch.Tensor:
        in_shape = x_fp16.shape
        x2 = x_fp16.reshape(-1, in_shape[-1])                                 # FP16 [N, in]
        x_i8, s_x = _QNTL(x2, dim=-1)                                         # INT8 [N,in], s FP16 [N,1]
        acc = int8_mac(x_i8, self.w_i8.t().contiguous())                      # INT32 [N, out]
        y = _DQTL(acc, s_x, self.w_scale.t())                                 # FP16 [N, out]
        if self.bias is not None:
            y = y + self.bias                                                 # FP16
        return y.reshape(*in_shape[:-1], self.out_features)


class FP16BlockLN(nn.Module):
    """FP4-block-quantized LayerNorm.  Stores:
        weight  [dim]  FP16
        bias    [dim]  FP16
    forward() = fp16_ln(x, weight, bias, dim, block_size).
    """

    def __init__(self, dim: int, block_size: int = 32):
        super().__init__()
        self.dim, self.block_size = dim, block_size
        self.weight = nn.Parameter(torch.ones(dim,  dtype=torch.float16))
        self.bias   = nn.Parameter(torch.zeros(dim, dtype=torch.float16))

    @classmethod
    def from_layernorm(cls, ln: nn.LayerNorm, block_size: int = 32) -> "FP16BlockLN":
        m = cls(ln.normalized_shape[0], block_size=block_size).to(ln.weight.device)
        m.weight.data.copy_(ln.weight.data.to(torch.float16))
        m.bias.data.copy_(ln.bias.data.to(torch.float16))
        return m

    def forward(self, x_fp16: torch.Tensor) -> torch.Tensor:
        return fp16_ln(x_fp16, self.weight, self.bias, self.dim, self.block_size)


# ═════════════════════════════════════════════════════════════════════════════
# Model mutation: FP16 cast of CGRA modules + INT8/FP16-LN swap of CIM modules
# ═════════════════════════════════════════════════════════════════════════════

def quantize_transformer_inplace(detr: nn.Module) -> None:
    """Walk `detr` and swap every transformer-scope nn.Linear → Int8DynLinear,
    nn.LayerNorm → FP16BlockLN, in place."""
    for parent_fqn, parent in detr.named_modules():
        for child_name, child in list(parent.named_children()):
            fqn = f"{parent_fqn}.{child_name}" if parent_fqn else child_name
            if not ("encoder" in fqn or "decoder" in fqn):
                continue
            if isinstance(child, nn.Linear):
                setattr(parent, child_name, Int8DynLinear.from_linear(child))
            elif isinstance(child, nn.LayerNorm):
                setattr(parent, child_name, FP16BlockLN.from_layernorm(child))


def fp16_cast_cgra(retr: nn.Module) -> None:
    """Cast the CGRA-mapped modules (backbone, input_proj, query_embed, heads,
    box_affine_transformer, seg head) to FP16 in place.  BatchNorm stats and
    affine params are cast too — CGRA runs BN in FP16."""
    core = retr.model.detr if hasattr(retr.model, "detr") else retr.model
    core.backbone.half()
    core.input_proj.half()
    core.input_proj_ver.half()
    core.query_embed.half()
    core.class_embed.half()
    core.bbox_embed.half()
    core.box_affine_transformer.half()
    # SEG mode: segmentation head modules sit on DETRsegm (retr.model), not on detr.
    if hasattr(retr.model, "bbox_attention"):
        retr.model.bbox_attention.half()
        retr.model.mask_head.half()
        retr.model.unet.half()


# ═════════════════════════════════════════════════════════════════════════════
# END-TO-END PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def dataflow_forward(retr, detr, hor: torch.Tensor, ver: torch.Tensor) -> dict:
    """One end-to-end forward pass through the ASIC-simulator pipeline.

    Args:
        hor, ver: FP16 [B, 4, H, W]  horizontal / vertical radar maps
    Returns:
        dict with the DETR-compatible keys:
            pred_logits, pred_boxes, pred_hboxes, pred_vboxes, pred_iboxes,
            pred_hboxes_aug, pred_vboxes_aug, proj_boxes, hs, memory, reference
    """
    assert hor.dtype == torch.float16 and ver.dtype == torch.float16
    B = hor.shape[0]

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 1 — FFT  (CGRA, per view, parameter-free)
    #   HW: 2D FFT [4,256,128] complex → |·| → log(·+ε) → normalize → FP16
    #   SW path: MMVR ships pre-computed post-FFT heatmaps; hor/ver already FP16.
    #   Twiddle ROM: ~0.2 KB on-chip (never DMA'd); ~4M radix-2 butterfly ops/view.
    # ─────────────────────────────────────────────────────────────────────────
    # [FFT already applied upstream; hor/ver ∈ FP16 [B, 4, 256, 128]]

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 2 — Backbone CNN  (CGRA, FP16 native; shared weights V=2 views)
    #   ResNet18 + FPN level-0–2.  BN is nn.BatchNorm2d; at inference (eval)
    #   BN uses frozen running_mean/var → acts as fixed affine transform, folded
    #   into the preceding conv by the CGRA compiler (0 extra MACs on HW).
    #   CGRA tiling strategy for conv1 (largest tile):
    #     input  [4,256,128] FP16 = 256 KB/view → 4 horizontal strips h=32
    #     each strip: 128 KB in + 128 KB conv1-out → 256 KB resident + weights
    # ─────────────────────────────────────────────────────────────────────────
    _igtr = detr.backbone[0].body.body          # IntermediateLayerGetter (ResNet18 dict)
    _fpn  = detr.backbone[0].body.fpn           # FeaturePyramidNetwork

    def _backbone_fpn(x_fp16: torch.Tensor):
        """ResNet18 + FPN for one view. Returns (f0, f1, f2) FP16."""
        # conv1 7×7 stride-2 + BN-fold + ReLU
        x = _igtr.relu(_igtr.bn1(_igtr.conv1(x_fp16)))    # FP16 [B, 64, 128, 64]
        # HW: 4 strips of H=32 rows (128 KB in + 128 KB out per strip)
        x = _igtr.maxpool(x)                               # FP16 [B, 64,  64, 32]  3×3 s=2
        # ResNet stages (each stage = 2×BasicBlock; BB = conv3×3→BN→ReLU→conv3×3→BN⊕res→ReLU)
        c2 = _igtr.layer1(x)                               # FP16 [B, 64,  64, 32]  BB(64→64,  s=1)×2
        c3 = _igtr.layer2(c2)                              # FP16 [B,128,  32, 16]  BB(64→128, s=2)×2
        c4 = _igtr.layer3(c3)                              # FP16 [B,256,  16,  8]  BB(128→256,s=2)×2
        c5 = _igtr.layer4(c4)                              # FP16 [B,512,   8,  4]  BB(256→512,s=2)×2
        # FPN top-down: lateral 1×1 (C→64), bilinear 2× upsample, 3×3 smooth
        p5 = _fpn.inner_blocks[3](c5)                                              # FP16 [B,64,  8,  4]
        p4 = _fpn.inner_blocks[2](c4) + F.interpolate(p5, size=c4.shape[-2:], mode='nearest')  # FP16 [B,64, 16,  8]
        p3 = _fpn.inner_blocks[1](c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')  # FP16 [B,64, 32, 16]
        p2 = _fpn.inner_blocks[0](c2) + F.interpolate(p3, size=c2.shape[-2:], mode='nearest')  # FP16 [B,64, 64, 32]
        f0 = _fpn.layer_blocks[0](p2)              # FP16 [B, 64, 64, 32]  ← det head + seg-L0
        f1 = _fpn.layer_blocks[1](p3)              # FP16 [B, 64, 32, 16]  ← seg-L1
        f2 = _fpn.layer_blocks[2](p4)              # FP16 [B, 64, 16,  8]  ← seg-L2
        return f0, f1, f2

    feats_h = _backbone_fpn(hor)               # tuple (f0,f1,f2) FP16
    feats_v = _backbone_fpn(ver)
    src_h_fp16 = feats_h[0]                    # FP16 [B, 64, 64, 32]
    src_v_fp16 = feats_v[0]                    # FP16 [B, 64, 64, 32]

    # 2D sine positional embedding (CGRA, parameter-free; called via Joiner PE module)
    from models.module_retr.misc import NestedTensor
    _nt_h = nested_tensor_from_tensor_list(hor)
    _nt_v = nested_tensor_from_tensor_list(ver)
    def _lvl_mask(nt, hw): return F.interpolate(nt.mask[None].float(), hw).to(torch.bool)[0]
    pos_h_fp16 = detr.backbone[1](
        NestedTensor(src_h_fp16, _lvl_mask(_nt_h, src_h_fp16.shape[-2:]))
    ).to(torch.float16)                        # CVT FP32→FP16; FP16 [B, 256, 64, 32]
    pos_v_fp16 = detr.backbone[1](
        NestedTensor(src_v_fp16, _lvl_mask(_nt_v, src_v_fp16.shape[-2:]))
    ).to(torch.float16)

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 3a — input_proj: 1×1 Conv2d [64→256]  (CGRA, FP16)
    # ─────────────────────────────────────────────────────────────────────────
    src_proj_h = detr.input_proj(src_h_fp16)   # FP16 [B, 256, 64, 32]   1×1 conv lift 64→256
    src_proj_v = detr.input_proj_ver(src_v_fp16)  # FP16 [B, 256, 64, 32]

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 3b — Tokenizer: L2-norm → top-K → gather  (CGRA, parameter-free)
    #   L2-norm over channels: [B,256,64,32] → [B,64,32] (2048 scores/view)
    #   top-256 spatial locs by L2 magnitude: argsort → [B,256] indices
    #   gather features + PE: [B,256,256] → reshape [B,256,16,16]
    #   CGRA: sort 2048 scores or offload 4 KB of scores to FPGA for top-K.
    # ─────────────────────────────────────────────────────────────────────────
    def _topk_select(feat: torch.Tensor, pos: torch.Tensor, topk: int = 256):
        """Top-K spatial tokens by L2 norm. feat,pos: FP16 [B,C,H,W]."""
        Bv, C, Hf, Wf = feat.shape
        l2_scores = feat.norm(dim=1)                                       # FP16 [B, H, W]  L2-norm over channels
        _, idx = l2_scores.flatten(1).topk(topk, dim=1)                   # [B, topk] indices in flattened spatial
        sqrt_k = int(math.isqrt(topk))                                     # 16
        idx_e  = idx.unsqueeze(1).expand(-1, C, -1)                        # [B, C, topk]
        fea_sel = feat.flatten(2).gather(2, idx_e).reshape(Bv, C, sqrt_k, sqrt_k)   # FP16 [B, 256, 16, 16]
        idx_ep = idx.unsqueeze(1).expand(-1, pos.shape[1], -1)             # [B, 256, topk]
        pos_sel = pos.flatten(2).gather(2, idx_ep).reshape(Bv, pos.shape[1], sqrt_k, sqrt_k)  # FP16 [B, 256, 16, 16]
        return fea_sel, pos_sel

    topk_fea_h, topk_pos_h = _topk_select(src_proj_h, pos_h_fp16)         # each FP16 [B, 256, 16, 16]
    topk_fea_v, topk_pos_v = _topk_select(src_proj_v, pos_v_fp16)

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Encoder input assembly — concat hor/ver tokens along the sequence axis
    # ─────────────────────────────────────────────────────────────────────────
    src = torch.cat([topk_fea_h.flatten(2).permute(2, 0, 1),       # FP16 [256, B, 256]
                     topk_fea_v.flatten(2).permute(2, 0, 1)],      # FP16 [256, B, 256]
                    dim=0).contiguous()                            # FP16 [N_E=512, B, 256]
    pos_embed = torch.cat([topk_pos_h.flatten(2).permute(2, 0, 1),
                           topk_pos_v.flatten(2).permute(2, 0, 1)],
                          dim=0).contiguous()                      # FP16 [N_E=512, B, 256]

    # ─────────────────────────────────────────────────────────────────────────
    # 7b. CGRA→CIM boundary: QNT-at-production for the inputs the encoder
    #     consumes via INT8 MACs.  pos_embed is constant across all 6 encoder
    #     layers AND is reused by the decoder cross-attention — one QNT here
    #     replaces 12 (6 enc + 6 dec).  src is QNT'd once at the boundary;
    #     subsequent updates are QNT'd at each LN output (see (i),(l) below).
    # ─────────────────────────────────────────────────────────────────────────
    src_i8, s_src = _QNTL(src.reshape(-1, D),       dim=-1)      # QNT @ encoder input (CGRA topk concat)
    pos_i8, s_pos = _QNTL(pos_embed.reshape(-1, D), dim=-1)      # QNT once; reused by every enc + dec layer

    # ─────────────────────────────────────────────────────────────────────────
    # 8. Encoder ×6 — every Linear/BMM expanded inline as MAC→DQT→FPU.
    #    QNT is fused with each producing op (LN, BMM-out, softmax, ReLU,
    #    concat) so activations cross the long-haul edge as (INT8, FP16
    #    scale).  FP16 forms are local/transient (residual buffer only).
    # ─────────────────────────────────────────────────────────────────────────
    for i in range(L_ENC):
        L = detr.encoder.layers[i]                                 # ConditionalTransformerEncoderLayer (i of 6)

        # (a) Six content/pos projections — 3× MAC on src_i8, 3× MAC on pos_i8.
        #     Inputs already QNT'd at production (LN2 of prev iter / boundary).
        W = L.ca_qcontent_proj
        acc = int8_mac(src_i8, W.w_i8.t().contiguous())                                       # MAC  INT32 [N,256]
        q_c = (_DQTL(acc, s_src, W.w_scale.t()) + W.bias).reshape(N_E, B, D)               # DQT+FPU → FP16 [N_E,B,256]

        W = L.ca_kcontent_proj
        acc = int8_mac(src_i8, W.w_i8.t().contiguous())                                       # MAC
        k_c = (_DQTL(acc, s_src, W.w_scale.t()) + W.bias).reshape(N_E, B, D)               # DQT+FPU

        W = L.ca_v_proj
        acc = int8_mac(src_i8, W.w_i8.t().contiguous())                                       # MAC
        v_c = (_DQTL(acc, s_src, W.w_scale.t()) + W.bias).reshape(N_E, B, D)               # DQT+FPU

        W = L.ca_kpos_proj
        acc = int8_mac(pos_i8, W.w_i8.t().contiguous())                                       # MAC
        k_p = (_DQTL(acc, s_pos, W.w_scale.t()) + W.bias).reshape(N_E, B, D)               # DQT+FPU

        W = L.ca_vpos_proj
        acc = int8_mac(pos_i8, W.w_i8.t().contiguous())                                       # MAC
        v_p = (_DQTL(acc, s_pos, W.w_scale.t()) + W.bias).reshape(N_E, B, D)               # DQT+FPU

        W = L.ca_qpos_sine_proj
        acc = int8_mac(pos_i8, W.w_i8.t().contiguous())                                       # MAC
        q_s = (_DQTL(acc, s_pos, W.w_scale.t()) + W.bias).reshape(N_E, B, D)               # DQT+FPU

        # (b) TPE concat → 2d-wide [N_E, B, 512].  QNT-at-production: concat
        #     output is stored as (INT8, FP16 scale) — never as FP16 tile.
        q_cat = torch.cat([q_c.view(N_E, B, H, H_SA), q_s.view(N_E, B, H, H_SA)], dim=3).reshape(N_E, B, 2 * D)
        k_cat = torch.cat([k_c.view(N_E, B, H, H_SA), k_p.view(N_E, B, H, H_SA)], dim=3).reshape(N_E, B, 2 * D)
        v_cat = torch.cat([v_c.view(N_E, B, H, H_SA), v_p.view(N_E, B, H, H_SA)], dim=3).reshape(N_E, B, 2 * D)
        qcat_i8, s_qcat = _QNTL(q_cat.reshape(-1, 2 * D), dim=-1)                           # QNT @ concat
        kcat_i8, s_kcat = _QNTL(k_cat.reshape(-1, 2 * D), dim=-1)                           # QNT @ concat
        vcat_i8, s_vcat = _QNTL(v_cat.reshape(-1, 2 * D), dim=-1)                           # QNT @ concat

        # (c) SA Q/K/V projections — 3× 512→512 MAC.
        W = L.self_attn_q_proj
        acc = int8_mac(qcat_i8, W.w_i8.t().contiguous())                                      # MAC  INT32 [N,512]
        q   = (_DQTL(acc, s_qcat, W.w_scale.t()) + W.bias).reshape(N_E, B, 2 * D)          # DQT+FPU
        q   = q * (H_CA ** -0.5)                                                              # FPU  scale

        W = L.self_attn_k_proj
        acc = int8_mac(kcat_i8, W.w_i8.t().contiguous())                                      # MAC
        k   = (_DQTL(acc, s_kcat, W.w_scale.t()) + W.bias).reshape(N_E, B, 2 * D)          # DQT+FPU

        W = L.self_attn_v_proj
        acc = int8_mac(vcat_i8, W.w_i8.t().contiguous())                                      # MAC
        v   = (_DQTL(acc, s_vcat, W.w_scale.t()) + W.bias).reshape(N_E, B, 2 * D)          # DQT+FPU

        # (d) Multi-head reshape + QNT-at-production for the attention BMMs.
        q  = q.view(N_E, B * H, H_CA).transpose(0, 1).contiguous()                            # FP16 [B·H=4, 512, 128]
        k  = k.view(N_E, B * H, H_CA).transpose(0, 1).contiguous()                            # FP16 [B·H, 512, 128]
        v  = v.view(N_E, B * H, H_CA).transpose(0, 1).contiguous()                            # FP16 [B·H, 512, 128]
        kT = k.transpose(1, 2).contiguous()                                                   # FP16 [B·H, 128, 512]
        # (e) Attention score BMM: Q · Kᵀ  → [B·H,512,512]
        if _FLASH_ATTN is not None:
            ctx = _FLASH_ATTN(q, k, v)                                                        # FP16 [H, N_q, h_v]
        else:
            # explicit QNT+MAC+DQT+SFM (used for fp16 / e8m0 per-token recipes)
            q_i8,  s_q  = _QNTA(q,  dim=-1)                                                   # QNT @ q  (per-token)
            kT_i8, s_kT = _QNTA(kT, dim=-2)                                                   # QNT @ kT (per-col)
            v_i8,  s_v  = _QNTA(v,  dim=-2)                                                   # QNT @ v  (per-col, AV BMM)
            acc_qk = int8_mac(q_i8, kT_i8)                                                    # MAC  INT32
            scores = _DQTA(acc_qk, s_q, s_kT)                                                 # DQT  FP16
            attn   = fp16_softmax_fp32_accum(scores, dim=-1)                                   # SFM  FP16
            a_i8, s_a = _QNTA(attn, dim=-1)                                                   # QNT @ softmax (per-token)
            acc_av = int8_mac(a_i8, v_i8)                                                     # MAC  INT32
            ctx    = _DQTA(acc_av, s_a, s_v)                                                  # DQT  FP16
        ctx    = ctx.transpose(0, 1).contiguous().reshape(N_E, B, 2 * D)                      # FP16 [N_E, B, 512]
        ctx_i8, s_ctx = _QNTL(ctx.reshape(-1, 2 * D), dim=-1)                               # QNT @ BMM-out

        # (h) Out-projection — 512→512 MAC; keep first D=256 (TPE inner slice).
        W = L.self_attn_out_proj
        acc = int8_mac(ctx_i8, W.w_i8.t().contiguous())                                       # MAC  INT32 [N, 512]
        out = (_DQTL(acc, s_ctx, W.w_scale.t()) + W.bias).reshape(N_E, B, 2 * D)[..., :D]  # DQT+FPU, slice → [N_E,B,256]

        # (i) Residual + LN1, with QNT-at-production fused into the LN output.
        src_r  = src + out                                                                    # FPU residual (FP16 src local)
        src_mx = mx_fp4(src_r, block_size=32)                                                 # MXQ  FP16→FP4E2M1 block=32
        src    = F.layer_norm(src_mx, (D,), L.norm1.weight, L.norm1.bias)                     # LN   FP16
        ffn_i8, s_ffn = _QNTL(src.reshape(-1, D), dim=-1)                                   # QNT @ LN1 → linear1

        # (j) FFN linear1 (256→2048).
        W = L.linear1
        acc = int8_mac(ffn_i8, W.w_i8.t().contiguous())                                       # MAC  INT32 [N, 2048]
        y   = (_DQTL(acc, s_ffn, W.w_scale.t()) + W.bias)                                  # DQT+FPU bias → FP16 [N, 2048]
        y   = F.relu(y)                                                                       # FPU  ReLU
        y_i8, s_y = _QNTL(y, dim=-1)                                                          # QNT @ ReLU → linear2

        # (k) FFN linear2 (2048→256).
        W = L.linear2
        acc = int8_mac(y_i8, W.w_i8.t().contiguous())                                         # MAC  INT32 [N, 256]
        y   = (_DQTL(acc, s_y, W.w_scale.t()) + W.bias).reshape(N_E, B, D)                 # DQT+FPU bias

        # (l) Residual + LN2, with QNT-at-production fused into the LN output
        #     for next layer's q_c/k_c/v_c MACs (or memory_i8 after final iter).
        src_r  = src + y                                                                      # FPU residual (FP16)
        src_mx = mx_fp4(src_r, block_size=32)                                                 # MXQ
        src    = F.layer_norm(src_mx, (D,), L.norm2.weight, L.norm2.bias)                     # LN
        src_i8, s_src = _QNTL(src.reshape(-1, D), dim=-1)                                   # QNT @ LN2 → next layer (or memory for decoder)

    memory      = src                                                                         # FP16 [N_E=512, B, D=256]
    memory_i8   = src_i8                                                                      # INT8: reused by decoder ca_kcontent + ca_v (no re-QNT)
    s_memory    = s_src                                                                       # FP16 scale carried into decoder CA

    # ─────────────────────────────────────────────────────────────────────────
    # 9. Decoder preamble
    #    query_embed: nn.Embedding(N_D=10, D=256) FP16 — constant across all
    #                 6 decoder layers, so QNT once at this boundary.
    #    ref_point_head: MLP D → D → 3   (INT8dq Linears, sigmoid head)
    # ─────────────────────────────────────────────────────────────────────────
    query_embed = detr.query_embed.weight.unsqueeze(1).repeat(1, B, 1)   # FP16 [N_D=10, B, 256]
    tgt = torch.zeros_like(query_embed)                                  # FP16 [N_D, B, 256]
    qe_i8, s_qe = _QNTL(query_embed.reshape(-1, D), dim=-1)            # QNT once; reused by sa_qpos/sa_kpos every layer + ca_qpos in layer 0 + ref_point_head

    # ref_point_head MLP (2 linears, ReLU in between): 256→256 + ReLU + 256→3
    W = detr.decoder.ref_point_head.layers[0]
    acc = int8_mac(qe_i8, W.w_i8.t().contiguous())                       # MAC  INT32 [N, 256]
    rp  = (_DQTL(acc, s_qe, W.w_scale.t()) + W.bias)                  # DQT+FPU → FP16 [N, 256]
    rp  = F.relu(rp)                                                     # FPU ReLU
    rp_i8, s_rp = _QNTL(rp, dim=-1)                                    # QNT @ ReLU → next MAC
    W = detr.decoder.ref_point_head.layers[1]
    acc = int8_mac(rp_i8, W.w_i8.t().contiguous())                       # MAC  INT32 [N, 3]
    rp  = (_DQTL(acc, s_rp, W.w_scale.t()) + W.bias).reshape(N_D, B, 3)  # DQT+FPU → FP16 [N_D,B,3]
    reference_points = rp.sigmoid().transpose(0, 1)                      # FPU sigmoid → FP16 [B, N_D, 3]

    # ─────────────────────────────────────────────────────────────────────────
    # 10. Decoder ×6 — every Linear/BMM expanded inline as MAC→DQT→FPU.
    #     QNT is fused with each producing op (LN, BMM-out, softmax, ReLU,
    #     sineembed_3d, query_scale MLP).  Cross-layer-constant inputs are
    #     QNT'd ONCE outside the loop and reused:
    #       memory   → memory_i8, s_memory  (from end-of-encoder; ca_kcontent + ca_v ×6 layers)
    #       pos_embed→ pos_i8, s_pos        (from encoder boundary; ca_kpos ×6 layers)
    #       query_embed → qe_i8, s_qe       (from preamble; sa_qpos + sa_kpos ×6 + ca_qpos in layer 0)
    #     Per-layer QNT-at-production happens at:
    #       LN1 → out_i8 (for ca_qcontent), LN2 → ffn_i8 (for linear1),
    #       LN3 → out_i8 (for next iter's SA), ReLU → y_i8 (for linear2),
    #       sineembed (+optional query_scale mul) → sine_i8 (for ca_qpos_sine),
    #       BMM-out + softmax: per-attention QNT pairs (q/kT/v/attn).
    # ─────────────────────────────────────────────────────────────────────────
    output = tgt                                                          # FP16 [N_D, B, 256]
    out_i8, s_out = _QNTL(output.reshape(-1, D), dim=-1)                # QNT initial output (zeros) once; refreshed per-LN inside the loop
    intermediate = []

    for i in range(L_DEC):
        L = detr.decoder.layers[i]                                        # ConditionalTransformerDecoderLayer (i of 6)

        # (a) Reference-point sine embedding (per-layer; (b) may scale it).
        obj_center = reference_points[..., :3].transpose(0, 1)            # FP16 [N_D, B, 3]
        sine_emb   = sineembed_3d(obj_center)                             # FPU sin/cos → FP16 [N_D, B, 256]

        # (b) Query-scale TPE (layers i>0): MLP × sine_emb.
        #     query_scale.layers[0] reads the same activation as SA — reuse out_i8.
        if i > 0:
            W = detr.decoder.query_scale.layers[0]
            acc = int8_mac(out_i8, W.w_i8.t().contiguous())                                       # MAC  INT32 [N,256]
            qs  = (_DQTL(acc, s_out, W.w_scale.t()) + W.bias)                                  # DQT+FPU bias
            qs  = F.relu(qs)                                                                      # FPU ReLU
            qs2_i8, s_qs2 = _QNTL(qs, dim=-1)                                                   # QNT @ ReLU → next MAC
            W = detr.decoder.query_scale.layers[1]
            acc = int8_mac(qs2_i8, W.w_i8.t().contiguous())                                       # MAC
            pos_xform = (_DQTL(acc, s_qs2, W.w_scale.t()) + W.bias).reshape(N_D, B, D)         # DQT+FPU bias
            sine_emb = sine_emb * pos_xform                                                       # FPU mul

        # QNT-at-production: sineembed_3d (+ optional pos_xform mul) feeds ca_qpos_sine.
        sine_i8, s_sine = _QNTL(sine_emb.reshape(-1, D), dim=-1)                                # QNT @ sineembed

        # ── (c) Self-attention ───────────────────────────────────────────────
        # out_i8 from prev iter's LN3 (or initial QNT before loop); qe_i8 from preamble (constant).
        W = L.sa_qcontent_proj
        acc = int8_mac(out_i8, W.w_i8.t().contiguous())                                           # MAC
        qc_sa = (_DQTL(acc, s_out, W.w_scale.t()) + W.bias).reshape(N_D, B, D)                 # DQT+FPU

        W = L.sa_kcontent_proj
        acc = int8_mac(out_i8, W.w_i8.t().contiguous())                                           # MAC
        kc_sa = (_DQTL(acc, s_out, W.w_scale.t()) + W.bias).reshape(N_D, B, D)                 # DQT+FPU

        W = L.sa_v_proj
        acc = int8_mac(out_i8, W.w_i8.t().contiguous())                                           # MAC
        v_sa_pre = (_DQTL(acc, s_out, W.w_scale.t()) + W.bias).reshape(N_D, B, D)              # DQT+FPU

        W = L.sa_qpos_proj
        acc = int8_mac(qe_i8, W.w_i8.t().contiguous())                                            # MAC
        qp_sa = (_DQTL(acc, s_qe, W.w_scale.t()) + W.bias).reshape(N_D, B, D)                  # DQT+FPU

        W = L.sa_kpos_proj
        acc = int8_mac(qe_i8, W.w_i8.t().contiguous())                                            # MAC
        kp_sa = (_DQTL(acc, s_qe, W.w_scale.t()) + W.bias).reshape(N_D, B, D)                  # DQT+FPU

        # Q/K assembly (FPU adds) + pre-scale on Q
        q_sa = (qc_sa + qp_sa) * (H_SA ** -0.5)                                                   # FPU add + scale
        k_sa = kc_sa + kp_sa                                                                      # FPU add

        # Multi-head reshape + QNT-at-production for SA BMM operands.
        q_sa  = q_sa.view(N_D, B * H, H_SA).transpose(0, 1).contiguous()                          # FP16 [B·H, 10, 64]
        k_sa  = k_sa.view(N_D, B * H, H_SA).transpose(0, 1).contiguous()                          # FP16 [B·H, 10, 64]
        v_sa  = v_sa_pre.view(N_D, B * H, H_SA).transpose(0, 1).contiguous()                      # FP16 [B·H, 10, 64]
        kT_sa = k_sa.transpose(1, 2).contiguous()                                                 # FP16 [B·H, 64, 10]
        # SA score BMM: Q·Kᵀ → [B·H,10,10]
        if _FLASH_ATTN is not None:
            ctx_sa = _FLASH_ATTN(q_sa, k_sa, v_sa)                                               # FP16 [H, N_d, h_v]
        else:
            # explicit QNT+MAC+DQT+SFM (used for fp16 / e8m0 per-token recipes)
            qsa_i8,  s_qsa  = _QNTA(q_sa,  dim=-1)                                               # QNT @ q  (per-token)
            ksaT_i8, s_ksaT = _QNTA(kT_sa, dim=-2)                                               # QNT @ kT (per-col)
            vsa_i8,  s_vsa  = _QNTA(v_sa,  dim=-2)                                               # QNT @ v  (per-col)
            acc_sa_qk = int8_mac(qsa_i8, ksaT_i8)                                                # MAC
            scores_sa = _DQTA(acc_sa_qk, s_qsa, s_ksaT)                                          # DQT
            attn_sa   = fp16_softmax_fp32_accum(scores_sa, dim=-1)                                # SFM
            asa_i8, s_asa = _QNTA(attn_sa, dim=-1)                                               # QNT @ softmax
            acc_sa_av = int8_mac(asa_i8, vsa_i8)                                                  # MAC
            ctx_sa    = _DQTA(acc_sa_av, s_asa, s_vsa)                                            # DQT
        ctx_sa    = ctx_sa.transpose(0, 1).contiguous().reshape(N_D, B, D)                        # FP16 [N_D, B, 256]
        ctxsa_i8, s_ctxsa = _QNTL(ctx_sa.reshape(-1, D), dim=-1)                                # QNT @ BMM-out

        # SA out-proj: 256→256 MAC
        W = L.self_attn.out_proj
        acc = int8_mac(ctxsa_i8, W.w_i8.t().contiguous())                                         # MAC
        sa_out = (_DQTL(acc, s_ctxsa, W.w_scale.t()) + W.bias).reshape(N_D, B, D)              # DQT+FPU

        # (d) Residual + LN1, with QNT-at-production fused into the LN output for ca_qcontent.
        out_r  = output + sa_out                                                                  # FPU residual (FP16)
        out_mx = mx_fp4(out_r, block_size=32)                                                     # MXQ
        output = F.layer_norm(out_mx, (D,), L.norm1.weight, L.norm1.bias)                         # LN
        out_i8, s_out = _QNTL(output.reshape(-1, D), dim=-1)                                    # QNT @ LN1 → ca_qcontent

        # ── (e) Cross-attention ──────────────────────────────────────────────
        # memory_i8 (from end-of-encoder), pos_i8 (from encoder boundary),
        # sine_i8 (from (a)/(b)), qe_i8 (from preamble) all reused.  Only
        # out_i8 was just refreshed by LN1.
        W = L.ca_qcontent_proj
        acc = int8_mac(out_i8, W.w_i8.t().contiguous())                                           # MAC
        qc_ca = (_DQTL(acc, s_out, W.w_scale.t()) + W.bias).reshape(N_D, B, D)                 # DQT+FPU

        W = L.ca_kcontent_proj
        acc = int8_mac(memory_i8, W.w_i8.t().contiguous())                                        # MAC
        kc_ca = (_DQTL(acc, s_memory, W.w_scale.t()) + W.bias).reshape(N_E, B, D)              # DQT+FPU

        W = L.ca_v_proj
        acc = int8_mac(memory_i8, W.w_i8.t().contiguous())                                        # MAC
        v_ca_pre = (_DQTL(acc, s_memory, W.w_scale.t()) + W.bias).reshape(N_E, B, D)           # DQT+FPU

        W = L.ca_kpos_proj
        acc = int8_mac(pos_i8, W.w_i8.t().contiguous())                                           # MAC
        kp_ca = (_DQTL(acc, s_pos, W.w_scale.t()) + W.bias).reshape(N_E, B, D)                 # DQT+FPU

        W = L.ca_qpos_sine_proj
        acc = int8_mac(sine_i8, W.w_i8.t().contiguous())                                          # MAC
        q_sine = (_DQTL(acc, s_sine, W.w_scale.t()) + W.bias).reshape(N_D, B, D)               # DQT+FPU

        if i == 0:
            # Layer 0 only: ca_qpos_proj on query_embed (reuses qe_i8 from preamble).
            W = L.ca_qpos_proj
            acc = int8_mac(qe_i8, W.w_i8.t().contiguous())                                        # MAC
            qp_ca = (_DQTL(acc, s_qe, W.w_scale.t()) + W.bias).reshape(N_D, B, D)              # DQT+FPU
            q_ca_base = qc_ca + qp_ca                                                             # FPU add
            k_ca_base = kc_ca + kp_ca                                                             # FPU add (is_first only)
        else:
            q_ca_base = qc_ca                                                                     # FP16 [N_D, B, 256]
            k_ca_base = kc_ca                                                                     # no k_pos for non-first layers

        # 2d-wide TPE concat (content || pos-feature) on Q and K → H_CA=128 per head
        q_ca = torch.cat([q_ca_base.view(N_D, B, H, H_SA),
                          q_sine.view(N_D, B, H, H_SA)], dim=3).reshape(N_D, B, 2 * D)            # FP16 [N_D,B,512]
        k_ca = torch.cat([k_ca_base.view(N_E, B, H, H_SA),
                          kp_ca.view(N_E, B, H, H_SA)],  dim=3).reshape(N_E, B, 2 * D)            # FP16 [N_E,B,512]

        q_ca = q_ca * (H_CA ** -0.5)                                                              # FPU scale
        # Multi-head reshape + QNT-at-production for CA BMM operands.
        q_ca  = q_ca.view(N_D, B * H, H_CA).transpose(0, 1).contiguous()                          # FP16 [B·H, 10, 128]
        k_ca  = k_ca.view(N_E, B * H, H_CA).transpose(0, 1).contiguous()                          # FP16 [B·H, 512, 128]
        v_ca  = v_ca_pre.view(N_E, B * H, H_SA).transpose(0, 1).contiguous()                      # FP16 [B·H, 512, 64]
        kT_ca = k_ca.transpose(1, 2).contiguous()                                                 # FP16 [B·H, 128, 512]
        # CA score BMM: Q·Kᵀ → [B·H,10,512]
        if _FLASH_ATTN is not None:
            ctx_ca = _FLASH_ATTN(q_ca, k_ca, v_ca)                                               # FP16 [H, N_d, h_v]
        else:
            # explicit QNT+MAC+DQT+SFM (used for fp16 / e8m0 per-token recipes)
            qca_i8,  s_qca  = _QNTA(q_ca,  dim=-1)                                               # QNT @ q  (per-token)
            kcaT_i8, s_kcaT = _QNTA(kT_ca, dim=-2)                                               # QNT @ kT (per-col)
            vca_i8,  s_vca  = _QNTA(v_ca,  dim=-2)                                               # QNT @ v  (per-col)
            acc_ca_qk = int8_mac(qca_i8, kcaT_i8)                                                # MAC
            scores_ca = _DQTA(acc_ca_qk, s_qca, s_kcaT)                                          # DQT
            attn_ca   = fp16_softmax_fp32_accum(scores_ca, dim=-1)                                # SFM
            aca_i8, s_aca = _QNTA(attn_ca, dim=-1)                                               # QNT @ softmax
            acc_ca_av = int8_mac(aca_i8, vca_i8)                                                  # MAC
            ctx_ca    = _DQTA(acc_ca_av, s_aca, s_vca)                                            # DQT
        ctx_ca    = ctx_ca.transpose(0, 1).contiguous().reshape(N_D, B, D)                        # FP16 [N_D, B, 256]
        ctxca_i8, s_ctxca = _QNTL(ctx_ca.reshape(-1, D), dim=-1)                                # QNT @ BMM-out

        # CA out-proj: 256→256 MAC (cross_attn.out_proj)
        W = L.cross_attn.out_proj
        acc = int8_mac(ctxca_i8, W.w_i8.t().contiguous())                                         # MAC
        ca_out = (_DQTL(acc, s_ctxca, W.w_scale.t()) + W.bias).reshape(N_D, B, D)              # DQT+FPU

        # (f) Residual + LN2, with QNT-at-production fused into the LN output for FFN linear1.
        out_r  = output + ca_out                                                                  # FPU residual (FP16)
        out_mx = mx_fp4(out_r, block_size=32)                                                     # MXQ
        output = F.layer_norm(out_mx, (D,), L.norm2.weight, L.norm2.bias)                         # LN
        ffn_i8, s_ffn = _QNTL(output.reshape(-1, D), dim=-1)                                    # QNT @ LN2 → linear1

        # (g) FFN: linear1 (256→2048) + ReLU + linear2 (2048→256)
        W = L.linear1
        acc = int8_mac(ffn_i8, W.w_i8.t().contiguous())                                           # MAC
        y   = (_DQTL(acc, s_ffn, W.w_scale.t()) + W.bias)                                      # DQT+FPU
        y   = F.relu(y)                                                                           # FPU ReLU
        y_i8, s_y = _QNTL(y, dim=-1)                                                              # QNT @ ReLU → linear2
        W = L.linear2
        acc = int8_mac(y_i8, W.w_i8.t().contiguous())                                             # MAC
        y   = (_DQTL(acc, s_y, W.w_scale.t()) + W.bias).reshape(N_D, B, D)                     # DQT+FPU

        # (h) Residual + LN3, with QNT-at-production fused into the LN output for next iter's SA.
        out_r  = output + y                                                                       # FPU residual (FP16)
        out_mx = mx_fp4(out_r, block_size=32)                                                     # MXQ
        output = F.layer_norm(out_mx, (D,), L.norm3.weight, L.norm3.bias)                         # LN
        out_i8, s_out = _QNTL(output.reshape(-1, D), dim=-1)                                    # QNT @ LN3 → next iter's SA (sa_qcontent/sa_kcontent/sa_v + query_scale)

        # (i) Final decoder norm (MXQ + LN) — FP16 path to heads/box-affine.
        inter_mx = mx_fp4(output, block_size=32)                                                  # MXQ
        inter    = F.layer_norm(inter_mx, (D,),
                                detr.decoder.norm.weight, detr.decoder.norm.bias)                 # LN
        intermediate.append(inter)                                                                # FP16 [N_D, B, 256]

    hs = torch.stack(intermediate).transpose(1, 2)                                                # FP16 [L_DEC=6, B, N_D=10, D=256]

    # ─────────────────────────────────────────────────────────────────────────
    # 11. Heads (X, FP16)
    #     class_embed: Linear(D → n_classes+1) FP16
    #     bbox_embed : MLP(D → D → D → 6) FP16 (3 Linear layers + ReLU)
    #     reference_points are added to the first pos_dim box coords (inverse_sigmoid domain)
    # ─────────────────────────────────────────────────────────────────────────
    ref_pre_sig = inverse_sigmoid(reference_points)                       # FP16 [B, N_D, 3]  FPU log
    outputs_coords = []
    for lvl in range(hs.shape[0]):
        tmp = detr.bbox_embed(hs[lvl])                                    # FP16 [B, N_D, 6]  3× Linear FP16
        tmp[..., :detr.pos_dim] = tmp[..., :detr.pos_dim] + ref_pre_sig   # FP16  FPU
        outputs_coords.append(tmp.sigmoid())                              # FP16  FPU
    outputs_coord = torch.stack(outputs_coords)                           # FP16 [L_DEC, B, N_D, 6]
    outputs_class = detr.class_embed(hs)                                  # FP16 [L_DEC, B, N_D, n_classes+1]

    pred_logits = outputs_class[-1]                                       # FP16 [B, N_D, n_classes+1]
    pred_boxes  = outputs_coord[-1]                                       # FP16 [B, N_D, 6]

    # ─────────────────────────────────────────────────────────────────────────
    # 12. Plane decomposition + box_affine_transformer (X, FP16)
    # ─────────────────────────────────────────────────────────────────────────
    bbox = box_ops.box_cxcyczwhd_to_xyzxyz(pred_boxes.reshape(-1, 6)).reshape(pred_boxes.shape)  # FP16 [B, N_D, 6]
    hbox = torch.stack([b[:, [0, 2, 3, 5]] for b in bbox])                # FP16 [B, N_D, 4]
    vbox = torch.stack([b[:, [1, 2, 4, 5]] for b in bbox])                # FP16 [B, N_D, 4]
    _, ibox = detr.calc_v_props(hbox, alignment=False, v_props=vbox, normed=True)
    ibox = torch.stack(ibox)                                              # FP16 [B, N_D, 4]
    pred_hboxes = box_ops.box_xyxy_to_cxcywh(hbox)                        # FP16
    pred_vboxes = box_ops.box_xyxy_to_cxcywh(vbox)                        # FP16
    iboxc       = box_ops.box_xyxy_to_cxcywh(ibox)                        # FP16

    bat_in  = torch.cat((iboxc, pred_boxes), dim=-1).view(-1, detr.num_queries)        # FP16 [B*N_D, 10]
    bat_out = detr.box_affine_transformer(bat_in).view(-1, detr.num_queries, 4)        # FP16 [B, N_D, 4]   Linear+BN1d+LReLU×2+Linear
    pred_iboxes = torch.sigmoid(iboxc - bat_out) + 1e-5                                # FP16 [B, N_D, 4]

    # ─────────────────────────────────────────────────────────────────────────
    # 13. Segmentation head (CGRA FP16 — bbox_attention, mask_head, unet)
    #     Mirrors DETRsegm.forward() seg block.  All ops are CGRA FP16:
    #       MHAttentionMap : q_linear (FPU Linear) + k_linear as 1×1 conv (FPU)
    #                        + multi-head einsum + SFM → bbox_mask
    #       MaskHeadSmallConv : Conv2d + GroupNorm cascade + FPN skip-adds
    #       bilinear upsample to (112, 112)
    #       Unet : encoder/decoder Conv2d + BatchNorm2d + ConvTranspose2d
    #     FPN inputs: feats_v[2] (coarsest 8×16) → feats_v[1] (16×32) → feats_v[0] (finest 32×64)
    #     These are the intermediate backbone features from the vertical view.
    # ─────────────────────────────────────────────────────────────────────────
    segm = retr.model if hasattr(retr.model, "bbox_attention") else None
    mask_logits = None
    if segm is not None:
        # (a) MHAttentionMap — multi-head attention: decoder queries × vertical feature map
        q_ba   = hs[-1]                                                              # FP16 [B, N_D, 256]
        q_proj = segm.bbox_attention.q_linear(q_ba)                                 # FPU Linear → FP16 [B, N_D, 256]
        k_feat = F.conv2d(                                                           # FPU 1×1 conv (k_linear recast)
            src_proj_v,
            segm.bbox_attention.k_linear.weight.unsqueeze(-1).unsqueeze(-1),
            segm.bbox_attention.k_linear.bias,
        )                                                                            # FP16 [B, 256, H, W]
        n_heads  = segm.bbox_attention.num_heads                                     # 4
        head_dim = segm.bbox_attention.hidden_dim // n_heads                         # 64
        H_feat, W_feat = k_feat.shape[-2:]
        qh = q_proj.view(B, N_D, n_heads, head_dim)                                 # FP16 [B, N_D, 4, 64]
        kh = k_feat.view(B, n_heads, head_dim, H_feat, W_feat)                      # FP16 [B, 4, 64, H, W]
        nf = float(head_dim) ** -0.5                                                 # 0.125
        weights   = torch.einsum("bqnc,bnchw->bqnhw", qh * nf, kh)                 # FPU einsum → FP16 [B, N_D, 4, H, W]
        bbox_mask = fp16_softmax_fp32_accum(weights.flatten(2), dim=-1).view(       # SFM FP16 [B, N_D, 4, H, W]
            weights.size()
        )

        # (b) MaskHeadSmallConv — Conv2d + GroupNorm cascade with FPN-level skip adds.
        #     fpns ordered coarsest→finest to match mask_head adapter routing:
        #       fpns[0] → adapter1 (64→256), fpns[1] → adapter2 (64→128), fpns[2] → adapter3 (64→64)
        fpns      = [feats_v[2], feats_v[1], feats_v[0]]   # FP16 [B, 64, ·, ·] (plain tensors, not NestedTensors)
        mask_feat = segm.mask_head(src_proj_h, bbox_mask, fpns)                     # FP16 [B·N_D, 32, H', W']

        # (c) Bilinear upsample to (112, 112) then Unet encoder–decoder.
        mask_feat  = F.interpolate(mask_feat, size=(112, 112), mode="bilinear")     # FP16 [B·N_D, 32, 112, 112]
        mask_logits = segm.unet(mask_feat)                                          # FP16 [B·N_D, 1, 112, 112]

    # ─────────────────────────────────────────────────────────────────────────
    # 14. Output dict (mirrors DETR.forward) + pixel-scaled augmented boxes
    # ─────────────────────────────────────────────────────────────────────────
    out = {
        "pred_logits":  pred_logits,
        "pred_boxes":   pred_boxes,
        "pred_hboxes":  pred_hboxes,
        "pred_vboxes":  pred_vboxes,
        "pred_iboxes":  pred_iboxes,
        "hs":           hs,
        "memory":       memory,
        "reference":    reference_points,
    }
    if mask_logits is not None:
        out["mask_logits"] = mask_logits                                             # FP16 [B·N_D, 1, 112, 112]
    def _cxcywh_to_xyxy_scaled(t, w_scale, h_scale):
        t = t.clone()
        xy = box_ops.box_cxcywh_to_xyxy(t.reshape(-1, 4)).reshape(t.shape)
        xy[:, :, [0, 2]] *= w_scale
        xy[:, :, [1, 3]] *= h_scale
        return xy
    out["pred_hboxes_aug"] = _cxcywh_to_xyxy_scaled(pred_hboxes, detr.w, detr.h)
    out["pred_vboxes_aug"] = _cxcywh_to_xyxy_scaled(pred_vboxes, detr.w, detr.h)
    out["proj_boxes"]      = _cxcywh_to_xyxy_scaled(pred_iboxes, detr.iw, detr.ih)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# POST-PROCESSING & EVAL
# ═════════════════════════════════════════════════════════════════════════════

def retr_postprocess_det(out: dict, thresh_mask: float, retr=None) -> list:
    """Threshold per-query scores and assemble per-sample prediction dicts.

    Mirrors the post-processing block in RETR.forward (retr.py:260-285).
    Outputs FP32 tensors because torchmetrics.MeanAveragePrecision requires FP32.
    When `retr` is provided and out["mask_logits"] exists, also assembles
    per-sample binary segmentation masks via maskrcnn_inference + paste_masks_in_image.
    """
    scores_all = out["pred_logits"].sigmoid()[..., 0]        # FP16 [B, N_D]
    iboxes = out["proj_boxes"]                               # FP16 [B, N_D, 4]  (image-plane xyxy, pixel-scaled)
    hboxes = out["pred_hboxes_aug"]                          # FP16 [B, N_D, 4]  (horizontal radar xyxy, pixel-scaled)
    vboxes = out["pred_vboxes_aug"]                          # FP16 [B, N_D, 4]  (vertical radar xyxy, pixel-scaled)

    has_seg = retr is not None and out.get("mask_logits") is not None
    if has_seg:
        B_s = scores_all.shape[0]
        # maskrcnn_inference splits [B*N_D, 1, H, W] → list of [N_D, 1, H, W], applies sigmoid
        dummy_labels = [torch.zeros(N_D, dtype=torch.long, device=scores_all.device)
                        for _ in range(B_s)]
        masks_probs = maskrcnn_inference(out["mask_logits"], dummy_labels)          # list [N_D, 1, 112, 112]

    preds = []
    for b in range(scores_all.shape[0]):
        s = scores_all[b].float()
        keep = s > thresh_mask
        n = int(keep.sum())
        pred = {
            "iboxes": iboxes[b][keep].float(),
            "scores": s[keep],
            "labels": torch.zeros(n, dtype=torch.long, device=s.device),
            "hboxes": hboxes[b][keep].float(),
            "vboxes": vboxes[b][keep].float(),
        }
        if has_seg:
            pred_masks = masks_probs[b][keep]                                       # [N_kept, 1, 112, 112]
            boxes_pix  = iboxes[b][keep].float()                                    # [N_kept, 4] pixel xyxy
            pasted = retr.paste_masks_in_image(pred_masks, boxes_pix, (retr.ih, retr.iw))
            # pasted: [N_kept, 1, ih, iw] — union of all detected-object masks → binary [ih, iw]
            final_mask = torch.zeros(retr.ih, retr.iw, device=s.device, dtype=torch.long)
            for cur in pasted:
                final_mask[cur[0] > thresh_mask] = 1
            pred["masks"] = final_mask
        preds.append(pred)
    return preds


def run_eval(retr, detr, device: torch.device, args) -> dict:
    """Iterate the MMVR test split, run the ASIC simulator, accumulate AP+IoU metrics.

    Returns the populated `metrics.res` dict (image-plane AP, radar-plane AP, seg IoU).
    Pass --task DETSEG to enable the segmentation head and IoU computation.
    """
    from tqdm import tqdm

    seg = (args.task == "DETSEG") and hasattr(retr.model, "bbox_attention")

    dataset_path = Path(args.root) / args.split[:2]
    _, _, test_loader = get_dataloader(
        MMVRDetSeg,
        dataset_path,
        split=args.split,
        batch_size=args.batch_size,
        collate_fn=collate_det_seg,
        num_workers=args.workers,
    )

    metrics = Metrics(seg=seg).to(device)
    n_seen = 0
    for batch in tqdm(test_loader, desc=f"eval {args.split} {'DETSEG' if seg else 'DET'}"):
        batch = move_to_device(batch, device)
        hor = batch["hm_hori"].detach().to(torch.float16)
        ver = batch["hm_vert"].detach().to(torch.float16)
        labels = batch["labels"]

        out_dict = dataflow_forward(retr, detr, hor, ver)
        preds = retr_postprocess_det(out_dict, thresh_mask=args.thresh_mask,
                                     retr=retr if seg else None)

        metrics.compute(labels, preds)

        n_seen += hor.shape[0]
        if args.max_samples is not None and n_seen >= args.max_samples:
            break

    metrics.get_result()
    return metrics.res


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained",
                    default=str(_REPO / "logs/pretrained_model/p2s1_retr_detseg.pth"))
    ap.add_argument("--task", default="DET", choices=["DET", "DETSEG"])
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--batch", type=int, default=1, help="smoke-test batch size")
    ap.add_argument("--h-w", nargs=2, type=int, default=[128, 256])
    ap.add_argument("--save-out", default=None, help="smoke-test: save out dict to this path")
    ap.add_argument("--seed", type=int, default=42)
    # ── eval-mode flags ─────────────────────────────────────────────────────
    ap.add_argument("--eval", action="store_true",
                    help="run over the MMVR test split and report AP metrics")
    ap.add_argument("--root", default=str(_REPO / "MMVR/segment_4_3"),
                    help="MMVR dataset root (contains P2/...)")
    ap.add_argument("--split", default="P2S1", choices=["P2S1", "P2S2"])
    ap.add_argument("--batch-size", type=int, default=8, dest="batch_size",
                    help="eval batch size")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-samples", type=int, default=None, dest="max_samples",
                    help="cap eval at N samples (for quick checks)")
    ap.add_argument("--thresh-mask", type=float, default=0.5, dest="thresh_mask",
                    help="sigmoid-score threshold to keep a query")
    # ── Scale-format experiment flags ───────────────────────────────────────
    ap.add_argument("--lin-scale-fmt", default="e8m0", choices=["fp16", "e8m0"],
                    dest="lin_scale_fmt",
                    help="scale format for linear-layer QNT/DQT (fp16|e8m0); canonical: e8m0")
    ap.add_argument("--attn-scale-fmt", default="blockk128",
                    choices=["fp16", "e8m0", "blockk16", "blockk32", "blockk64", "blockk128"],
                    dest="attn_scale_fmt",
                    help="scale format for attention BMMs; canonical: blockk128")
    args = ap.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # ── Wire up scale-dispatch globals ──────────────────────────────────────
    global _QNTL, _DQTL, _QNTA, _DQTA, _FLASH_ATTN, _BLK_Q, _BLK_K
    if args.lin_scale_fmt == "e8m0":
        _QNTL = lambda x, dim: qnt_act_e8m0(x, dim)
        _DQTL = dqt_e8m0_lin
    # else: keep fp16 defaults
    if args.attn_scale_fmt == "e8m0":
        _QNTA = lambda x, dim: qnt_act_e8m0(x, dim)
        _DQTA = dqt_e8m0_attn
        _FLASH_ATTN = None
    elif args.attn_scale_fmt.startswith("blockk"):
        bs = int(args.attn_scale_fmt[len("blockk"):])
        _BLK_K = bs
        _FLASH_ATTN = lambda q, k, v, _bk=bs: flash_attn_e8m0_tiled(q, k, v, blk_q=_BLK_Q, blk_k=_bk)
    # else fp16: keep defaults (_FLASH_ATTN = None, _QNTA/_DQTA = fp16)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Build RETR + load pretrained checkpoint.
    #    The shipped checkpoint is DETSEG (keys rooted at model.detr.*); we
    #    build in SEG mode to match, and unwrap DETR from DETRsegm below.
    # ─────────────────────────────────────────────────────────────────────────
    retr = RETR(task="SEG").to(device).eval()
    sd = torch.load(args.pretrained, map_location=device, weights_only=False)
    sd = migrate_encoder_mha_state_dict(sd)
    retr.load_state_dict(sd)
    detr = retr.model.detr if hasattr(retr.model, "detr") else retr.model

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Apply the ASIC recipe (once; amortized across all samples in --eval)
    # ─────────────────────────────────────────────────────────────────────────
    fp16_cast_cgra(retr)                          # CGRA: backbone + input_proj + heads → FP16
    quantize_transformer_inplace(detr)            # CIM : encoder/decoder Linears → INT8dq, LN → FP16+MXQ

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Dispatch: MMVR eval vs. one-shot smoke test
    # ─────────────────────────────────────────────────────────────────────────
    if args.eval:
        res = run_eval(retr, detr, device, args)
        # ── Save JSON result ─────────────────────────────────────────────────
        import json, datetime
        tag = f"lin-{args.lin_scale_fmt}_attn-{args.attn_scale_fmt}"
        samples_tag = f"_n{args.max_samples}" if args.max_samples else ""
        out_name = f"{args.split}_dflow_{tag}{samples_tag}.json"
        out_path = _REPO / "experiments" / "results" / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        def _to_serializable(v):
            if isinstance(v, torch.Tensor): return v.item() if v.numel() == 1 else v.tolist()
            if isinstance(v, dict): return {kk: _to_serializable(vv) for kk, vv in v.items()}
            return v
        payload = {
            "run": out_name,
            "split": args.split,
            "task": args.task,
            "lin_scale_fmt": args.lin_scale_fmt,
            "attn_scale_fmt": args.attn_scale_fmt,
            "max_samples": args.max_samples,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            **{k: _to_serializable(v) for k, v in res.items()},
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved → {out_path}")
        return

    B = args.batch
    rawH, rawW = args.h_w
    hor = torch.randn(B, 4, rawH, rawW, device=device, dtype=torch.float16)   # FP16 [B,4,H,W]
    ver = torch.randn(B, 4, rawH, rawW, device=device, dtype=torch.float16)   # FP16 [B,4,H,W]

    out = dataflow_forward(retr, detr, hor, ver)

    # ── Report ───────────────────────────────────────────────────────────────
    print(f"device={device}  task={args.task}  B={B}  in=(4,{rawH},{rawW})")
    for k, t in out.items():
        if isinstance(t, torch.Tensor):
            print(f"  {k:18s}  shape={tuple(t.shape)}  dtype={str(t.dtype).split('.')[-1]}  "
                  f"|x|max={t.abs().max().item():.4f}")

    if args.save_out is not None:
        torch.save({k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in out.items()}, args.save_out)
        print(f"\nsaved → {args.save_out}")

    print("\nOK — DATAFLOW.py end-to-end forward complete.")


if __name__ == "__main__":
    with torch.no_grad():
        main()
