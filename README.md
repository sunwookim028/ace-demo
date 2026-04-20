<!--
Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->
# [RETR: Multi-View Radar Detection Transformer for Indoor Perception [NeurIPS2024]](https://arxiv.org/abs/2411.10293)

PyTorch training and evaluation code for **RETR** (**R**adar d**E**tection **TR**ansformer).

RETR inherits the advantages of DETR, eliminating the need for hand-crafted components for object detection and segmentation in the image plane. RETR incorporates carefully designed modifications:
1) depth-prioritized feature similarity via a tunable positional encoding (TPE);
2) a tri-plane loss from both radar and camera coordinates;
3) using a calibrated or learnable radar-to-camera transformation via reparameterization, to account for the unique multi-view radar setting.
<table style="margin-left:auto;margin-right:auto;">
  <tr>
    <td style="text-align:center;">
      <img src="figs/retr.png" alt="RETR" width="900"/>
    </td>
  </tr>
</table>

## Installation

Follow the steps below
```commandline
conda create -n retr python=3.10
conda activate retr
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Download MMVR dataset

1. Download the MMVR dataset from [Zenodo](https://zenodo.org/records/12611978).
   - You should see the following four zip files:
     - `P1.zip`
     - `P2_00.zip`
     - `P2_01.zip`
     - `P2_02.zip`
2. Create a directory for the MMVR dataset:
   ```bash
   mkdir ./MMVR/
3. Unzip P1.zip as:
   ```bash
   unzip P1.zip "d1s1/*" "d1s2/*" "d2s2/*" "d3s1/*" "d3s2/*" "d4s1/*" -d ./MMVR/P1
4. Unzip P2_00.zip, P2_01.zip, P2_02.zip as
   ```bash
   unzip P2_00.zip -d ./MMVR/P2
   unzip P2_01.zip -d ./MMVR/P2
   unzip P2_02.zip -d ./MMVR/P2
5. The MMVR directory should have the following folder structure
    ```
    MMVR/
    ├── P1/
    │   ├── d1s1/
    │   │   ├── 000/
    │   │   |   ├──00000_meta.npz
    │   │   |   ├──00000_radar.npz
    │   │   |   ├──00000_bbox.npz
    │   │   |   ├──00000_pose.npz
    │   │   |   ├──00000_mask.npz
    │   │   |   .
    │   │   ├── 001/
    │   │   .
    │   .
    │   └── d4s1/
    └── P2/
        ├── d5s1/
        .
        └── d9s6/
    ```

## Data Preparation

RETR processes radar data by grouping a specific number of radar frames into segments. These segments are then used for model training and evaluation. This section provides instructions to create the segmented/grouped MMVR data from the above unziped MMVR folder.

1. Navigate to the `src/data` directory:
   ```bash
   cd ./src/data
2. Run the create_grouped_dataset.py script with the desired parameters:
    ```commandline
    python create_grouped_dataset.py --num_frames 4 --overlap 3 --dataset_dir ./MMVR --output ./MMVR
    ```
    ### Parameters
    - **`--num_frames`**
      Number of radar frames to combine into a single segment.
      *Example:* `--num_frames 4`

    - **`--overlap`**
      Number of overlapping frames between consecutive segments.
      *Example:* `--overlap 3`

    - **`--dataset_dir`**
      Path to the directory containing the unzipped MMVR dataset.
      *Example:* `--dataset_dir ./MMVR`

    - **`--output`**
      Path to the directory where the segmented/grouped dataset will be saved.
      *Example:* `--output ./MMVR/`

    The output directory will contain the segmented/grouped radar data in the format of `segment_{num_frames}_{overlap}`.

    In the above example, the unzipped MMVR directory will add a new folder `segment_4_3` under `./MMVR`.

## Quick Demo

Refer to `src/demo.ipynb`.

## Train
1. Navigate to the `src` directory:
   ```bash
   cd ./src
2. First, train the detection model using the grouped/segmented dataset:
    ```commandline
    python train.py --root ./MMVR/segment_4_3 --split P2S1 --task DET
    ```
    ### Parameters
    - **`--root`**
      The directory for grouped/segmented MMVR dataset. *Example:* `--root ./MMVR/segment_4_3`

    - **`--split`**
      Data protocol and data split in MMVR. [`P2S1` or `P2S2`].
      *Example:* `--split P2S1`

    - **`--task`**
      Detection task first.
      *Example:* `--task DET`

    You can specify training parameters such as batch size, learning rate, number of epochs, and number of workers using command-line arguments. Refer to `python train.py -h` for details.

    During training, log files and checkpoints are  automatically saved under the directory `../logs/refined/mmvr/[P2S1 or P2S2]/DET/YYYYmmdd_HHMMSS`, where `[P2S1 or P2S2]` corresponds to `--split` used for the training, and `YYYYmmdd_HHMMS` represents the timestamp of the training session.

3. Then, train the segmentation model using the trained detection model with frozen weights:
    ```commandline
    python train.py --root ./MMVR/segment_4_3 --split P2S1 --task SEG --det_path ../logs/refined/mmvr/[P2S1 or P2S2]/DET/YYYYmmdd_HHMMSS/best.pth
    ```
    ### Parameters
    - **`--root`**
      The directory for grouped/segmented MMVR dataset.
      *Example:* `--root ./MMVR/segment_4_3`

    - **`--split`**
      Data protocol and data split in MMVR. [`P2S1` or `P2S2`].
      *Example:* `--split P2S1`

    - **`--task`**
      now Segmentation task.
      *Example:* `--task SEG`

    - **`--det_path`**
      the best detection model checkpoint.
      *Example:* `--det_path ../logs/.../best.pth`

    During training, log files and segmentation checkpoints are automatically saved under the directory `../logs/refined/mmvr/[P2S1 or P2S2]/DETSEG/YYYYmmdd_HHMMSS`.


## Evaluation
1. Download pretrained RETR models from the links in the table below or directly from the GitHub repo `./logs/pretrained_model/[p2s1/p2s2]_retr_detseg.pth`.

   <table>
     <thead>
       <tr style="text-align: center;">
         <th>Dataset</th>
         <th>Split</th>
         <th>Backbone</th>
         <th>BBox AP</th>
         <th>BBox AR1</th>
         <th>Segm IoU</th>
         <th>Pretrained model link</th>
         <th>Size</th>
       </tr>
     </thead>
     <tbody>
       <tr style="text-align: center;">
         <td>MMVR</td>
         <td>P2S1</td>
         <td>ResNet18</td>
         <td>46.75</td>
         <td>42.19</td>
         <td>77.21</td>
         <td> <a href="https://github.com/merl-internal/retr_release/raw/refs/heads/main/logs/pretrained_model/p2s1_retr_detseg.pth">p2s1_retr_detseg.pth</a> </td>
         <td>156Mb</td>
       </tr>
       <tr style="text-align: center;">
         <td>MMVR</td>
         <td>P2S2</td>
         <td>ResNet18</td>
         <td>12.19</td>
         <td>19.70</td>
         <td>59.93</td>
         <td> <a href="https://github.com/merl-internal/retr_release/raw/refs/heads/main/logs/pretrained_model/p2s2_retr_detseg.pth">p2s2_retr_detseg.pth</a> </td>
         <td>156Mb</td>
       </tr>
     </tbody>
   </table>
2. Navigate to the `src` directory:
   ```bash
   cd ./src
3. Evaluate both Detection and Segmentation performance
    ```commandline
    python test.py --root ./MMVR/segment_4_3 --split [P2S1/P2S2] --task DETSEG --pretrained_path ../logs/pretrained_model/[p2s1/p2s2]_retr_detseg.pth
    ```
    ### Parameters
    - **`--root`**
      The directory for grouped/segmented MMVR dataset.
      *Example:* `--root ./MMVR/segment_4_3`

    - **`--split`**
      Data protocol and data split in MMVR. [`P2S1` or `P2S2`].
      *Example:* `--split P2S1`

    - **`--task`**
      Evaluate detetion and segmentation performance. [`DET` or `DETSEG`].
      *Example:* `--task DETSEG`

    - **`--pretrained_path`**
      The pretrained model checkpoints.
      *Example:* `--pretrained_path ../logs/pretrained_model/p2s1_retr_detseg.pth`

    You should obtain detection performance metrics such as `BBox AP`, `BBox AR1` and segmentation metrics like `Segm IoU` performance listed in the Table above. Additional metrics including `AP50`, `AP75`, and `AR10` are reported in our paper.

> **Note on AP values:** `test.py` and `eval_ptq.py` report different absolute AP for the same model. `test.py` runs the full P2S1 test split (23 k samples, CPU, FP32) and is the paper's reference. `eval_ptq.py` uses a 7,942-sample GPU subset and FP32 gives AP≈49.64; these numbers are the GPU quantization baseline.

## FP32 baseline (original, unquantized)

| Eval | Split | Samples | BBox AP | Seg IoU | Size |
|------|-------|---------|---------|---------|------|
| `test.py` CPU (paper reference) | P2S1 | 23,074 | 46.75 | 77.21 | 156.1 MB |
| `eval_ptq.py` GPU (quant baseline) | P2S1 | 7,942 | 49.64 | 74.92 | 156.1 MB |

The CPU/GPU AP difference is due to oneDNN floating-point operation ordering — not a quantization effect.

## Quantization (PTQ) — Commands and Results

See [DATAFLOW.py](DATAFLOW.py) for the full end-to-end pipeline: CGRA backbone/tokenizer, CIM transformer encoder+decoder, hardware recipes, QNT/DQT/SFM unit specs, and SRAM budgets.

All runs use `src/eval_ptq.py`.  Environment setup:
```bash
conda env create -f environment.yml   # creates retr-quant (Python 3.11, PyTorch 2.11, torchao 0.17.0)
conda activate retr-quant
```

### Reproducible anchor (GPU, P2S1, 7.9 k samples)

```bash
cd src
CUDA_VISIBLE_DEVICES=3 python eval_ptq.py \
  --root ../MMVR/segment_4_3 --split P2S1 --task DETSEG \
  --pretrained_path ../logs/pretrained_model/p2s1_retr_detseg.pth \
  --batch_size 16 --worker 2 --device cuda \
  --scheme int8dq --component all \
  --attn_bmm_bits 8 --attn_weights_bits 8
```

| Metric | Value |
|--------|-------|
| BBox AP | 49.76 |
| Seg IoU | 74.96 |
| Model size | 81.4 MB (1.92×) |
| vs GPU FP32 baseline (AP 49.64) | −0.00 AP (within noise) |

> **Note:** QUANT.md §CPU table (AP=42.66, 23 k samples) was generated on an AMD Ryzen AI MAX+ 395 with the full P2S1 test set via a separate evaluation run. `eval_ptq.py` uses the 7,942-sample GPU subset of P2S1 regardless of `--device`; it does not reproduce the 23 k-sample CPU numbers. The QUANT.md CPU table remains valid as a reference but cannot be regenerated from this script alone.

---

### Current best: INT8dq + NVFP4 attention + all-FP16 (GPU, P2S1, 7.9 k samples)

Full precision ladder from QUANT.md §G2+FP4: W8A8 transformer linears, NVFP4 E2M1 MX-scale attention activations, FP16 backbone, FP16 LayerNorm/softmax/linear-out/residuals.

```bash
cd src
CUDA_VISIBLE_DEVICES=3 python eval_ptq.py \
  --root ../MMVR/segment_4_3 --split P2S1 --task DETSEG \
  --pretrained_path ../logs/pretrained_model/p2s1_retr_detseg.pth \
  --batch_size 16 --worker 2 --device cuda \
  --scheme int8dq --component transformer \
  --attn_bmm_fp4 --attn_weights_fp4 --attn_fp4_mx \
  --use_autocast_fp16 \
  --fp16_half_backbone \
  --fp16_linear_out transformer \
  --fp16_ln all \
  --fp16_softmax all
```

| Metric | Value |
|--------|-------|
| BBox AP | 49.72 |
| Seg IoU | 75.02 |
| Model size | 59.6 MB (2.62×) |
| vs GPU FP32 baseline (AP 49.64) | −0.05 AP (within detection noise) |
| Result file | `experiments/results/P2S1_int8dq_transformer_bmmmxfp4_awmxfp4_ac16_hbe16_lo16-transformer_ln16-all_sm16-all.json` |

---

### INT4 fake-quant + NVFP4 attention + all-FP16 (#1 — max compression, simulated)

Replaces INT8dq transformer linears with INT4 fake-quant (`int4fq_g128`, group=128): weights are rounded to INT4 and immediately dequantized to FP32 before matmul. All other flags identical to current best.

```bash
cd src
CUDA_VISIBLE_DEVICES=3 python eval_ptq.py \
  --root ../MMVR/segment_4_3 --split P2S1 --task DETSEG \
  --pretrained_path ../logs/pretrained_model/p2s1_retr_detseg.pth \
  --batch_size 16 --worker 2 --device cuda \
  --scheme int4fq_g128 --component transformer \
  --attn_bmm_fp4 --attn_weights_fp4 --attn_fp4_mx \
  --use_autocast_fp16 \
  --fp16_half_backbone \
  --fp16_linear_out transformer \
  --fp16_ln all \
  --fp16_softmax all
```

| Metric | Value |
|--------|-------|
| BBox AP (full 7.9 k, GPU) | **50.16** |
| Seg IoU (full 7.9 k) | 74.85 |
| Model size | **47.6 MB (3.3×)** |
| Δ AP vs INT8dq+FP4+FP16 baseline (AP 49.72) | **+0.44 AP** (confirmed INT4 weight regularisation; identical +0.45 AP in pure-FP32 control run) |
| Δ AP vs GPU FP32 baseline (AP 49.64) | **+0.52 AP** |
| Result file | `P2S1_int4fq_g128_transformer_bmmmxfp4_awmxfp4_ac16_hbe16_lo16-transformer_ln16-all_sm16-all.json` |

---

### Backbone INT8 + full FP4+FP16 ladder (#3)

Adds per-output-channel INT8 weight fake-quant on all backbone Conv2d (any kernel size), bypassing the torchao 1×1 Conv2d scale bug via direct weight mutation. Stack after `--fp16_half_backbone` to quantize FP16 backbone weights to INT8 precision. Fake-quant measures accuracy impact; actual deployment stores backbone Conv2d weights as INT8 → effective size ~33 MB (4.7×).

```bash
cd src
CUDA_VISIBLE_DEVICES=3 python eval_ptq.py \
  --root ../MMVR/segment_4_3 --split P2S1 --task DETSEG \
  --pretrained_path ../logs/pretrained_model/p2s1_retr_detseg.pth \
  --batch_size 16 --worker 2 --device cuda \
  --scheme int8dq --component transformer \
  --attn_bmm_fp4 --attn_weights_fp4 --attn_fp4_mx \
  --use_autocast_fp16 \
  --fp16_half_backbone --int8_backbone \
  --fp16_linear_out transformer \
  --fp16_ln all \
  --fp16_softmax all
```

| Metric | Value |
|--------|-------|
| BBox AP (n=1000 quick, GPU) | 0.336 |
| Seg IoU (n=1000) | 0.755 |
| Measured model size (FP16 backbone fake-quant, fake-quant ≠ storage) | 59.6 MB |
| Effective size with INT8 backbone storage | **~33 MB (4.7×)** |
| Δ AP vs INT8dq+FP4+FP16 (same n=1000 subset) | −0.002 (noise) |
| Result file | `P2S1_int8dq_transformer_bmmmxfp4_awmxfp4_ac16_hbe16_bkb8_lo16-transformer_ln16-all_sm16-all_n1000.json` |

> **Note on n=1000 AP values:** The n=1000 quick runs sample the first ~1008 test examples; this subset yields systematically lower absolute AP (~0.34) than the full 7,942-sample GPU run (~0.49). Use these runs for Δ AP comparisons only. Reference point on same subset: INT8dq+FP4+FP16 AP=0.338, IoU=0.756.

---

### Quick sweep (n=1000, GPU)

```bash
cd ace-demo
bash experiments/run_sweep_quick.sh          # 4 CPU runs: FP32 + int8wo (decoder/transformer/all)
bash experiments/run_campaign.sh baseline    # GPU FP32 + INT8dq reproducibility anchors (full test set)
bash experiments/run_campaign.sh g2          # Old G2 recipe: FP16 backbone + INT8dq transformer + INT8 attn fake-quant + FP32 decoder LN
```

> **Note:** `run_campaign.sh g2` runs the earlier chiplet recipe (INT8 fake-quant attention, FP32 decoder LN), which differs from the **current best** recipe above (NVFP4 MX attention, all-FP16 LN/softmax). To reproduce the current best, use the explicit command in the section above.

---

## Hardware mapping — tapeout demo

System: **CGRA** (1 MB scratchpad) ←high-BW→ **FPGA** (off-chip buffer) ←high-BW→ **CIM** (1 MB scratchpad)

### Tokenizer placement: CGRA vs FPGA

The tokenizer computes: FPN feature map `[1, 256, 64, 32]` → L2-norm per spatial location → argsort 2,048 scores → top-512 gather → encoder token tensor `[512, 1, 256]`.

| Aspect | Tokenizer on **CGRA** | Tokenizer on **FPGA** |
|--------|----------------------|----------------------|
| FPGA bandwidth | 100% for weight prefetch to CIM | Split: weight prefetch + token stream |
| CGRA utilization | Backbone → tokenizer in one continuous pass | CGRA idles between backbone output and encoder start |
| Scratchpad pressure | FPN output (1 MB) stays in CGRA — no transfer | FPN output must be moved CGRA→FPGA (1 MB) before tokenizer can run |
| CIM stall risk | None — FPGA exclusively prefetches encoder layer weights | FPGA bandwidth contention if prefetch and tokenizer overlap |
| Data movement | FPN output local; 256 KB token tensor CGRA→FPGA→CIM | 1 MB CGRA→FPGA for FPN, then 256 KB FPGA→CIM for tokens |
| Sort hardware | Needs comparator/reduction tree on CGRA | Natural for FPGA LUT fabric |
| Complexity | CGRA must support argsort (2,048 elements) | Clean functional separation; easy FPGA HDL |

**Recommendation: tokenizer on CGRA**, for two reasons:

1. **FPGA bandwidth is the binding constraint.** Each encoder layer needs 1.9 MB of INT8 weights streamed from FPGA to CIM. With 6 encoder + 6 decoder layers = ~22 MB of weight traffic, keeping FPGA fully dedicated to weight prefetch eliminates CIM stalls. Sharing FPGA bandwidth with tokenizer risks serializing the backbone→encoder handoff against weight loading.

2. **FPN output stays local.** Post-backbone, the `[1, 256, 64, 32]` = 1 MB FPN feature map is already in CGRA scratchpad. Routing it to FPGA for tokenizer and back adds 2 MB of unnecessary inter-chip traffic at the critical path. Keeping it on CGRA allows the L2-norm + argsort + gather to run immediately at backbone completion with zero external traffic until the 256 KB token tensor is sent to CIM.

**If CGRA lacks a sort unit:** hybrid variant — CGRA computes L2-norms (256-wide dot products: native CGRA SIMD), sends 2,048 FP16 scores (4 KB) to FPGA for argsort, FPGA returns 512 indices (1 KB), CGRA performs gather. Total extra traffic: 5 KB; FPGA sort latency: negligible (2,048-element bitonic sort ≈ 11 pipeline stages in LUT fabric).

### Tiling strategy (1 MB scratchpad, scales to larger area)

**Invariant across all scratchpad sizes:** tile unit = 64-token flash-attention Q-tile + one weight chunk. Larger scratchpad reduces re-streaming frequency, not the tile interface.

**CGRA (backbone):** process conv1 in 4 horizontal strips of height 32 (each strip: 128 KB input + 128 KB output). Backbone weights (20 MB FP16) stream from FPGA one layer at a time. From layer2 onward, feature maps ≤128 KB fit without tiling.

**CIM (encoder, one layer at a time):**
- Token buffer `[512, 1, 256]` = 256 KB stays in scratchpad throughout all 6 encoder layers.
- Attention: flash-attention per head, 64-token Q-tile. Per tile: Q (16 KB FP4) + K/V sub-tiles (16 KB each) + score tile (64 KB FP16) ≈ 112 KB active — well within the 780 KB free after token buffer.
- FFN: process in 2 chunks of 512 output channels (128 KB weight + 256 KB output = 384 KB).
- Between layers: only weight streaming from FPGA; token buffer never evicted.

| Scratchpad | Weight streaming | Attention | FPGA role |
|------------|-----------------|-----------|-----------|
| 1 MB | Every op, one tile at a time | 1 head, 64-token Q-tile, K/V sub-tiled | Essential weight buffer |
| 4 MB | Once per layer (hold 1.9 MB + 256 KB tokens) | Full head in scratchpad | Optional staging |
| 16 MB+ | Multi-layer prefetch | All heads simultaneously | Inactive or I/O only |

**Fall-back demo options (hardware only):**
- **One encoder FFN:** `[512, 256]` → `[512, 2048]` INT8 GEMM, FP16 output. Exercises weight streaming, INT8 MAC array, fp16_linear_out hook.
- **One encoder attention head:** FP4 QKᵀ bmm → online softmax (FP32 coprocessor) → FP4 AV bmm. Exercises FP4 datapath end-to-end.
- Full accuracy numbers from simulator demo cover the rest.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for our policy on contributions.

## Citation

If you use this repo, please cite our paper:
```
@misc{yataka2024_retrmultiviewradardetection,
   title       = {{RETR}: Multi-View Radar Detection Transformer for Indoor Perception},
   author      = {Ryoma Yataka and Adriano Cardace and Pu Perry Wang and Petros Boufounos and Ryuhei Takahashi},
   year        = {2024},
   eprint      = {2411.10293},
   archivePrefix={arXiv},
   primaryClass= {cs.CV},
   url         = {https://arxiv.org/abs/2411.10293},
}
```

---
## License
Released under `AGPL-3.0-or-later` license, as found in the [LICENSE.md](LICENSE.md) file.

All files, except as noted below:
```commandline
Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
```

The following files:
- `./src/models/module_retr/misc.py`
- `./src/models/module_retr/backbone.py`

were taken without modification from [here](https://github.com/facebookresearch/detr) (license included in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)), with the following copyrights:
```commandline
Copyright (C) Facebook, Inc. and its affiliates.

SPDX-License-Identifier: Apache-2.0
```

The following file:
- `./src/models/module_retr/attention.py`

was taken without modification from [here](https://github.com/Atten4Vis/ConditionalDETR/tree/main) (license included in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)), with the following copyrights:
```commandline
Copyright (C) 2021 Microsoft.

SPDX-License-Identifier: Apache-2.0
```

The following files:
- `./src/models/module_retr/__init__.py`
- `./src/models/module_retr/box_ops.py`
- `./src/models/module_retr/position_encoding.py`
- `./src/models/module_retr/segmentation.py`

were adapted from [here](https://github.com/facebookresearch/detr) (license included in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)), with the following copyrights:
```commandline
Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
Copyright (C) Facebook, Inc. and its affiliates.

SPDX-License-Identifier: AGPL-3.0-or-later
SPDX-License-Identifier: Apache-2.0
```

The following file:
- `./src/models/module_retr/hubconf.py`

was adapted from [here](https://github.com/Atten4Vis/ConditionalDETR/tree/main) (license included in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)), with the following copyrights:
```commandline
Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
Copyright (C) 2021 Microsoft.

SPDX-License-Identifier: AGPL-3.0-or-later
SPDX-License-Identifier: Apache-2.0
```

The following file:
- `./src/models/module_retr/utils.py`

was adapted from [here](https://github.com/wuzhiwyyx/RFMask-PUB/tree/main) (license included in [LICENSES/MIT.txt](LICENSES/MIT.txt)), with the following copyrights:
```commandline
Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
Copyright (C) 2022 wuzhiwyyx.

SPDX-License-Identifier: AGPL-3.0-or-later
SPDX-License-Identifier: MIT
```

The following files:
- `./src/models/module_retr/detr.py`
- `./src/models/module_retr/matcher.py`
- `./src/models/module_retr/transformer.py`

were adapted from [here](https://github.com/facebookresearch/detr) and [here](https://github.com/Atten4Vis/ConditionalDETR/tree/main) (license included in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)), with the following copyrights:
```commandline
Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
Copyright (C) Facebook, Inc. and its affiliates.
Copyright (C) 2021 Microsoft.

SPDX-License-Identifier: AGPL-3.0-or-later
SPDX-License-Identifier: Apache-2.0
SPDX-License-Identifier: Apache-2.0
```
