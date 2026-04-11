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

## Quantization Study (PTQ)

Post-training quantization experiments using [torchao](https://github.com/pytorch/ao). All runs on CPU (AMD Ryzen AI MAX+ 395), PyTorch 2.11, `retr-quant` conda env, P2S1 test split (23,074 samples).

### Environment setup

PTQ experiments require the `retr-quant` environment (Python 3.11, PyTorch 2.11, torchao 0.17.0) — separate from the original `retr` training env (PyTorch 2.0):

```bash
# One-time setup (creates both envs)
bash setup.sh

# Or just the PTQ env
conda env create -f environment.yml   # creates retr-quant
conda activate retr-quant
```

> **Note:** CPU baseline differs from the GPU numbers above (AP 42.78 vs 46.75) due to oneDNN floating-point ordering. All quantization comparisons use the CPU baseline.

### Results (P2S1, CPU)

FP32 baseline: **AP=42.78 / AR1=39.79 / Seg IoU=74.41 / 156.1 MB**

| Scheme | Component | BBox AP | Δ AP | Seg IoU | Size | Compression |
|--------|-----------|---------|------|---------|------|-------------|
| fp32 | — | 42.78 | — | 74.41 | 156.1 MB | 1.00× |
| int8wo | decoder only | 42.73 | −0.05 | 74.41 | 124.0 MB | 1.26× |
| int8wo | transformer | 42.73 | −0.05 | 74.41 | 81.4 MB | 1.92× |
| int8wo | all | 42.73 | −0.05 | 74.41 | 81.4 MB | 1.92× |
| int8dq | all | 42.66 | −0.12 | 74.40 | 81.4 MB | 1.92× |
| int8dq + BF16 autocast | all | 41.07 | −1.71 | 74.22 | 81.4 MB | 1.92× |
| int4fq_g128 | all | 42.93 | +0.15 | 74.33 | 69.4 MB | 2.25× |
| int4fq_g64 | all | 42.24 | −0.54 | 74.10 | 70.2 MB | 2.22× |
| int4fq_g32 | all | 41.72 | −1.06 | 74.33 | 71.8 MB | 2.18× |
| fp32 + attn-bmm INT8† | — | 42.78 | +0.00 | 74.41 | 156.1 MB | — |
| int8wo + attn-bmm INT8† | all | 42.73 | −0.05 | 74.41 | 81.4 MB | 1.92× |
| int8dq + attn-bmm INT8† | all | 42.66 | −0.12 | 74.40 | 81.4 MB | 1.92× |
| int4fq_g128 + attn-bmm INT8† | all | 42.92 | +0.14 | 74.33 | 69.4 MB | 2.25× |
| fp32 + attn-weights UINT8‡ | — | 42.88 | +0.10 | 74.44 | 156.1 MB | — |
| fp32 + full attn INT8†‡ | — | 42.88 | +0.10 | 74.43 | 156.1 MB | — |
| int8wo + full attn INT8†‡ | all | 42.73 | −0.05 | 74.41 | 81.4 MB | 1.92× |
| int8dq + full attn INT8†‡ | all | 42.66 | −0.12 | 74.40 | 81.4 MB | 1.92× |

†attn-bmm INT8: symmetric per-tensor INT8 fake-quantization of Q, K, V tensors at the bmm input (post-projection, post-head-split) — covers both encoder [4, 512, 512] and decoder [4, 10, 512] attention matmuls. Implemented as a straight-through estimator (STE); measures accuracy impact without hardware INT8 matmul speed.

‡attn-weights UINT8: asymmetric per-tensor UINT8 fake-quantization of the attention weight matrix (softmax output) before AV bmm. Softmax output ∈ [0, 1]; UINT8 uses scale = max(x)/255 to exploit the full unsigned range (vs. symmetric INT8 which wastes half). Full attn INT8 = †+‡ combined.

> **Method citations:** STE fake-quant — PyTorch `torch.ao.quantization.FakeQuantize`; attention matmul quantization in transformers — FQ-ViT (Lin et al., ICCV 2021, arXiv:2111.13824); detection transformer W4A4 PTQ — Q-DETR (Xu et al., CVPR 2023, arXiv:2304.00253).

**Schemes:**
- `int8wo` — weights stored INT8, dequantized to FP32 before matmul; FP32 arithmetic (memory bandwidth saving only)
- `int8dq` — weights INT8 + activations quantized per-token at runtime; real INT8 GEMM via oneDNN
- `int4fq_gN` — symmetric per-group INT4 weight quantization (group size N); dequantized to FP32 before matmul. For weight-only quantization, this is numerically identical to a real INT4 implementation — both dequantize before compute.

**Components** (`--component` flag):
- `encoder` — all `nn.Linear` in encoder layers (72 modules)
- `decoder` — all `nn.Linear` in decoder layers (89 modules)
- `transformer` — encoder + decoder (161 modules)
- `all` — same as `transformer`; backbone has no `nn.Linear` (Conv2d only), so `all` and `transformer` produce identical results
- `ffn`, `proj` — FFN-only or projection-only subsets within the transformer
- `backbone` — no-op (backbone is Conv2d; torchao has a shape bug on 1×1 convolutions in this version)

**Key findings:**
- INT8 weight-only is effectively lossless at 1.92× compression (−0.05 AP, IoU unchanged)
- INT8 dynamic activation adds −0.07 AP on top of weight-only at the same compression
- Stacking BF16 autocast on top of int8dq causes −1.71 AP — error from BF16 backbone compounding with INT8 activation quantization; not a viable configuration
- INT8 Q/K/V at bmm input adds zero incremental loss — fp32+bmm8 is identical to fp32 baseline (Δ0.00); the dominant source of loss is weight quantization, not attention matmuls
- INT8 attention weight matrix (softmax output, UINT8) likewise adds zero incremental loss — full INT8 attention (Q/K/V + weights) costs the same as weight quantization alone
- int8dq + full attn INT8 (W8A8 everywhere including attention): AP=42.66, Δ−0.12 — same as int8dq without attention quantization; attention quantization is free

### Acceptable regression — literature survey (April 2026)

Community norm for INT8 PTQ on detection models: **≤1 AP point** (absolute). Our drops are well within this. No prior RETR/MMVR quantization results exist in the literature; these are the first.

| Standard | Threshold | int8wo (−0.05 AP) | int8dq (−0.12 AP) |
|----------|-----------|-------------------|-------------------|
| MLPerf 99% floor (42.35 AP) | ≥99% of FP32 | +0.38 above floor | +0.31 above floor |
| MLPerf 99.9% floor (42.74 AP) | ≥99.9% of FP32 | −0.01 (borderline) | −0.08 (just outside) |
| Detection community norm | ≤1.0 AP drop | 20× within | 8× within |
| Radar/embedded FPGA (MDPI Sensors 2024) | ≤0.74% accuracy drop | Pass | Pass |
| Q-DETR W4A4 PTQ, CVPR 2023 (lower bitwidth) | −2.6 AP drop | 52× smaller drop | 22× smaller drop |

> MLPerf floors computed as: 0.99 × 42.78 = 42.35 and 0.999 × 42.78 = 42.74. The 0.01 AP margin on the 99.9% tier is within measurement noise; treat as borderline rather than a firm pass/fail. AP is a bounded non-linear summary statistic — absolute drops are more interpretable than relative percentages.

**Hardware sizing implication:** INT8 MAC arrays are fully justified. Weight-only INT8 is essentially lossless; full W8A8 costs −0.07 AP additional, favorable if it halves activation SRAM bandwidth. Keep 32-bit accumulators. Validation gate: hardware INT8 should land within ±0.3 AP of the software INT8 reference on MMVR P2S1 test.

### Attention matrix sizes (hardware design reference)

Model parameters: nhead=4 (encoder and decoder), encoder tokens=2×topk=512, num_queries=10, encoder head_dim=128, decoder head_dim=64.

| Attention site | Q shape | K shape | QK^T shape | FP32 size / sample | Decision |
|----------------|---------|---------|------------|---------------------|----------|
| Encoder self-attn | [4, 512, 128] | [4, 512, 128] | [4, 512, 512] | 4×512×512×4 B = **4 MB** | Online softmax required |
| Decoder cross-attn | [4, 10, 64] | [4, 512, 64] | [4, 10, 512] | 4×10×512×4 B = **80 KB** | Materialise in full |
| Decoder self-attn | [4, 10, 64] | [4, 10, 64] | [4, 10, 10] | 4×10×10×4 B = **1.6 KB** | Trivial |

**Encoder online softmax:** The 512×512 attention matrix (1 MB/head) exceeds practical SRAM tile budgets. Flash Attention tiling applies: accumulate softmax numerator and running max in a single pass, never materialising the full matrix. PyTorch SDPA already uses this path; hardware implementation should mirror the same tiling.

**UCIe coprocessor scope:** Main INT8 MAC array handles GEMM (QK^T, AV, all linear projections). UCIe-connected FP coprocessor handles: online softmax reduction, BF16/FP32 residual adds, LayerNorm. Decoder attention (≤80 KB matrices) can be handled entirely in either unit.

### Running PTQ experiments

**Quick sweep — 4 runs, ~70 min on CPU:**
```bash
conda activate retr-quant
bash experiments/run_sweep_quick.sh
```

**Single experiment:**
```bash
conda activate retr-quant
cd src
python eval_ptq.py \
    --root ../MMVR/segment_4_3 --split P2S1 --task DETSEG \
    --pretrained_path ../logs/pretrained_model/p2s1_retr_detseg.pth \
    --batch_size 8 --worker 2 --device cpu \
    --scheme int8wo --component all
```

**With BF16 autocast stacked on top of any scheme** (covers backbone Conv2d and attention matmuls):
```bash
python eval_ptq.py ... --scheme int8dq --component all --use_autocast
```

> Note: if `python` resolves to the wrong interpreter, use the env Python directly:
> `~/miniforge3/envs/retr-quant/bin/python eval_ptq.py ...`

Results are saved to `experiments/results/<run_name>.json`. See `docs/experiments.md` for the full experiment catalogue and JSON schema.

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
