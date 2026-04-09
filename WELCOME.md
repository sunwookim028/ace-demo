<!--
Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# RETR — Getting Started

**RETR** (Radar dEtection TRansformer) is a DETR-style model for person detection and instance segmentation from multi-view indoor radar heatmaps (NeurIPS 2024). This repo extends the original release with post-training quantization (PTQ) experiments using [torchao](https://github.com/pytorch/ao).

> **Paper:** [RETR: Multi-View Radar Detection Transformer for Indoor Perception](https://arxiv.org/abs/2411.10293)

---

## Key results at a glance

Pretrained on MMVR P2S1 (`logs/pretrained_model/p2s1_retr_detseg.pth`, 156 MB):

| Metric | Value |
|--------|-------|
| BBox AP (mAP @ IoU 0.5:0.95) | **46.75** |
| BBox AR1 (recall @ 1 det/image) | **42.19** |
| Segm IoU | **77.21** |

All measured on 23,074 test frames across 27 takes (P2S1 split, environments d5–d9).

---

## Quickstart — reproduce baseline in 3 steps

```bash
# 1. Create environment
conda env create -f environment.yml        # retr-quant (PyTorch 2.4 + torchao)
# -- or for baseline eval only --
conda create -n retr python=3.10
conda install -n retr pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.7 -c pytorch -c nvidia
conda run -n retr pip install -r requirements.txt

# 2. Preprocess dataset  (skip if MMVR/segment_4_3 already exists)
conda run -n retr --cwd src/data \
    python create_grouped_dataset.py --num_frames 4 --overlap 3 \
    --dataset_dir ../../MMVR --output ../../MMVR

# 3. Run baseline evaluation
conda run -n retr --cwd src \
    python test.py --root ../MMVR/segment_4_3 --split P2S1 --task DETSEG \
    --pretrained_path ../logs/pretrained_model/p2s1_retr_detseg.pth
```

For the full automated setup see [`setup.sh`](setup.sh).

---

## What's in this repo

```
ace-demo/
├── src/
│   ├── eval_ptq.py            ← PTQ evaluation entry point  (this fork)
│   ├── quantize_torchao.py    ← Quantization schemes & components  (this fork)
│   ├── test.py                ← Upstream baseline evaluation
│   ├── train.py               ← Upstream training
│   ├── models/                ← RETR architecture
│   ├── data/                  ← MMVR dataset loader
│   └── utils/                 ← Metrics (torchmetrics)
├── experiments/
│   ├── run_sweep.sh           ← Full PTQ sweep (22 experiments)
│   └── run_sweep_quick.sh     ← Quick 4-experiment validation sweep
├── logs/pretrained_model/     ← Pretrained checkpoints (P2S1, P2S2)
├── MMVR/segment_4_3/          ← Preprocessed dataset (after setup)
├── setup.sh                   ← Push-button environment + preprocessing setup
├── MODEL.md                   ← Concise architecture/layer/metric reference
└── docs/
    ├── architecture.md        ← Deep-dive: layers, I/O shapes, code walkthrough
    └── experiments.md         ← PTQ experiment guide: what each run measures
```

---

## Detailed documentation

| Document | Contents |
|----------|----------|
| [docs/architecture.md](docs/architecture.md) | Model pipeline, layer-by-layer table with shapes and dtypes, data loading chain, annotated forward-pass walkthrough |
| [docs/experiments.md](docs/experiments.md) | What every quantization scheme/component combination measures, how to run sweeps, how to read result JSON files |
| [MODEL.md](MODEL.md) | Concise reference: architecture diagram, layer inventory, I/O dimensions, metric definitions, quantization targets |
| [README.md](README.md) | Original upstream README: installation, dataset download, training procedure, citation |

---

## Environments

Two conda environments are used:

| Env | Purpose | Key packages |
|-----|---------|-------------|
| `retr` | Upstream baseline eval & training | PyTorch 2.0, torchvision 0.15 |
| `retr-quant` | PTQ experiments | PyTorch ≥ 2.4, torchao ≥ 0.6 |

Create both with `bash setup.sh`, or individually:

```bash
conda env create -f environment.yml                               # retr-quant
conda create -n retr python=3.10 && \
  conda run -n retr pip install torch==2.0.0 torchvision==0.15.0 && \
  conda run -n retr pip install -r requirements.txt              # retr
```

---

## Running quantization experiments

```bash
# 4-experiment quick validation (CPU, ~70 min total)
bash experiments/run_sweep_quick.sh

# Full 22-experiment sweep (requires CUDA, ~8 hr)
bash experiments/run_sweep.sh

# Single experiment
conda run -n retr-quant --cwd src python eval_ptq.py \
    --root ../MMVR/segment_4_3 --split P2S1 --task DETSEG \
    --pretrained_path ../logs/pretrained_model/p2s1_retr_detseg.pth \
    --scheme int8wo --component all

# Results land in:
ls experiments/results/P2S1_*.json
```

See [docs/experiments.md](docs/experiments.md) for the full experiment catalogue and how to interpret results.
