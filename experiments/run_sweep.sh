#!/usr/bin/env bash
# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Reproducible PTQ experiment sweep for RETR P2S1.
#
# Prerequisites
# -------------
#   conda activate retr-quant
#   MMVR/segment_4_3 preprocessed (see MODEL.md)
#   logs/pretrained_model/p2s1_retr_detseg.pth present
#
# Usage
#   cd ace-demo
#   bash experiments/run_sweep.sh [--dry-run]
#
# Results land in experiments/results/<run_name>.json.
# Each JSON file contains BBox AP, AR1, Seg IoU, model size, latency.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${REPO_ROOT}/src"
DATA="${REPO_ROOT}/MMVR/segment_4_3"
CKPT="${REPO_ROOT}/logs/pretrained_model/p2s1_retr_detseg.pth"
SPLIT="P2S1"
TASK="DETSEG"
BATCH=32
WORKERS=2
DEVICE="cuda"

DRY_RUN=0
for arg in "$@"; do
  [[ "$arg" == "--dry-run" ]] && DRY_RUN=1
done

run() {
    echo ">>> $*"
    [[ $DRY_RUN -eq 0 ]] && python "$SRC/eval_ptq.py" "$@"
}

COMMON=(
    --root "$DATA"
    --split "$SPLIT"
    --task "$TASK"
    --pretrained_path "$CKPT"
    --batch_size "$BATCH"
    --worker "$WORKERS"
    --device "$DEVICE"
)

# ------------------------------------------------------------------
# 1. FP32 baseline
# ------------------------------------------------------------------
run "${COMMON[@]}"

# ------------------------------------------------------------------
# 2. INT8 weight-only — per component
# ------------------------------------------------------------------
for COMP in backbone encoder decoder transformer ffn proj all; do
    run "${COMMON[@]}" --scheme int8wo --component "$COMP"
done

# ------------------------------------------------------------------
# 3. INT4 weight-only, group_size sweep — full model
# ------------------------------------------------------------------
for GS in 128 64 32; do
    run "${COMMON[@]}" --scheme "int4wo_g${GS}" --component all
done

# ------------------------------------------------------------------
# 4. INT4 weight-only g=128 — per component
# ------------------------------------------------------------------
for COMP in backbone encoder decoder transformer ffn proj; do
    run "${COMMON[@]}" --scheme int4wo_g128 --component "$COMP"
done

# ------------------------------------------------------------------
# 5. INT8 dynamic activation + INT8 weight
# ------------------------------------------------------------------
for COMP in encoder decoder transformer all; do
    run "${COMMON[@]}" --scheme int8dq --component "$COMP"
done

# ------------------------------------------------------------------
# 6. INT8 weight-only + INT8 attention activations (decoder hooks)
# ------------------------------------------------------------------
run "${COMMON[@]}" --scheme int8wo --component decoder --attn_act_bits 8
run "${COMMON[@]}" --scheme int8wo --component all     --attn_act_bits 8

# ------------------------------------------------------------------
# 7. INT4 g=128 weight + INT8 attention activations
# ------------------------------------------------------------------
run "${COMMON[@]}" --scheme int4wo_g128 --component all --attn_act_bits 8

echo ""
echo "Sweep complete. Results in ${REPO_ROOT}/experiments/results/"
