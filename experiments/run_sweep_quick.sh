#!/usr/bin/env bash
# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Quick PTQ sweep — 4 runs to validate the pipeline and get first accuracy signals.
# Run this before run_sweep.sh to catch issues early.
#
# Usage:
#   conda activate retr-quant
#   cd ace-demo
#   bash experiments/run_sweep_quick.sh [--dry-run]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${REPO_ROOT}/src"
DATA="${REPO_ROOT}/MMVR/segment_4_3"
CKPT="${REPO_ROOT}/logs/pretrained_model/p2s1_retr_detseg.pth"
SPLIT="P2S1"
TASK="DETSEG"
BATCH=8
WORKERS=2
DEVICE="cpu"

DRY_RUN=0
for arg in "$@"; do
  [[ "$arg" == "--dry-run" ]] && DRY_RUN=1
done

run() {
    echo ">>> $*"
    if [[ $DRY_RUN -eq 0 ]]; then
        ~/miniforge3/bin/conda run -n retr-quant --cwd "$SRC" python eval_ptq.py "$@"
    fi
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

# 1. FP32 baseline
run "${COMMON[@]}"

# 2. INT8 weight-only, full model
run "${COMMON[@]}" --scheme int8wo --component all

# 3. INT8 weight-only, transformer (encoder + decoder) only
run "${COMMON[@]}" --scheme int8wo --component transformer

# 4. INT8 weight-only, decoder only
run "${COMMON[@]}" --scheme int8wo --component decoder

echo ""
echo "Quick sweep complete. Results in ${REPO_ROOT}/experiments/results/"
