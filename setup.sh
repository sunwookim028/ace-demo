#!/usr/bin/env bash
# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Push-button environment setup for RETR.
#
# What this script does:
#   1. Creates the retr-quant conda environment (PyTorch 2.4 + torchao)
#   2. Creates the baseline retr conda environment (PyTorch 2.0)
#   3. (Optional) Preprocesses raw MMVR data into segment_4_3 format
#
# Usage:
#   bash setup.sh [--skip-retr] [--skip-retr-quant] [--preprocess]
#
# Flags:
#   --skip-retr         Skip creating the baseline 'retr' environment
#   --skip-retr-quant   Skip creating the 'retr-quant' environment
#   --preprocess        Run data preprocessing (requires raw MMVR/P2 data)
#   --help              Show this message
#
# Prerequisites:
#   - conda / miniforge installed and on PATH
#   - For --preprocess: raw MMVR dataset extracted under MMVR/P2/
#     (download from https://zenodo.org/records/12611978)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA="${CONDA_EXE:-$(which conda 2>/dev/null || echo "$HOME/miniforge3/bin/conda")}"

DO_RETR=1
DO_RETR_QUANT=1
DO_PREPROCESS=0

for arg in "$@"; do
    case "$arg" in
        --skip-retr)        DO_RETR=0 ;;
        --skip-retr-quant)  DO_RETR_QUANT=0 ;;
        --preprocess)       DO_PREPROCESS=1 ;;
        --help)
            sed -n '2,/^set -/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'
            exit 0
            ;;
    esac
done

echo "=== RETR setup ==="
echo "Repo:  $REPO_ROOT"
echo "Conda: $CONDA"
echo ""

# ------------------------------------------------------------------ retr-quant
if [[ $DO_RETR_QUANT -eq 1 ]]; then
    echo "--- Creating retr-quant environment (PyTorch 2.4 + torchao) ---"
    "$CONDA" env create -f "$REPO_ROOT/environment.yml" --force
    echo ""
fi

# ------------------------------------------------------------------ retr
if [[ $DO_RETR -eq 1 ]]; then
    echo "--- Creating retr environment (PyTorch 2.0 baseline) ---"
    "$CONDA" create -n retr python=3.10 -y
    "$CONDA" run -n retr pip install \
        torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 \
        --index-url https://download.pytorch.org/whl/cu117
    "$CONDA" run -n retr pip install -r "$REPO_ROOT/requirements.txt"
    echo ""
fi

# ------------------------------------------------------------------ preprocess
if [[ $DO_PREPROCESS -eq 1 ]]; then
    RAW_DIR="$REPO_ROOT/MMVR/P2"
    OUT_DIR="$REPO_ROOT/MMVR"

    if [[ ! -d "$RAW_DIR" ]]; then
        echo "ERROR: $RAW_DIR does not exist."
        echo "Download and extract P2_00.zip, P2_01.zip, P2_02.zip to MMVR/P2/ first."
        echo "See README.md for download instructions."
        exit 1
    fi

    echo "--- Preprocessing MMVR P2 data (4 frames, overlap 3) ---"
    echo "Input:  $RAW_DIR"
    echo "Output: $OUT_DIR/segment_4_3/P2/"
    "$CONDA" run -n retr --cwd "$REPO_ROOT/src/data" \
        python create_grouped_dataset.py \
        --num_frames 4 \
        --overlap 3 \
        --dataset_dir "$REPO_ROOT/MMVR" \
        --output "$REPO_ROOT/MMVR"
    echo ""
fi

# ------------------------------------------------------------------ verify
echo "--- Verifying setup ---"

CKPT="$REPO_ROOT/logs/pretrained_model/p2s1_retr_detseg.pth"
if [[ -f "$CKPT" ]]; then
    echo "[OK] Pretrained checkpoint: $CKPT"
else
    echo "[MISSING] $CKPT"
    echo "  Download from: https://github.com/merl-internal/retr_release/raw/refs/heads/main/logs/pretrained_model/p2s1_retr_detseg.pth"
fi

DATA="$REPO_ROOT/MMVR/segment_4_3"
if [[ -d "$DATA" ]]; then
    N=$(find "$DATA" -name "*_radar.npz" 2>/dev/null | wc -l)
    echo "[OK] Preprocessed dataset: $DATA ($N radar files)"
else
    echo "[MISSING] $DATA  — run with --preprocess (requires raw MMVR/P2/)"
fi

if "$CONDA" env list | grep -q "^retr-quant "; then
    echo "[OK] conda env: retr-quant"
else
    echo "[MISSING] conda env: retr-quant  — run without --skip-retr-quant"
fi

if "$CONDA" env list | grep -q "^retr "; then
    echo "[OK] conda env: retr"
else
    echo "[MISSING] conda env: retr  — run without --skip-retr"
fi

echo ""
echo "=== Next steps ==="
echo ""
echo "  Baseline evaluation:"
echo "    conda run -n retr --cwd src python test.py \\"
echo "        --root ../MMVR/segment_4_3 --split P2S1 --task DETSEG \\"
echo "        --pretrained_path ../logs/pretrained_model/p2s1_retr_detseg.pth"
echo ""
echo "  Quick PTQ sweep (4 runs, CPU, ~70 min):"
echo "    bash experiments/run_sweep_quick.sh"
echo ""
echo "  Full PTQ sweep (22 runs, requires GPU):"
echo "    bash experiments/run_sweep.sh"
echo ""
echo "  See WELCOME.md for full documentation."
