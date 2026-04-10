# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Unified PTQ evaluation script for RETR.

Wraps the logic of test.py, adds --scheme / --component / --attn_act_bits
arguments, and saves a JSON result to experiments/results/.

Example
-------
# FP32 baseline
python eval_ptq.py --root ./MMVR/segment_4_3 --split P2S1 --task DETSEG \\
    --pretrained_path ../logs/pretrained_model/p2s1_retr_detseg.pth

# INT8 weight-only, all components
python eval_ptq.py ... --scheme int8wo --component all

# INT4 per-group=128, decoder only
python eval_ptq.py ... --scheme int4wo_g128 --component decoder

# INT8 dynamic activation + INT8 weight, encoder + decoder
python eval_ptq.py ... --scheme int8dq --component transformer

# INT8 weight + INT8 attention activations
python eval_ptq.py ... --scheme int8wo --component all --attn_act_bits 8

Notes
-----
* metrics.get_result() in the upstream does NOT return a value (upstream bug).
  We access metrics.res directly after calling get_result().
* All results are saved to experiments/results/<run_name>.json.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# det_seg_dataset.py hardcodes "./utils/data_split.npz" relative to CWD.
# test.py is designed to be run from src/. We enforce that here.
_SRC_DIR = Path(__file__).resolve().parent
os.chdir(_SRC_DIR)
sys.path.insert(0, str(_SRC_DIR))

from data.dataloader import collate_det_seg, get_dataloader
from data.det_seg_dataset import MMVRDetSeg
from models import RETR
from quantize_torchao import (
    COMPONENT_FILTERS,
    DTYPE_SCHEMES,
    SCHEMES,
    apply_ptq,
    migrate_encoder_mha_state_dict,
    model_size_mb,
    register_attention_act_quant_hooks,
)
from utils.common import move_to_device
from utils.detection_process import Metrics

project_root = Path(__file__).resolve().parent.parent
results_dir = project_root / "experiments" / "results"


def get_args_parser():
    parser = argparse.ArgumentParser("RETR PTQ evaluation", add_help=False)

    # --- dataset ---
    parser.add_argument("--root", type=str, required=True,
                        help="Path to MMVR/segment_4_3")
    parser.add_argument("--split", default="P2S1", type=str)
    parser.add_argument("--task", default="DETSEG", type=str,
                        choices=["DETSEG", "DET"])
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--worker", default=2, type=int)

    # --- model ---
    parser.add_argument("--pretrained_path",
                        default="../logs/pretrained_model/p2s1_retr_detseg.pth",
                        type=str)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)

    # --- quantization ---
    parser.add_argument("--scheme", default=None,
                        choices=list(SCHEMES.keys()) + [None],
                        help="Quantization scheme. Omit for FP32 baseline.")
    parser.add_argument("--component", default="all",
                        choices=list(COMPONENT_FILTERS.keys()),
                        help="Which model components to quantize.")
    parser.add_argument("--attn_act_bits", default=None, type=int,
                        choices=[8],
                        help="If set, also fake-quantize decoder attention "
                             "activations (Q/K/V) to this many bits.")
    parser.add_argument("--use_autocast", action="store_true",
                        help="Wrap forward pass in torch.autocast(bfloat16). "
                             "Stacks on top of any --scheme: linears stay INT8 "
                             "(int8dq dispatch bypasses autocast), backbone Conv2d "
                             "and attention bmm run BF16.")

    # --- output ---
    parser.add_argument("--run_name", default=None, type=str,
                        help="Override output JSON filename stem. "
                             "Defaults to <split>_<scheme>_<component>.")
    return parser


def build_run_name(args) -> str:
    scheme_tag = args.scheme if args.scheme else "fp32"
    attn_tag = f"_attn{args.attn_act_bits}" if args.attn_act_bits else ""
    autocast_tag = "_bf16" if getattr(args, "use_autocast", False) else ""
    return f"{args.split}_{scheme_tag}_{args.component}{attn_tag}{autocast_tag}"


def main(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ------------------------------------------------------------------ model
    task_internal = "SEG" if args.task == "DETSEG" else "DET"
    model = RETR(task=task_internal).to(device)
    params = torch.load(args.pretrained_path, map_location=device)
    params = migrate_encoder_mha_state_dict(params)
    model.load_state_dict(params)
    model.eval()

    fp32_size = model_size_mb(model)

    # ------------------------------------------------------------ quantization
    attn_hooks = []
    if args.scheme is not None:
        apply_ptq(model, scheme=args.scheme, component=args.component)

    if args.attn_act_bits is not None:
        attn_hooks = register_attention_act_quant_hooks(model, bits=args.attn_act_bits)

    quant_size = model_size_mb(model)

    # ------------------------------------------------------------------- data
    dataset_path = Path(args.root) / args.split[:2]
    _, _, test_loader = get_dataloader(
        MMVRDetSeg,
        dataset_path,
        split=args.split,
        batch_size=args.batch_size,
        collate_fn=collate_det_seg,
        num_workers=args.worker,
    )

    # --------------------------------------------------------------- evaluate
    metrics = Metrics(seg=(task_internal == "SEG")).to(device)

    latencies = []
    use_autocast = args.scheme in DTYPE_SCHEMES or getattr(args, "use_autocast", False)
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if use_autocast else torch.autocast(device_type=device.type, enabled=False)
    )

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=build_run_name(args)):
            batch = move_to_device(batch, device)
            rf_hor = batch["hm_hori"].detach()
            rf_ver = batch["hm_vert"].detach()
            labels = batch["labels"]

            t0 = time.perf_counter()
            with autocast_ctx:
                out = model(rf_hor, rf_ver)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) / rf_hor.shape[0])

            metrics.compute(labels, out)

    # get_result() prints and populates metrics.res but does NOT return.
    metrics.get_result()

    # ----------------------------------------------------------------- result
    res = metrics.res
    result = {
        "run_name": build_run_name(args),
        "split": args.split,
        "task": args.task,
        "scheme": args.scheme if args.scheme else "fp32",
        "component": args.component,
        "attn_act_bits": args.attn_act_bits,
        "use_autocast": getattr(args, "use_autocast", False),
        "pretrained_path": args.pretrained_path,
        # accuracy
        "bbox_ap": res["det_img"]["map"].item(),
        "bbox_ap50": res["det_img"]["map_50"].item(),
        "bbox_ap75": res["det_img"]["map_75"].item(),
        "bbox_ar1": res["det_img"]["mar_1"].item(),
        "bbox_ar10": res["det_img"]["mar_10"].item(),
        "seg_iou": res["seg_img"].item() if res["seg_img"] is not None else None,
        # model size
        "model_size_fp32_mb": round(fp32_size, 2),
        "model_size_quant_mb": round(quant_size, 2),
        # latency (per-sample, seconds)
        "latency_mean_s": round(float(np.mean(latencies)), 5),
        "latency_std_s": round(float(np.std(latencies)), 5),
        "n_batches": len(latencies),
    }

    # ------------------------------------------------------------------ save
    results_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name if args.run_name else build_run_name(args)
    out_path = results_dir / f"{run_name}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ---------------------------------------------------------- cleanup hooks
    for h in attn_hooks:
        h.remove()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("RETR PTQ eval", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
