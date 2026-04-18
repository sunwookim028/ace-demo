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
    apply_fp16_half_backbone,
    apply_ptq,
    migrate_encoder_mha_state_dict,
    model_size_mb,
    register_attention_act_quant_hooks,
    register_attention_bmm_fp4_hooks,
    register_attention_bmm_quant_hooks,
    register_attention_weights_fp4_hooks,
    register_attention_weights_quant_hooks,
    register_fp16_linear_out_hooks,
    register_fp16_ln_hooks,
    register_fp16_softmax_hooks,
    register_fp32_ln_hooks,
    register_fp32_softmax_hooks,
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
    parser.add_argument("--attn_bmm_bits", default=None, type=int,
                        choices=[8],
                        help="Fake-quantize Q/K/V at bmm input (post-projection, "
                             "post-head-split) to INT{bits}. Covers encoder and "
                             "decoder attention matmuls (QK^T and AV).")
    parser.add_argument("--attn_weights_bits", default=None, type=int,
                        choices=[8],
                        help="Fake-quantize attention weight matrix (softmax output) "
                             "to UINT8 before AV bmm. Non-negative ∈ [0,1]; uses "
                             "asymmetric unsigned quantization.")
    parser.add_argument("--attn_bmm_fp4", action="store_true",
                        help="NVFP4 E2M1 block fake-quant on Q/K/V at bmm input "
                             "(block size 16, per-block FP8 scale). Mutually exclusive "
                             "with --attn_bmm_bits.")
    parser.add_argument("--attn_weights_fp4", action="store_true",
                        help="NVFP4 E2M1 block fake-quant on attention weights "
                             "(post-softmax, before AV bmm). Mutually exclusive with "
                             "--attn_weights_bits.")
    parser.add_argument("--attn_fp4_mx", action="store_true",
                        help="Use power-of-2 (MX microscaling) scale for FP4 blocks "
                             "instead of FP8 scale. ~3x cheaper scale compute. "
                             "Requires --attn_bmm_fp4 or --attn_weights_fp4.")
    parser.add_argument("--use_autocast", action="store_true",
                        help="Wrap forward pass in torch.autocast(bfloat16). "
                             "Stacks on top of any --scheme: linears stay INT8 "
                             "(int8dq dispatch bypasses autocast), backbone Conv2d "
                             "and attention bmm run BF16.")
    parser.add_argument("--use_autocast_fp16", action="store_true",
                        help="Wrap forward in torch.autocast(float16). "
                             "For GPU FP16 study; softmax/LN auto-upcast to FP32 "
                             "by default (torch autocast op lists).")
    parser.add_argument("--fp16_half_backbone", action="store_true",
                        help="Convert RETR backbone + input_proj{,_ver} weights to "
                             "FP16 (.half()). Casts inputs at backbone entry and "
                             "outputs back to FP32 at input_proj. Skips AQT modules.")
    parser.add_argument("--fp32_ln", default="none",
                        choices=["none", "encoder", "decoder", "transformer", "all"],
                        help="Force in-scope nn.LayerNorm to run FP32 regardless of "
                             "surrounding autocast (chiplet LN-on-FPGA recipe).")
    parser.add_argument("--fp32_softmax", default="none",
                        choices=["none", "encoder", "decoder", "transformer", "all"],
                        help="Force softmax to run FP32 in in-scope attention "
                             "modules via attribute injection.")
    parser.add_argument("--fp16_linear_out", default="none",
                        choices=["none", "encoder", "decoder", "transformer", "all"],
                        help="Cast nn.Linear output FP32→FP16 after each forward pass in scope. "
                             "Fixes the torchao int8dq FP32 output so residual adds run in FP16.")
    parser.add_argument("--fp16_ln", default="none",
                        choices=["none", "encoder", "decoder", "transformer", "all"],
                        help="Force in-scope nn.LayerNorm to run FP16 (casts params "
                             "and input). Overrides --fp32_ln for the same scope.")
    parser.add_argument("--fp16_softmax", default="none",
                        choices=["none", "encoder", "decoder", "transformer", "all"],
                        help="Force softmax to run FP16 in in-scope attention modules. "
                             "Warning: FP16 softmax on encoder 512×512 maps may overflow.")
    parser.add_argument("--max_samples", default=None, type=int,
                        help="If set, stop evaluation after the first N samples "
                             "(fast Quick runs). None = full test set.")

    # --- output ---
    parser.add_argument("--run_name", default=None, type=str,
                        help="Override output JSON filename stem. "
                             "Defaults to <split>_<scheme>_<component>.")
    return parser


def build_run_name(args) -> str:
    scheme_tag = args.scheme if args.scheme else "fp32"
    attn_tag = f"_attn{args.attn_act_bits}" if args.attn_act_bits else ""
    bmm_tag = f"_bmm{args.attn_bmm_bits}" if getattr(args, "attn_bmm_bits", None) else ""
    aw_tag = f"_aw{args.attn_weights_bits}" if getattr(args, "attn_weights_bits", None) else ""
    mx_suffix = "mx" if getattr(args, "attn_fp4_mx", False) else ""
    bmm_fp4_tag = f"_bmm{mx_suffix}fp4" if getattr(args, "attn_bmm_fp4", False) else ""
    aw_fp4_tag = f"_aw{mx_suffix}fp4" if getattr(args, "attn_weights_fp4", False) else ""
    autocast_tag = "_bf16" if getattr(args, "use_autocast", False) else ""
    fp16_autocast_tag = "_ac16" if getattr(args, "use_autocast_fp16", False) else ""
    fp16_be_tag = "_hbe16" if getattr(args, "fp16_half_backbone", False) else ""
    ln_tag = f"_ln32-{args.fp32_ln}" if getattr(args, "fp32_ln", "none") != "none" else ""
    sm_tag = f"_sm32-{args.fp32_softmax}" if getattr(args, "fp32_softmax", "none") != "none" else ""
    lo16_tag = f"_lo16-{args.fp16_linear_out}" if getattr(args, "fp16_linear_out", "none") != "none" else ""
    ln16_tag = f"_ln16-{args.fp16_ln}" if getattr(args, "fp16_ln", "none") != "none" else ""
    sm16_tag = f"_sm16-{args.fp16_softmax}" if getattr(args, "fp16_softmax", "none") != "none" else ""
    max_tag = f"_n{args.max_samples}" if getattr(args, "max_samples", None) else ""
    return (f"{args.split}_{scheme_tag}_{args.component}{attn_tag}{bmm_tag}{aw_tag}"
            f"{bmm_fp4_tag}{aw_fp4_tag}{autocast_tag}{fp16_autocast_tag}"
            f"{fp16_be_tag}{ln_tag}{sm_tag}{lo16_tag}{ln16_tag}{sm16_tag}{max_tag}")


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

    if getattr(args, "attn_bmm_bits", None) is not None:
        attn_hooks += register_attention_bmm_quant_hooks(model, bits=args.attn_bmm_bits)

    if getattr(args, "attn_weights_bits", None) is not None:
        attn_hooks += register_attention_weights_quant_hooks(model)

    mx = getattr(args, "attn_fp4_mx", False)
    if getattr(args, "attn_bmm_fp4", False):
        attn_hooks += register_attention_bmm_fp4_hooks(model, block_size=16, mx_scale=mx)

    if getattr(args, "attn_weights_fp4", False):
        attn_hooks += register_attention_weights_fp4_hooks(model, block_size=16, mx_scale=mx)

    if getattr(args, "fp16_half_backbone", False):
        attn_hooks += apply_fp16_half_backbone(model)

    if getattr(args, "fp32_ln", "none") != "none":
        attn_hooks += register_fp32_ln_hooks(model, scope=args.fp32_ln)

    if getattr(args, "fp32_softmax", "none") != "none":
        attn_hooks += register_fp32_softmax_hooks(model, scope=args.fp32_softmax)

    if getattr(args, "fp16_linear_out", "none") != "none":
        attn_hooks += register_fp16_linear_out_hooks(model, scope=args.fp16_linear_out)

    if getattr(args, "fp16_ln", "none") != "none":
        attn_hooks += register_fp16_ln_hooks(model, scope=args.fp16_ln)

    if getattr(args, "fp16_softmax", "none") != "none":
        attn_hooks += register_fp16_softmax_hooks(model, scope=args.fp16_softmax)

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
    fp16_autocast = args.scheme == "fp16" or getattr(args, "use_autocast_fp16", False)
    bf16_autocast = args.scheme == "bf16" or getattr(args, "use_autocast", False)
    use_autocast = fp16_autocast or bf16_autocast
    autocast_dtype = torch.float16 if fp16_autocast else torch.bfloat16
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=autocast_dtype)
        if use_autocast else torch.autocast(device_type=device.type, enabled=False)
    )

    model.eval()
    n_samples_seen = 0
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

            n_samples_seen += rf_hor.shape[0]
            if args.max_samples is not None and n_samples_seen >= args.max_samples:
                break

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
        "attn_bmm_bits": getattr(args, "attn_bmm_bits", None),
        "attn_weights_bits": getattr(args, "attn_weights_bits", None),
        "attn_bmm_fp4": getattr(args, "attn_bmm_fp4", False),
        "attn_weights_fp4": getattr(args, "attn_weights_fp4", False),
        "attn_fp4_mx": getattr(args, "attn_fp4_mx", False),
        "use_autocast": getattr(args, "use_autocast", False),
        "use_autocast_fp16": getattr(args, "use_autocast_fp16", False),
        "fp16_half_backbone": getattr(args, "fp16_half_backbone", False),
        "fp32_ln": getattr(args, "fp32_ln", "none"),
        "fp32_softmax": getattr(args, "fp32_softmax", "none"),
        "fp16_linear_out": getattr(args, "fp16_linear_out", "none"),
        "fp16_ln": getattr(args, "fp16_ln", "none"),
        "fp16_softmax": getattr(args, "fp16_softmax", "none"),
        "max_samples": getattr(args, "max_samples", None),
        "n_samples_seen": n_samples_seen,
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
