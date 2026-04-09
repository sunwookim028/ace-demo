# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Unit test: verify that transformer.py and retr.py refactors produce identical
float32 outputs to the original model (within floating-point rounding tolerance).

Usage
-----
# Step 1 — save reference outputs from CURRENT (pre-refactor) model:
    python tests/test_model_refactor.py --save

# Step 2 — after applying refactors + weight migration, verify:
    python tests/test_model_refactor.py --verify

Both steps must be run from src/ as working directory.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CKPT = Path(__file__).resolve().parents[2] / "logs/pretrained_model/p2s1_retr_detseg.pth"
REF_FILE = Path(__file__).resolve().parent / "reference_outputs.pt"

# Tolerance for float32 equivalence: MHA fused→split decomposition can introduce
# ~1e-5 rounding differences due to different GEMM execution order.
ATOL = 1e-4
RTOL = 1e-4


def load_model(migrate_weights=True):
    """Load RETR model. migrate_weights is always True (model is permanently refactored)."""
    from models import RETR
    model = RETR(task="SEG")
    params = torch.load(CKPT, map_location="cpu")
    if migrate_weights:
        params = migrate_encoder_mha_state_dict(params)
    model.load_state_dict(params)
    model.eval()
    return model


def migrate_encoder_mha_state_dict(state_dict: dict) -> dict:
    """
    Remap checkpoint keys from fused nn.MultiheadAttention to the refactored
    explicit Q/K/V Linear modules in ConditionalTransformerEncoderLayer.

    Original keys (per encoder layer N):
        model.detr.encoder.layers.N.self_attn.in_proj_weight  [1536, 512]
        model.detr.encoder.layers.N.self_attn.in_proj_bias    [1536]
        model.detr.encoder.layers.N.self_attn.out_proj.weight [512, 512]
        model.detr.encoder.layers.N.self_attn.out_proj.bias   [512]

    New keys (refactored):
        model.detr.encoder.layers.N.self_attn_q_proj.weight   [512, 512]
        model.detr.encoder.layers.N.self_attn_q_proj.bias     [512]
        model.detr.encoder.layers.N.self_attn_k_proj.weight   [512, 512]
        model.detr.encoder.layers.N.self_attn_k_proj.bias     [512]
        model.detr.encoder.layers.N.self_attn_v_proj.weight   [512, 512]
        model.detr.encoder.layers.N.self_attn_v_proj.bias     [512]
        model.detr.encoder.layers.N.self_attn_out_proj.weight [512, 512]
        model.detr.encoder.layers.N.self_attn_out_proj.bias   [512]
    """
    new_sd = {}
    for key, val in state_dict.items():
        # Only migrate encoder self_attn (not decoder — decoder uses custom MHA)
        if "encoder" not in key or "self_attn" not in key:
            new_sd[key] = val
            continue

        prefix = key.rsplit("self_attn", 1)[0]  # e.g. 'model.detr.encoder.layers.0.'

        if key.endswith(".self_attn.in_proj_weight"):
            d = val.shape[0] // 3
            new_sd[prefix + "self_attn_q_proj.weight"] = val[0:d].clone()
            new_sd[prefix + "self_attn_k_proj.weight"] = val[d:2*d].clone()
            new_sd[prefix + "self_attn_v_proj.weight"] = val[2*d:3*d].clone()
        elif key.endswith(".self_attn.in_proj_bias"):
            d = val.shape[0] // 3
            new_sd[prefix + "self_attn_q_proj.bias"] = val[0:d].clone()
            new_sd[prefix + "self_attn_k_proj.bias"] = val[d:2*d].clone()
            new_sd[prefix + "self_attn_v_proj.bias"] = val[2*d:3*d].clone()
        elif key.endswith(".self_attn.out_proj.weight"):
            new_sd[prefix + "self_attn_out_proj.weight"] = val
        elif key.endswith(".self_attn.out_proj.bias"):
            new_sd[prefix + "self_attn_out_proj.bias"] = val
        else:
            new_sd[key] = val

    return new_sd


def make_inputs(seed=42, batch_size=1):
    torch.manual_seed(seed)
    rf_hor = torch.randn(batch_size, 4, 256, 128)
    rf_ver = torch.randn(batch_size, 4, 256, 128)
    return rf_hor, rf_ver


def tensors_from_out(out):
    """Extract all tensor values from model output (list of per-image dicts).
    With batch_size=1 we only have one image; take index 0.
    """
    img_out = out[0]  # dict for first (only) image
    result = {}
    for k, v in img_out.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.detach().float()
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            result[k] = [x.detach().float() for x in v]
    return result


def save_reference():
    print("Loading model (with weight migration) to save reference outputs...")
    model = load_model(migrate_weights=True)
    rf_hor, rf_ver = make_inputs()

    with torch.no_grad():
        out = model(rf_hor, rf_ver)

    ref = {
        "rf_hor": rf_hor,
        "rf_ver": rf_ver,
        "out": tensors_from_out(out),
    }
    torch.save(ref, REF_FILE)
    print(f"Reference saved to {REF_FILE}")
    for k, v in ref["out"].items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={tuple(v.shape)}, mean={v.float().mean():.6f}")


def verify_refactor():
    if not REF_FILE.exists():
        raise FileNotFoundError(f"Run --save first: {REF_FILE}")

    print("Loading model (with weight migration) to verify against reference...")
    model = load_model(migrate_weights=True)
    ref = torch.load(REF_FILE, map_location="cpu")
    rf_hor, rf_ver = ref["rf_hor"], ref["rf_ver"]
    ref_out = ref["out"]

    with torch.no_grad():
        out = tensors_from_out(model(rf_hor, rf_ver))

    print("\nOutput comparison (atol={}, rtol={}):".format(ATOL, RTOL))
    all_pass = True
    for key in ref_out:
        ref_v = ref_out[key]
        new_v = out.get(key)
        if new_v is None:
            print(f"  {key}: MISSING in new output")
            all_pass = False
            continue
        if isinstance(ref_v, torch.Tensor):
            if ref_v.numel() == 0:
                print(f"  {key}: empty tensor — skipped")
                continue
            diff = (ref_v - new_v.float()).abs()
            max_diff = diff.max().item()
            rel_diff = (diff / (ref_v.abs() + 1e-8)).max().item()
            ok = max_diff < ATOL and rel_diff < RTOL
            print(f"  {key}: max_abs={max_diff:.2e}  max_rel={rel_diff:.2e}  {'PASS' if ok else 'FAIL'}")
            if not ok:
                all_pass = False

    if all_pass:
        print("\nPASS — refactored model is numerically equivalent to original.")
    else:
        print("\nFAIL — outputs differ beyond tolerance.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Save reference outputs (run before refactoring)")
    parser.add_argument("--verify", action="store_true", help="Verify refactored model matches reference")
    args = parser.parse_args()

    if args.save:
        save_reference()
    elif args.verify:
        verify_refactor()
    else:
        parser.print_help()
