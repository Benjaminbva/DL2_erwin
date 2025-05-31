#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_cluster import knn_graph

from experiments.datasets import CosmologyDataset
from experiments.wrappers.cosmology import CosmologyModel
from experiments.wrappers.cosmology_equi import CosmologyEquiModel
from models.erwin import ErwinTransformer
from models.GATrErwin import EquivariantErwinTransformer
from gatr_utils.interface.point import extract_point

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def sample_so3(rng: np.random.Generator, angle_degrees: float = None) -> torch.Tensor:
    if angle_degrees is not None:
        # Generate rotation with specific angle
        angle_rad = math.radians(angle_degrees)
        # Random axis
        axis = rng.standard_normal(3)
        axis = axis / np.linalg.norm(axis)
        # Rodrigues' rotation formula
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + sin_angle * K + (1 - cos_angle) * np.dot(K, K)
        return torch.tensor(R, dtype=torch.float32)
    else:
        # Original random rotation
        q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        return torch.tensor(q, dtype=torch.float32)


def load_model(ckpt_path: str, equi: bool, device: torch.device):
    print(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]

    if equi:
        net = EquivariantErwinTransformer(
            mv_dim_in=cfg["mv_dim_in"], mv_dims=cfg["mv_dims"], s_dims=cfg["s_dims"],
            enc_num_heads=cfg["enc_num_heads"], enc_depths=cfg["enc_depths"],
            dec_num_heads=cfg["dec_num_heads"], dec_depths=cfg["dec_depths"],
            strides=cfg["strides"], ball_sizes=cfg["ball_sizes"], rotate=cfg["rotate"],
            mp_steps=cfg.get("mp_steps", 0), use_rad=cfg.get("use_rad", True),
            rbf_dim=cfg.get("rbf_dim", 0), max_l=cfg.get("max_l", 0),
            cutoff=cfg.get("cutoff", 0.0),
        ).to(device)
        model = CosmologyEquiModel(net).to(device)
    else:
        net = ErwinTransformer(
            c_in=cfg["c_in"], c_hidden=cfg["c_hidden"],
            enc_num_heads=cfg["enc_num_heads"], enc_depths=cfg["enc_depths"],
            dec_num_heads=cfg["dec_num_heads"], dec_depths=cfg["dec_depths"],
            strides=cfg["strides"], ball_sizes=cfg["ball_sizes"], rotate=cfg["rotate"],
            mp_steps=cfg.get("mp_steps", 0),
        ).to(device)
        model = CosmologyModel(net).to(device)

    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    return model


def make_loader(root: Path, num_samples: int, batch_size: int, skip_std: bool):
    ds = CosmologyDataset(
        task="node", split="test", num_samples=num_samples,
        tfrecords_path=str(root), knn=10, skip_standardize=skip_std,
    )
    ld = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    collate_fn=ds.collate_fn, num_workers=4, pin_memory=True)
    return ds, ld


def apply_transform(batch: Dict[str, torch.Tensor], R: torch.Tensor | None, t: torch.Tensor | None, k: int, skip_std: bool = False):
    """Return a *new* batch dict on **same device** with transformed pos/vel/edges."""
    pos = batch["pos"]
    vel = batch["target"]
    bidx = batch["batch_idx"]

    if R is not None:
        # centroid per halo for rotation around halo center
        num_halo = int(bidx.max().item()) + 1
        centroid = torch.zeros(num_halo, 3, device=pos.device)
        counts   = torch.zeros(num_halo, 1, device=pos.device)
        centroid.index_add_(0, bidx, pos)
        counts.index_add_(0, bidx, torch.ones_like(pos[:, :1]))
        centroid = centroid / counts.clamp_min(1)

        pos = (pos - centroid[bidx]) @ R.T + centroid[bidx]
        vel = vel @ R.T

    if t is not None:
        if skip_std:
            # Raw coordinates: apply translation directly
            pos = pos + t
        else:
            # Standardized coordinates: scale translation by position std
            # Standard deviations for x, y, z from COSMOLOGY dataset
            pos_std = torch.tensor([288.71, 288.75, 288.70], device=pos.device)
            scaled_t = t / pos_std
            pos = pos + scaled_t

    edge = knn_graph(pos, k=k, batch=bidx).to(pos.device)
    return {**batch, "pos": pos, "target": vel, "edge_index": edge}


def make_forward(model, equi: bool):
    if equi:
        def _f(b):
            pred_mv, pred_s = model(b["pos"], batch_idx=b["batch_idx"], edge_index=b["edge_index"])
            return extract_point(pred_mv).squeeze()
    else:
        def _f(b):
            return model(b["pos"], batch_idx=b["batch_idx"], edge_index=b["edge_index"])
    return _f


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--equi", action="store_true", default=False)
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--mode", choices=["rotation", "translation", "both", "none"], required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-samples", type=int, default=512)
    ap.add_argument("--t-min", type=float, default=-200.0)
    ap.add_argument("--t-max", type=float, default=200.0)
    ap.add_argument("--translation-magnitude", type=float, default=None, help="Fixed translation magnitude (overrides t-min/t-max)")
    ap.add_argument("--rotation-angle", type=float, default=20.0, help="Rotation angle in degrees (default: 20.0)")
    args = ap.parse_args()

    torch.manual_seed(args.seed or random.randint(0, 2 ** 31 - 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rng = np.random.default_rng(args.seed)
    R = sample_so3(rng, args.rotation_angle) if args.mode in {"rotation", "both"} else None
    if R is not None:
        R = R.to(device)
    if args.mode in {"translation", "both"}:
        if args.translation_magnitude is not None:
            # Fixed magnitude, random direction
            direction = rng.standard_normal(3)
            direction = direction / np.linalg.norm(direction)
            t = torch.tensor(direction * args.translation_magnitude, dtype=torch.float32)
        else:
            # Original random range
            t = torch.tensor(rng.uniform(args.t_min, args.t_max, (3,)), dtype=torch.float32)
    else:
        t = None
    if t is not None:
        t = t.to(device)

    print("Transform:")
    if R is not None:
        ang = math.degrees(torch.acos((torch.trace(R) - 1) / 2).item())
        print(f"  rotation angle ≈ {ang:.1f}°")
    if t is not None:
        print(f"  |t|_2 = {t.norm().item():.3f}")

    # Always use standardization unless explicitly specified
    skip_std = False  # Force standardization to match training
    ds, loader = make_loader(Path(args.data_path), args.num_samples, args.batch_size, skip_std)

    model   = load_model(args.ckpt, args.equi, device)
    forward = make_forward(model, args.equi)

    base_tot = equiv_tot = n_elem = 0.0
    k_nn = ds.knn

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # Original predictions for baseline MSE
            y_orig = forward(batch)
            base_tot += F.mse_loss(y_orig, batch["target"], reduction="sum").item()

            if args.mode == "none":
                # No transformation applied - equivariance MSE should be 0
                equiv_tot += 0.0
            else:
                # NEW APPROACH: (1) Transform X and Y, (2) forward X to get Y', (3) compare Y' with transformed Y
                batch_transformed = apply_transform(batch, R, t, k_nn, skip_std)
                y_pred_on_transformed = forward(batch_transformed)  # Step 2: forward transformed X to get Y'
                
                # Step 3: compare Y' with correctly transformed Y
                equiv_tot += F.mse_loss(y_pred_on_transformed, batch_transformed["target"], reduction="sum").item()

            n_elem += y_orig.numel()

            del y_orig
            torch.cuda.empty_cache()
    model_parts = args.ckpt.split('/')[2].split('_')
    exp = '_'.join(model_parts[:-2])
    print(f"{exp}\t{base_tot / n_elem}")
    print(f"{exp}\t{equiv_tot / n_elem}")


if __name__ == "__main__":
    main()
