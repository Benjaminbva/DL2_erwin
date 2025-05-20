from typing import Optional, Union

import torch
from equi_erwin import EquivariantErwinEmbedding, EquivariantErwinTransformer
from experiments.datasets import CosmologyDataset
from gatr.interface import embed_point, embed_rotation
from gatr.layers import EquiLinear
from gatr.primitives.bilinear import geometric_product
from gatr.primitives.linear import reverse
from models import ErwinTransformer
from torch.utils.data import DataLoader
from experiments.wrappers.cosmology_equi import CosmologyEquiModel
import sys

def check_equivariance_quaternions(
    layer: EquivariantErwinEmbedding,
    points: torch.Tensor,        # (3,) or (B,3)
    quaternions: torch.Tensor,   # (4,) or (B,4)
    cos_conf,
    atol: float = 1e-6
) -> bool:
    """
    Returns True if for each i:
      layer(embed_point(points[i])) rotated by quaternion[i]
        == layer(embed_point(points[i] rotated by quaternion[i]))
    Checks both multivector channels and scalar channels.
    """

    # --- 1) Batchify points and scalars ---
    pts = points.unsqueeze(0) if points.ndim == 1 else points  # (B,3)
    B = pts.shape[0]

    # --- 2) Embed points to PGA multivectors ---
    #    embed_point: (...,3) -> (...,16)
    p_mv = embed_point(pts)               # (B,16)
    p_mv = p_mv.unsqueeze(1)              # (B,1,16)

    # --- 3) Forward on original points ---
    out_mv, out_s = layer(p_mv, **cos_conf)       # out_mv: (B,C_mv,16), out_s: (B,C_s)

    # --- 4) Prepare rotors & inverses ---
    # broadcast single quaternion if needed
    qs = quaternions.unsqueeze(0) if quaternions.ndim == 1 else quaternions  # (B,4)
    r_mv  = embed_rotation(qs)            # (B,16)
    r_mv  = r_mv.unsqueeze(1)             # (B,1,16)
    r_inv = reverse(r_mv)                 # (B,1,16)

    # --- 5) Rotate inputs via sandwich product: r ⋅ p ⋅ r⁻¹ ---
    tmp    = geometric_product(r_mv,    p_mv)   # (B,1,16)
    p_rot  = geometric_product(tmp,     r_inv) # (B,1,16)

    # --- 6) Forward on rotated inputs ---
    out_mv_r, out_s_r = layer(p_rot, **cos_conf)

    # --- 7) Rotate the original outputs the same way ---
    C_mv    = out_mv.shape[1]
    r_mv_bc = r_mv.expand(-1, C_mv, -1)  # (B,C_mv,16)
    r_inv_bc= r_inv.expand(-1, C_mv, -1) # (B,C_mv,16)

    tmp2       = geometric_product(r_mv_bc, out_mv)
    out_mv_rot = geometric_product(tmp2,     r_inv_bc)  # (B,C_mv,16)

    # --- 8) Compare multivector & scalar outputs ---
    mv_close = torch.allclose(out_mv_rot, out_mv_r, atol=atol)
    s_close  = torch.allclose(out_s,      out_s_r,   atol=atol)

    if not (mv_close and s_close):
        print(f"Equivariance failed: mv_match={mv_close}, scalar_match={s_close}")

    return mv_close and s_close




# 3) build data and layer
train_dataset = CosmologyDataset(
        task='node', 
        split='train', 
        num_samples=32, 
        tfrecords_path='./data/cosmology', 
        knn=10,
    )

train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=16,
    )

for batch in train_loader:
    #print(batch['pos'].shape, batch['target'].shape, 
    #      batch['batch_idx'].shape, batch['edge_index'].shape)
    point=batch['pos']
    batch_idx = batch['batch_idx']
    break

#point      = torch.tensor([1., 2., 3.])
quaternion = torch.tensor([0.1, 0.2, 0.3, 0.9])
quaternion = quaternion / quaternion.norm()

config_w_pool = {
    "c_in": 16,
    "c_hidden": [16,32],
    "ball_sizes": [128, 128],
    "enc_num_heads": [1,1],
    "enc_depths": [1,1],
    "dec_num_heads": [4],
    "dec_depths": [4],
    "strides": [4,], # 0.25 coarsening
    "mp_steps": 0, # no MPNN
    "decode": True, # no decoder
    "dimensionality": 3, # for visualization
    "rotate": 0,
}

config = {
    "mv_dim_in": 16,
    "mv_dims": [16,32],
    "s_dims": [16,32],
    "ball_sizes": [128, 128],
    "enc_num_heads": [1,1],
    "enc_depths": [1,1],
    "dec_num_heads": [4],
    "dec_depths": [4],
    "strides": [4,], # 0.25 coarsening
    "mp_steps": 2, # no MPNN
    "decode": True, # no decoder
    "dimensionality": 2, # for visualization
    "rotate": 0,
}


equimodel = EquivariantErwinTransformer(**config)
#normalmodel = ErwinTransformer(**config_w_pool)

model = CosmologyEquiModel(equimodel)
cos_conf = {"batch_idx":batch_idx,
            "radius" : torch.tensor(2.0)}
#print(check_equivariance_quaternions(model, point, quaternion, cos_conf))
equiout = model(point, **cos_conf)
print(equiout)
#equimodel.train()
#model.train()
#equiout = model(point, **cos_conf)
#normalmodel(torch.ones(10000,16), point, batch_idx)
#assert check_equivariance_quaternions(layer, point, quaternion), "Embedding is not equivariant!"