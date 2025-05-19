from typing import Optional, Union

import torch
from equi_erwin import EquivariantErwinEmbedding, EquivariantErwinTransformer
from experiments.datasets import CosmologyDataset
from gatr.interface import embed_point, embed_rotation, extract_point
from gatr.layers import EquiLinear
from gatr.primitives.bilinear import geometric_product
from gatr.primitives.linear import reverse
from models import ErwinTransformer
from torch.utils.data import DataLoader
from experiments.wrappers.cosmology_equi import CosmologyEquiModel
import sys

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
    print(batch['pos'].shape, batch['target'].shape, 
          batch['batch_idx'].shape, batch['edge_index'].shape)
    points=batch['pos']
    batch_idx = batch['batch_idx']
    break

quaternions = torch.tensor([0.1, 0.2, 0.3, 0.9])
quaternions = quaternions / quaternions.norm()

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
    "mp_steps": 0, # no MPNN
    "decode": True, # no decoder
    "dimensionality": 2, # for visualization
    "rotate": 0,
}

points_mv = embed_point(points)
qs = quaternions.unsqueeze(0) if quaternions.ndim == 1 else quaternions
rotation_mv  = embed_rotation(qs).unsqueeze(1)
rotation_inv = reverse(rotation_mv)
tmp    = geometric_product(rotation_mv, points_mv)   # (B,1,16)
mv_rotated  = geometric_product(tmp, rotation_inv) # (B,1,16)
points_rot = extract_point(mv_rotated).squeeze(0)

#run model
equimodel = EquivariantErwinTransformer(**config)
model = CosmologyEquiModel(equimodel)
cos_conf = {"batch_idx":batch_idx}
out_mv, out_s = model(points, **cos_conf)

#run on rotated
out_mv_r, out_s_r = model(points_rot, **cos_conf)

C_mv    = out_mv.shape[1]
r_mv_bc = rotation_mv.expand(-1, C_mv, -1)  # (B,C_mv,16)
r_inv_bc= rotation_inv.expand(-1, C_mv, -1) # (B,C_mv,16)

tmp2       = geometric_product(r_mv_bc, out_mv)
out_mv_rot = geometric_product(tmp2,     r_inv_bc)  # (B,C_mv,16)

mv_close = torch.allclose(out_mv_rot, out_mv_r, atol=1e-3)

print(mv_close)