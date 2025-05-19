import torch
from torchviz import make_dot
import unittest
import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation

# --- Import the necessary classes from your provided code ---
# (Assuming the code is in a file named 'equivariant_erwin.py')
from equi_erwin import (
    EquivariantNode,
    EquivariantErwinEmbedding,
    EquivariantBallAttention,
    EquivariantBallPooling,
    EquivariantBallUnpooling,
    EquivariantErwinBlock,
    EquivariantBasicLayer,
    EquiLinear, # Needed for Unpooling test
    GeoMLP,     # Needed for Block test
    EquiLayerNorm, # Needed for Block test
    embed_point,
    extract_point,
    construct_reference_multivector,
    EquivariantErwinTransformer
)
from gatr.layers.mlp.config import MLPConfig # Needed for Block test GeoMLP

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from typing import Literal, List, Optional, Tuple, Union
from dataclasses import dataclass

# Assume these imports from GATr library/primitives
from gatr.layers.linear import EquiLinear
from gatr.layers.attention.self_attention import SelfAttention
from gatr.layers.attention.config import SelfAttentionConfig
from gatr.layers.mlp.mlp import GeoMLP
from gatr.layers.mlp.config import MLPConfig
from gatr.layers.layer_norm import EquiLayerNorm
from gatr.interface import embed_point, extract_point, embed_scalar
from gatr.primitives import geometric_product # May be needed for MPNN
from gatr.utils.tensors import construct_reference_multivector
import torch_scatter
import torch_cluster



# Assume balltree library is available
from balltree import build_balltree_with_rotations

import torch
from torch.utils.tensorboard import SummaryWriter
num_points = 64 # Example number of points
batch_size = 1
c_in_scalar = 3  # Example: 3 initial scalar features
dimensionality = 3

# Model hyperparameters (example values)
mv_dims = [16, 32, 64]  # Channels for multivector features at each encoder level + bottleneck
s_dims = [8, 16, 32]    # Channels for scalar features
ball_sizes = [16, 16, 16] # Ball sizes for attention
enc_num_heads = [4, 4, 4]
enc_depths = [2, 2, 2]
dec_num_heads = [4, 4] # For two decoder layers (strides has len(mv_dims)-1)
dec_depths = [2, 2]
strides = [2, 2] # Two pooling/unpooling stages
rotate = True
mp_steps = 1 # Number of MPNN steps in embedding

model = EquivariantErwinTransformer(
    c_in_scalar=c_in_scalar,
    mv_dims=mv_dims,
    s_dims=s_dims,
    ball_sizes=ball_sizes,
    enc_num_heads=enc_num_heads,
    enc_depths=enc_depths,
    dec_num_heads=dec_num_heads,
    dec_depths=dec_depths,
    strides=strides,
    rotate=rotate,
    mlp_ratio=4,
    dimensionality=dimensionality,
    mp_steps=mp_steps,
    dropout=0.1,
    out_dim_scalar=c_in_scalar, # Example: predict same number of scalar features
    out_dim_cartesian=False
)
model.eval() # Important if you have dropout/batchnorm layers behaving differently

# --- Create Dummy Inputs ---
# These inputs need to have the correct shapes and types.
# The values can be random.
# Batching: Assume all points belong to a single batch item for simplicity here.
# (num_points,) for batch_idx
node_features_scalar = torch.randn(num_points, c_in_scalar)
node_positions_cartesian = torch.randn(num_points, dimensionality)
batch_idx = torch.zeros(num_points, dtype=torch.long)

# For the diagram, we might need to provide dummy tree_idx and tree_mask,
# or mock the build_balltree_with_rotations call if it's complex
# and not essential for tracing module connections.
# Assuming the balltree output is a permutation of all points:
permuted_indices = torch.randperm(num_points)
dummy_tree_idx = permuted_indices
dummy_tree_mask = torch.ones(num_points, dtype=torch.bool) # All points are valid

# Dummy edge_index for MPNN if mp_steps > 0
# Create a simple fully connected graph for dummy purposes, or random edges
if mp_steps > 0:
    # Example: each point connects to 2 other random points (ensure no self-loops for this example)
    src = torch.arange(num_points).repeat_interleave(2)
    dst = torch.randint(0, num_points, (num_points * 2,))
    # Remove self-loops from dummy edges
    valid_edges = src != dst
    dummy_edge_index = torch.stack([src[valid_edges], dst[valid_edges]], dim=0)
    if dummy_edge_index.shape[1] == 0: # if all were self-loops somehow
         dummy_edge_index = torch.tensor([[0],[1]]) if num_points >1 else torch.empty(2,0, dtype=torch.long) # fallback
else:
    dummy_edge_index = None

# Create a SummaryWriter instance
writer = SummaryWriter('runs/erwin_transformer_experiment')

# Add the graph to TensorBoard
# Note: The model's forward pass must be runnable with these dummy inputs.
# This includes handling or mocking external calls like build_balltree_with_rotations.
try:
    # Mocking the balltree call (same as Torchviz example)
    original_balltree_func = build_balltree_with_rotations
    def mock_balltree(pos, batch_idx_bt, strides_bt, ball_sizes_bt, rotate_bt):
        n = pos.shape[0]
        tree_idx_rot_list_dummy = []
        for _ in range(len(strides_bt)):
            tree_idx_rot_list_dummy.append(torch.randperm(n)) # Simplified dummy
        return torch.randperm(n), torch.ones(n, dtype=torch.bool), tree_idx_rot_list_dummy
    build_balltree_with_rotations = mock_balltree

    # For add_graph, it's best if the model directly accepts all inputs needed
    # Or, wrap the model call if it returns a dict.
    # TensorBoard's add_graph expects the model to be callable and inputs to be args/kwargs.
    # The model's forward method signature is:
    # forward(self, node_features_scalar, node_positions_cartesian, batch_idx,
    #         edge_index=None, tree_idx=None, tree_mask=None, radius=None, **kwargs)

    # We need to provide all positional arguments.
    # If `edge_index` is None and `mp_steps` is 0, this is fine.
    # If `mp_steps > 0` but `edge_index` is None, `radius_graph` is called.
    # Let's ensure `dummy_edge_index` is passed if `mp_steps > 0`.

    writer.add_graph(model, (node_features_scalar, node_positions_cartesian, batch_idx, dummy_edge_index))
    writer.close()
    print("TensorBoard graph data saved. Run 'tensorboard --logdir=runs' to view.")
    build_balltree_with_rotations = original_balltree_func # Restore

except Exception as e:
    print(f"Could not generate graph with TensorBoard due to: {e}")
    build_balltree_with_rotations = original_balltree_func # Restore