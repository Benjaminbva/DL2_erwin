import torch
import torch.nn as nn
from einops import rearrange, reduce
from typing import Literal, List, Optional
from dataclasses import dataclass

from gatr.layers.linear import EquiLinear
from gatr.layers.attention.self_attention import SelfAttention
from gatr.layers.attention.config import SelfAttentionConfig
from gatr.layers.mlp.mlp import GeoMLP
from gatr.layers.mlp.config import MLPConfig
from gatr.layers.layer_norm import EquiLayerNorm
from gatr.interface import embed_point,embed_translation
from gatr.utils.tensors import construct_reference_multivector
import torch_cluster
import sys
import os
sys.path.append("../../")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from balltree import build_balltree_with_rotations

import torch
import torch.nn as nn

import math
#from torch_scatter import scatter_mean

class BesselBasisFunctions(nn.Module):
    """Bessel-style radial basis functions."""
    def __init__(self, num_rbf: int, cutoff: float, trainable: bool = False):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff

        # Fixed frequency parameters (n*pi/cutoff)
        freqs = torch.arange(1, num_rbf + 1) * math.pi / cutoff
        self.register_buffer("freqs", freqs)

    def cutoff_fn(self, distances: torch.Tensor) -> torch.Tensor:
        """C¹-continuous cosine cutoff function."""
        # Return 0 for distances beyond cutoff
        mask = (distances < self.cutoff).float()
        return 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1) * mask

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Apply Bessel basis functions to distances."""
        # Reshape distances for broadcasting
        distances = distances.unsqueeze(-1)  # (..., 1)

        # Apply smooth cutoff function
        f_cut = self.cutoff_fn(distances)

        # Compute Bessel-inspired basis: sin(n*π*r/rc)/r
        # Avoid singularity at r=0 with small epsilon
        safe_dist = distances + 1e-8
        bessel = torch.sin(self.freqs * distances) / safe_dist

        return bessel * f_cut

def cosine_cutoff(distances: torch.Tensor, cutoff: float) -> torch.Tensor:
    """C¹-continuous cosine cutoff function."""
    # Return 0 for distances beyond cutoff
    mask = (distances < cutoff).float()
    return 0.5 * (torch.cos(distances * math.pi / cutoff) + 1) * mask

class SphericalHarmonics(nn.Module):
    """Proper real spherical harmonics for angular basis functions."""
    def __init__(self, max_l: int):
        super().__init__()
        self.max_l = max_l
        self.num_spherical = (max_l + 1)**2  # Total number of harmonics

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        """Compute real spherical harmonics for given vectors."""
        # Normalize vectors
        norm = torch.norm(vectors, dim=-1, keepdim=True)
        directions = vectors / (norm + 1e-10)

        # Extract xyz components
        x, y, z = directions[..., 0], directions[..., 1], directions[..., 2]

        # Initialize result tensor
        result = torch.empty((*directions.shape[:-1], self.num_spherical),
                              dtype=directions.dtype, device=directions.device)

        # l=0 (1 component)
        result[..., 0] = 0.2820947917738781  # 1/2 * sqrt(1/π)

        if self.max_l >= 1:
            # l=1 (3 components)
            result[..., 1] = 0.4886025119029199 * y  # Y_1,-1
            result[..., 2] = 0.4886025119029199 * z  # Y_1,0
            result[..., 3] = 0.4886025119029199 * x  # Y_1,1

        if self.max_l >= 2:
            # l=2 (5 components) with correct constants
            result[..., 4] = 0.5462742152960396 * x * y              # Y_2,-2
            result[..., 5] = 0.5462742152960396 * y * z              # Y_2,-1
            result[..., 6] = 0.6307831305050401 * (3*z*z - 1) / 2.0  # Y_2,0
            result[..., 7] = 0.5462742152960396 * x * z              # Y_2,1
            result[..., 8] = 0.5462742152960396 * (x*x - y*y) / 2.0  # Y_2,2

        return result


class RadMPNN(nn.Module):
    """Enhanced Message Passing Neural Network using learned radial and angular basis functions."""
    def __init__(self, scalar_feature_dim: int, mp_steps: int, 
                rbf_dim: int = 16, max_l: int = 2, cutoff: float = 10.0,
                rbf_subset_size: int = 8):
        super().__init__()
        self.mp_steps = mp_steps
        self.max_l = max_l
        self.cutoff = cutoff
        self.rbf_dim = rbf_dim
        self.rbf_subset_size = min(rbf_subset_size, rbf_dim)

        # Use Bessel basis functions
        self.rbf = BesselBasisFunctions(rbf_dim, cutoff)
        self.sph = SphericalHarmonics(max_l)

        # Calculate feature dimensions
        angular_dim = (max_l + 1)**2
        if max_l == 0:
            edge_feat_dim = rbf_dim + angular_dim  # Just concatenate
        else:
            # RBF * l=0 + subset of RBF * higher components
            edge_feat_dim = rbf_dim + self.rbf_subset_size * (angular_dim - 1)

        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_feat_dim, scalar_feature_dim),
            nn.GELU(),
            nn.LayerNorm(scalar_feature_dim)
        )

        self.message_fns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * scalar_feature_dim + scalar_feature_dim, scalar_feature_dim),
                nn.GELU(),
                nn.LayerNorm(scalar_feature_dim)
            ) for _ in range(mp_steps)
        ])

        self.update_fns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * scalar_feature_dim, scalar_feature_dim),
                nn.LayerNorm(scalar_feature_dim)
            ) for _ in range(mp_steps)
        ])

    def compute_edge_attr(self, cartesian_pos: torch.Tensor, edge_index: torch.Tensor):
        row, col = edge_index
        rel_vec = cartesian_pos[row] - cartesian_pos[col]
        dist = torch.norm(rel_vec, dim=-1)

        # Apply radial basis functions
        rbf_feats = self.rbf(dist)  # (num_edges, rbf_dim)

        # Apply spherical harmonics for angular features
        sph_feats = self.sph(rel_vec)  # (num_edges, angular_dim)

        # Create cutoff mask for spherical harmonics
        cutoff_mask = (dist < self.cutoff).float().unsqueeze(-1)
        # Apply cutoff to spherical harmonics
        sph_feats = sph_feats * cutoff_mask

        # Memory-efficient feature combination
        if self.max_l == 0:
            edge_feats = torch.cat([rbf_feats, sph_feats], dim=-1)
        else:
            # Combine RBF with l=0 component
            l0_feats = rbf_feats * sph_feats[..., 0:1]

            # Combine subset of RBF with higher l components
            higher_l_feats = torch.einsum("eb,es->ebs",
                                           rbf_feats[:, :self.rbf_subset_size],
                                           sph_feats[..., 1:])
            higher_l_feats = higher_l_feats.reshape(rel_vec.shape[0], -1)

            edge_feats = torch.cat([l0_feats, higher_l_feats], dim=-1)

        return self.edge_embedding(edge_feats)

    def forward(self, scalar_features: torch.Tensor, cartesian_pos: torch.Tensor, edge_index: torch.Tensor):
        h_s = scalar_features
        if self.mp_steps == 0:
            return h_s

        edge_attr = self.compute_edge_attr(cartesian_pos, edge_index)
        row, col = edge_index

        for message_fn, update_fn in zip(self.message_fns, self.update_fns):
            # Message computation
            message_inputs = torch.cat([h_s[row], h_s[col], edge_attr], dim=-1)
            messages = message_fn(message_inputs)

            # Aggregation
            aggregated_messages = scatter_mean(messages, col, h_s.size(0))

            # Update
            update_inputs = torch.cat([h_s, aggregated_messages], dim=-1)
            h_s_update = update_fn(update_inputs)
            h_s = h_s + h_s_update  # Residual connection

        return h_s

@dataclass
class EquivariantNode:
    """ Dataclass to store the hierarchical node information."""
    x_mv: torch.Tensor               # Multivector features (..., C_mv, 16)
    pos_mv: torch.Tensor             # Multivector positions (..., 1, 16) - embedded points
    x_s: Optional[torch.Tensor] = None # Optional scalar features (..., C_s)
    pos_cartesian: Optional[torch.Tensor] = None # Original Cartesian coords (..., D), potentially needed for tree building
    batch_idx: Optional[torch.Tensor] = None     # Batch indices (N,)
    tree_idx_rot: Optional[torch.Tensor] = None  # Indices for rotated tree permutation
    children: Optional['EquivariantNode'] = None # Link to finer level node during unpooling

def scatter_mean(src: torch.Tensor, idx: torch.Tensor, num_receivers: int):
    """ 
    Averages all values from src into the receivers at the indices specified by idx.

    Args:
        src (torch.Tensor): Source tensor of shape (N, D).
        idx (torch.Tensor): Indices tensor of shape (N,).
        num_receivers (int): Number of receivers (usually the maximum index in idx + 1).
    
    Returns:
        torch.Tensor: Result tensor of shape (num_receivers, D).
    """
    result = torch.zeros(num_receivers, src.size(1), dtype=src.dtype, device=src.device)
    count = torch.zeros(num_receivers, dtype=torch.long, device=src.device)
    result.index_add_(0, idx, src)
    count.index_add_(0, idx, torch.ones_like(idx, dtype=torch.long))
    return result / count.unsqueeze(1).clamp(min=1)

class InvMPNN(nn.Module):
    """ 
    Message Passing Neural Network (see Gilmer et al., 2017).
        m_ij = MLP([h_i, h_j, pos_i - pos_j])       message
        m_i = mean(m_ij)                            aggregate
        h_i' = MLP([h_i, m_i])                      update
    """
    def __init__(self, scalar_feature_dim: int, mp_steps: int, distance_dim: int = 1):
        super().__init__()
        self.mp_steps = mp_steps

        self.intial_proj = nn.Linear(1, scalar_feature_dim)

        self.message_fns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * scalar_feature_dim + distance_dim, scalar_feature_dim),
                nn.GELU(),
                nn.LayerNorm(scalar_feature_dim)
            ) for _ in range(mp_steps)
        ])
        self.update_fns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * scalar_feature_dim, scalar_feature_dim),
                nn.LayerNorm(scalar_feature_dim)
            ) for _ in range(mp_steps)
        ])

    @torch.no_grad()
    def compute_edge_attr(self, cartesian_pos: torch.Tensor, edge_index: torch.Tensor):
        row, col = edge_index
        dist = torch.norm(cartesian_pos[row] - cartesian_pos[col], dim=-1, keepdim=True)
        return dist

    def forward(self, scalar_features: torch.Tensor, cartesian_pos: torch.Tensor, edge_index: torch.Tensor):
        h_s = scalar_features
        if self.mp_steps == 0:
            return h_s

        edge_attr = self.compute_edge_attr(cartesian_pos, edge_index)
        row, col = edge_index
        if h_s is None:
            h_s = self.intial_proj(edge_attr)
            h_s = scatter_mean(h_s, col, cartesian_pos.size(0))
        
        for message_fn, update_fn in zip(self.message_fns, self.update_fns):
            message_inputs = torch.cat([h_s[row], h_s[col], edge_attr], dim=-1)
            messages = message_fn(message_inputs)

            aggregated_messages = scatter_mean(messages, col, h_s.size(0))

            update_inputs = torch.cat([h_s, aggregated_messages], dim=-1)
            h_s_update = update_fn(update_inputs)

        return h_s + h_s_update


class EquivariantErwinEmbedding(nn.Module):
    """ Linear projection -> MPNN."""
    def __init__(self, 
                 in_mv_dim: int, 
                 out_mv_dim: int, 
                 in_s_dim: Optional[int] = None,
                 out_s_dim: Optional[int] = None,
                 mp_steps: int = 0, 
                 dimensionality: int = 3):
        super().__init__()

        self.mp_steps = mp_steps
        self.embed_fn = EquiLinear(
            in_mv_channels=in_mv_dim, 
            out_mv_channels=out_mv_dim,
            in_s_channels=in_s_dim,
            out_s_channels=out_s_dim

        )
        self.mpnn = InvMPNN(scalar_feature_dim=out_mv_dim, mp_steps=mp_steps, distance_dim=1)

    def forward(self, x_mv: torch.Tensor, x_s: Optional[torch.Tensor], cartesian_pos: torch.Tensor = torch.Tensor([0]), edge_index: torch.Tensor = torch.tensor([0])):
        x_mv, x_s = self.embed_fn(x_mv, x_s)
        if self.mp_steps > 0 :
            x_s = self.mpnn(x_s, cartesian_pos, edge_index)
            #x_mv[:,:,0] = x_mv[:,:,0] + x_s
            return x_mv, x_s
        else:
            return x_mv, x_s
        
class RadEquivariantErwinEmbedding(nn.Module):
    def _init_(self, 
                 in_mv_dim: int, 
                 out_mv_dim: int, 
                 in_s_dim: Optional[int] = None,
                 out_s_dim: Optional[int] = None,
                 mp_steps: int = 0, 
                 dimensionality: int = 3,
                 rbf_dim: int = 16,
                 max_l: int = 2,
                 cutoff: float = 10.0):
        super()._init_()

        self.mp_steps = mp_steps
        self.embed_fn = EquiLinear(
            in_mv_channels=in_mv_dim, 
            out_mv_channels=out_mv_dim,
            in_s_channels=in_s_dim,
            out_s_channels=out_s_dim
        )
        self.mpnn = RadMPNN(scalar_feature_dim=out_s_dim,
                            mp_steps=mp_steps
                            )

    def forward(self, x_mv: torch.Tensor, x_s: Optional[torch.Tensor], cartesian_pos: torch.Tensor = torch.Tensor([0]), edge_index: torch.Tensor = torch.tensor([0])):
        x_mv, x_s = self.embed_fn(x_mv) #Bx1x16 -> #Bxout_cx16
        if x_s is None and self.mp_steps > 0:
            row, col = edge_index
            x_s = torch.norm(cartesian_pos[row] - cartesian_pos[col], dim=-1, keepdim=True)
        if self.mp_steps > 0 :
            x_s = self.mpnn(x_s, cartesian_pos, edge_index)
            return x_mv, x_s
        else:
            return x_mv, x_s


class EquivariantBallAttention(nn.Module):
    """ Equivariant Ball Multi-Head Self-Attention (BMSA) using GATr's SelfAttention. """
    def __init__(self, mv_dim: int, s_dim: Optional[int], num_heads: int, ball_size: int, dropout: float = 0.0, increase_hidden_channels_factor: int = 2): # Added increase_hidden_channels_factor
        super().__init__()
        self.ball_size = ball_size

        attn_config = SelfAttentionConfig(
            in_mv_channels=mv_dim,
            out_mv_channels=mv_dim,
            in_s_channels=s_dim,
            out_s_channels=s_dim,
            num_heads=num_heads,
            dropout_prob=dropout,
            pos_encoding=False,
            increase_hidden_channels=increase_hidden_channels_factor

        )
        self.attention = SelfAttention(config=attn_config)
        self.sigma_att = nn.Parameter(-1 + 0.01 * torch.randn((1, num_heads, 1, 1)))

    @torch.no_grad()
    def create_attention_mask(self, pos: torch.Tensor):
        """ Distance-based attention bias (eq. 10). """
        pos = rearrange(pos, '(n m) d -> n m d', m=self.ball_size)
        return self.sigma_att * torch.cdist(pos, pos, p=2).unsqueeze(1)
    
    def forward(self, x_mv: torch.Tensor, x_s: Optional[torch.Tensor], pos_mv: torch.Tensor, pos_cartesian: torch.Tensor, batch_idx: torch.Tensor):


        num_total_nodes = x_mv.shape[0]

        num_balls = num_total_nodes // self.ball_size

        x_mv_batched = rearrange(x_mv, '(b n) c val -> b n c val', b=num_balls, n=self.ball_size)
        x_s_batched = None
        if x_s is not None:
            x_s_batched = rearrange(x_s, '(b n) cs -> b n cs', b=num_balls, n=self.ball_size)


        out_mv, out_s = self.attention(x_mv_batched, scalars=x_s_batched,
                                       attention_mask = self.create_attention_mask(pos_cartesian))
        out_mv = rearrange(out_mv, 'b n c val -> (b n) c val')
        if out_s is not None:
            out_s = rearrange(out_s, 'b n cs -> (b n) cs')

        return out_mv, out_s


class EquivariantBallPooling(nn.Module):
    """ Equivariant pooling using mean aggregation and EquiLinear projection. """
    def __init__(self, in_mv_dim: int, in_s_dim: Optional[int], out_mv_dim: int, out_s_dim: Optional[int], stride: int, dimensionality: int = 3):
        super().__init__()
        self.stride = stride
        self.in_mv_dim = in_mv_dim
        self.out = out_mv_dim
        self.proj = EquiLinear(in_mv_dim * stride + stride, out_mv_dim, 
                               in_s_dim * stride + stride, out_s_dim)
        self.norm = EquiLayerNorm()

    def forward(self, node: EquivariantNode) -> EquivariantNode:
        if self.stride == 1:
            return EquivariantNode(
                x_mv=node.x_mv, x_s=node.x_s, pos_mv=node.pos_mv,
                pos_cartesian=node.pos_cartesian, batch_idx=node.batch_idx, children=node
            )

        with torch.no_grad():
            batch_idx = node.batch_idx[::self.stride]
            centers = reduce(node.pos_cartesian, "(n s) d -> n d", 'mean', s=self.stride)
            pos = rearrange(node.pos_cartesian, "(n s) d -> n s d", s=self.stride)
            rel_pos = pos - centers[:, None]

        x_mv = torch.cat(
            [rearrange(node.x_mv, "(n s) c m -> n (s c) m", s=self.stride),
             embed_translation(rel_pos)], dim = 1)
        if node.x_s is not None:
            x_s = torch.cat([rearrange(node.x_s, "(n s) c -> n (s c)", s=self.stride), 
                            torch.norm(rel_pos, dim = -1)], dim = 1)
        else:
            x_s = None
            
        x_mv, x_s = self.proj(x_mv, x_s)
        x_mv, x_s = self.norm(x_mv, x_s)
        return EquivariantNode(x_mv=x_mv, 
                               x_s=x_s,
                               pos_mv=embed_point(centers),
                               pos_cartesian=centers,
                               batch_idx=batch_idx, 
                               children=node)
    


class EquivariantBallUnpooling(nn.Module):
    """ Equivariant pooling using mean aggregation and EquiLinear projection. """
    def __init__(self, in_mv_dim: int, in_s_dim: Optional[int], out_mv_dim: int, out_s_dim: Optional[int], stride: int, dimensionality: int = 3):
        super().__init__()
        self.stride = stride
        self.proj = EquiLinear(in_mv_dim + stride, out_mv_dim*stride, 
                               in_s_dim + stride if in_s_dim is not None else None, 
                               out_s_dim*stride if in_s_dim is not None else None)
        self.norm = EquiLayerNorm()

    def forward(self, node: EquivariantNode) -> EquivariantNode:
        if self.stride == 1:
            return EquivariantNode(
                x_mv=node.x_mv, x_s=node.x_s, pos_mv=node.pos_mv,
                pos_cartesian=node.pos_cartesian, batch_idx=node.batch_idx, children=node
            )

        with torch.no_grad():
            rel_pos = rearrange(node.children.pos_cartesian, "(n m) d -> n m d", m=self.stride) - node.pos_cartesian[:, None]
        x_mv = torch.cat([node.x_mv, embed_translation(rel_pos)], dim = -2)
        if node.x_s is not None:
            x_s = torch.cat([node.x_s, torch.norm(rel_pos, dim = -1)], dim = 1)
        else:
            x_s = None
        x_mv, x_s = self.proj(x_mv, x_s)
        x_mv, x_s = self.norm(x_mv, x_s)
        node.children.x_mv = node.children.x_mv + rearrange(x_mv, "n (m d) s-> (n m) d s", m = self.stride)
        node.children.x_s = node.children.x_s + rearrange(x_s, "n (m d) -> (n m) d ", m = self.stride) if x_s is not None else None
                
        node.children.x_mv, node.children.x_s = self.norm(node.children.x_mv, node.children.x_s)

        return node.children

class EquivariantErwinBlock(nn.Module):
    """ Equivariant version of the Erwin Transformer Block. """
    def __init__(self, mv_dim: int, s_dim: Optional[int], num_heads: int, ball_size: int, mlp_ratio: int, dropout: float = 0.0, dimensionality: int = 3):
        super().__init__()
        self.mv_dim = mv_dim
        self.s_dim = s_dim
        self.norm1 = EquiLayerNorm()
        self.attn = EquivariantBallAttention(mv_dim, s_dim, num_heads, ball_size, dropout=dropout)

        self.norm2 = EquiLayerNorm()

        hidden_mv_dim = int(mv_dim*2)
        hidden_s_dim = int(s_dim*2) if s_dim is not None else None

        mlp_config = MLPConfig(
            mv_channels=(mv_dim, hidden_mv_dim, mv_dim),
            s_channels=(s_dim, hidden_s_dim, s_dim) if s_dim is not None else None,
            activation='gelu',
            dropout_prob=dropout
        )
        self.mlp = GeoMLP(config=mlp_config)

    def forward(self, node: EquivariantNode, reference_mv: torch.Tensor):
        x_mv, x_s, pos_mv, batch_idx = node.x_mv, node.x_s, node.pos_mv, node.batch_idx
        shortcut_mv, shortcut_s = x_mv, x_s

        x_mv_norm, x_s_norm = self.norm1(x_mv, x_s)
        attn_out_mv, attn_out_s = self.attn(x_mv_norm, x_s_norm, pos_mv, node.pos_cartesian ,batch_idx)

        x_mv = shortcut_mv + attn_out_mv
        if x_s is not None and attn_out_s is not None:
            x_s = shortcut_s + attn_out_s
        shortcut_mv, shortcut_s = x_mv, x_s

        x_mv_norm, x_s_norm = self.norm2(x_mv, x_s)

        mlp_out_mv, mlp_out_s = self.mlp(x_mv_norm, scalars=x_s_norm, reference_mv=reference_mv)

        x_mv = shortcut_mv + mlp_out_mv
        if x_s is not None and mlp_out_s is not None:
            x_s = shortcut_s + mlp_out_s

        node.x_mv = x_mv
        node.x_s = x_s
        return node


class EquivariantBasicLayer(nn.Module):
    """ Equivariant version of Erwin's BasicLayer. """
    def __init__(
        self,
        direction: Literal['down', 'up', None], 
        depth: int,
        stride: int,
        in_mv_dim: int, in_s_dim: Optional[int],
        out_mv_dim: int, out_s_dim: Optional[int], 
        hidden_mv_dim: int, hidden_s_dim: Optional[int], 
        num_heads: int,
        ball_size: int,
        mlp_ratio: int,
        rotate: bool,
        dimensionality: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_mv_dim = in_mv_dim if direction == 'down' else out_mv_dim
        hidden_s_dim = hidden_mv_dim
        self.blocks = nn.ModuleList([
            EquivariantErwinBlock(
                hidden_mv_dim, hidden_s_dim, num_heads, ball_size, mlp_ratio, dropout, dimensionality
            ) for _ in range(depth)
        ])
        self.rotate = [i % 2 for i in range(depth)] if rotate else [False] * depth

        self.pool = lambda node: node
        self.unpool = lambda node: node

        if direction == 'down' and stride > 1:
            self.pool = EquivariantBallPooling(
                hidden_mv_dim, hidden_s_dim, out_mv_dim, out_s_dim, stride, dimensionality
            )
        elif direction == 'up' and stride > 1:
            self.unpool = EquivariantBallUnpooling(
                 in_mv_dim, in_s_dim,
                 out_mv_dim=hidden_mv_dim, out_s_dim=hidden_s_dim,
                 stride=stride, dimensionality=dimensionality
            )

    def forward(self, node: EquivariantNode, reference_mv: torch.Tensor) -> EquivariantNode:
        node = self.unpool(node)

        tree_idx_rot_inv = None
        if len(self.rotate) > 1 and self.rotate[1]:
            assert node.tree_idx_rot is not None, "tree_idx_rot must be provided for rotation"
            tree_idx_rot_inv = torch.argsort(node.tree_idx_rot)

        for use_rotation, blk in zip(self.rotate, self.blocks):
            if use_rotation:
                node_rotated = EquivariantNode(
                    x_mv = node.x_mv[node.tree_idx_rot],
                    x_s = node.x_s[node.tree_idx_rot] if node.x_s is not None else None,
                    pos_mv = node.pos_mv[node.tree_idx_rot], 
                    pos_cartesian = node.pos_cartesian[node.tree_idx_rot] if node.pos_cartesian is not None else None,
                    batch_idx = node.batch_idx[node.tree_idx_rot] if node.batch_idx is not None else None,
                )
                processed_node_rotated = blk(node_rotated, reference_mv)
                node.x_mv = processed_node_rotated.x_mv[tree_idx_rot_inv]
                if node.x_s is not None:
                    node.x_s = processed_node_rotated.x_s[tree_idx_rot_inv]
            else:
                node = blk(node, reference_mv)

        node = self.pool(node)

        return node


class EquivariantErwinTransformer(nn.Module):
    """
    Equivariant Erwin Transformer using GATr primitives.

    Args:
        c_in_scalar (int): number of initial scalar input channels.
        mv_dims (List): list of multivector channel dimensions for each encoder + bottleneck layer.
        s_dims (List[Optional[int]]): list of scalar channel dimensions. Length must match mv_dims. Use None if no scalars.
        ball_sizes (List): list of ball sizes for attention at each encoder layer.
        enc_num_heads (List): list of number of heads for each encoder layer.
        enc_depths (List): list of number of ErwinTransformerBlock layers for each encoder layer.
        dec_num_heads (List): list of number of heads for each decoder layer.
        dec_depths (List): list of number of ErwinTransformerBlock layers for each decoder layer.
        strides (List): list of strides for pooling/unpooling.
        rotate (bool): Whether to use rotation for cross-ball connections.
        decode (bool): whether to decode or not. If not, returns latent representation at the coarsest level.
        mlp_ratio (int): ratio of MLP hidden dim to layer's hidden dim.
        dimensionality (int): dimensionality of the input Cartesian coordinates.
        mp_steps (int): number of message passing steps in the initial Embedding.
        dropout (float): dropout rate.
        out_dim_scalar (Optional[int]): If specified, projects final scalar features to this dimension.
        out_dim_cartesian (bool): If True, extracts Cartesian coordinates from final point multivectors.
    """
    def __init__(
        self,
        mv_dim_in: int,
        mv_dims: List[int],
        s_dims: Optional[List[Optional[int]]],
        ball_sizes: List[int],
        enc_num_heads: List[int],
        enc_depths: List[int],
        dec_num_heads: List[int],
        dec_depths: List[int],
        strides: List[int],
        rotate: bool,
        decode: bool = True,
        mlp_ratio: int = 4,
        dimensionality: int = 3,
        mp_steps: int = 3,
        dropout: float = 0.0,
        out_dim_scalar: Optional[int] = None,
        out_dim_cartesian: bool = False,
    ):
        super().__init__()
        assert len(mv_dims) == len(s_dims), "mv_dims and s_dims must have the same length"
        assert len(enc_num_heads) == len(enc_depths) == len(ball_sizes) == len(mv_dims)
        assert len(dec_num_heads) == len(dec_depths) == len(strides)
        assert len(strides) == len(mv_dims) - 1

        self.rotate = rotate > 0
        self.decode = decode
        self.ball_sizes = ball_sizes
        self.strides = strides
        self.out_dim_scalar = out_dim_scalar
        self.out_dim_cartesian = out_dim_cartesian

        self.embed = EquivariantErwinEmbedding(mv_dim_in, mv_dims[0], 
                                               in_s_dim=mv_dim_in, out_s_dim=s_dims[0], mp_steps=mp_steps)
        #self.embed = RadEquivariantErwinEmbedding(mv_dim_in, mv_dims[0], 
        #                    in_s_dim=mv_dim_in, out_s_dim=s_dims[0], mp_steps=mp_steps)

        num_layers = len(enc_depths) - 1

        self.encoder = nn.ModuleList()
        for i in range(num_layers):
            self.encoder.append(
                EquivariantBasicLayer(
                    direction='down',
                    depth=enc_depths[i],
                    stride=strides[i],
                    in_mv_dim=mv_dims[i], in_s_dim=s_dims[i], # Input from previous layer
                    out_mv_dim=mv_dims[i+1], out_s_dim=s_dims[i+1], # Output dim after pooling projection
                    hidden_mv_dim=mv_dims[i], hidden_s_dim=s_dims[i], # Dim used within blocks
                    num_heads=enc_num_heads[i],
                    ball_size=ball_sizes[i],
                    mlp_ratio=mlp_ratio,
                    rotate=self.rotate,
                    dimensionality=dimensionality,
                    dropout=dropout,
                )
            )

        self.bottleneck = EquivariantBasicLayer(
            direction=None,
            depth=enc_depths[-1],
            stride=1,
            in_mv_dim=mv_dims[-1], in_s_dim=s_dims[-1],
            out_mv_dim=mv_dims[-1], out_s_dim=s_dims[-1],
            hidden_mv_dim=mv_dims[-1], hidden_s_dim=s_dims[-1],
            num_heads=enc_num_heads[-1],
            ball_size=ball_sizes[-1],
            mlp_ratio=mlp_ratio,
            rotate=self.rotate,
            dimensionality=dimensionality,
            dropout=dropout,
        )

        if decode:
            self.decoder = nn.ModuleList()
            for i in range(num_layers - 1, -1, -1):
                self.decoder.append(
                    EquivariantBasicLayer(
                        direction='up',
                        depth=dec_depths[i],
                        stride=strides[i],
                        in_mv_dim=mv_dims[i+1], in_s_dim=s_dims[i+1],
                        out_mv_dim=mv_dims[i], out_s_dim=s_dims[i],
                        hidden_mv_dim=mv_dims[i], hidden_s_dim=s_dims[i],
                        num_heads=dec_num_heads[i],
                        ball_size=ball_sizes[i],
                        mlp_ratio=mlp_ratio,
                        rotate=self.rotate,
                        dimensionality=dimensionality,
                        dropout=dropout,
                    )
                )

        self.in_dim = mv_dim_in
        self.out_dim = mv_dims[0]



    def forward(self, x_mv: torch.Tensor,
                      node_positions_cartesian: torch.Tensor,
                      batch_idx: torch.Tensor,
                      x_s: Optional[torch.Tensor] = None,
                      edge_index: Optional[torch.Tensor] = None, 
                      tree_idx: Optional[torch.Tensor] = None,
                      tree_mask: Optional[torch.Tensor] = None,
                      radius: Optional[float] = None,
                      **kwargs):

        if tree_idx is None or tree_mask is None:

            tree_idx, tree_mask, tree_idx_rot_list = build_balltree_with_rotations(
                node_positions_cartesian, batch_idx, self.strides, self.ball_sizes, self.rotate)
            if tree_idx_rot_list is None: tree_idx_rot_list = []

        if edge_index is None and self.embed.mp_steps > 0:
            if radius is None:
                 raise ValueError("Radius must be provided for MPNN if edge_index is not given.")
            if 'torch_cluster' not in globals():
                 raise RuntimeError("torch_cluster required for radius_graph.")
            edge_index = torch_cluster.radius_graph(node_positions_cartesian, radius, batch=batch_idx, loop=True)


        self.ref_mv_global = construct_reference_multivector('data', x_mv)
        x_mv, x_s= self.embed(x_mv, x_s, node_positions_cartesian, edge_index)
        node = EquivariantNode(
            x_mv=x_mv[tree_idx],
            x_s=x_s[tree_idx] if x_s is not None else None,
            pos_mv=0,
            pos_cartesian=node_positions_cartesian[tree_idx],
            batch_idx=batch_idx[tree_idx],
            tree_idx_rot=None,
        )

        skip_connections = []

        # 4. Encoder Path
        rot_idx_iter = iter(tree_idx_rot_list)
        for i, layer in enumerate(self.encoder):
            node.tree_idx_rot = next(rot_idx_iter, None)
            skip_connections.append(node)
            node = layer(node, self.ref_mv_global)

        node.tree_idx_rot = next(rot_idx_iter, None)
        node = self.bottleneck(node, self.ref_mv_global)

        if self.decode:
            for i, layer in enumerate(self.decoder):
                skip_node = skip_connections.pop()
                node.children = skip_node
                node = layer(node, self.ref_mv_global)

            final_x_mv = node.x_mv[tree_mask]
            final_x_s = node.x_s[tree_mask] if node.x_s is not None else None
            original_indices = torch.argsort(tree_idx[tree_mask])
            final_x_mv = final_x_mv[original_indices]
            if final_x_s is not None:
                 final_x_s = final_x_s[original_indices]


            return final_x_mv, final_x_s


        else:
            return {'bottleneck_mv': node.x_mv, 'bottleneck_s': node.x_s, 'bottleneck_batch_idx': node.batch_idx}

