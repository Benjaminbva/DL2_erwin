import torch
import torch.nn as nn

import math
from torch_scatter import scatter_mean

class BesselBasisFunctions(nn.Module):
    """Bessel-style radial basis functions."""
    def __init__(self, num_rbf: int, cutoff: float, trainable: bool = False):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff

        freqs = torch.arange(1, num_rbf + 1) * math.pi / cutoff
        self.register_buffer("freqs", freqs)

    def cutoff_fn(self, distances: torch.Tensor) -> torch.Tensor:
        """C¹-continuous cosine cutoff function."""
        mask = (distances < self.cutoff).float()
        return 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1) * mask

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Apply Bessel basis functions to distances."""
        distances = distances.unsqueeze(-1)  

        f_cut = self.cutoff_fn(distances)

        safe_dist = distances + 1e-8
        bessel = torch.sin(self.freqs * distances) / safe_dist

        return bessel * f_cut

def cosine_cutoff(distances: torch.Tensor, cutoff: float) -> torch.Tensor:
    """C¹-continuous cosine cutoff function."""
    mask = (distances < cutoff).float()
    return 0.5 * (torch.cos(distances * math.pi / cutoff) + 1) * mask

class SphericalHarmonics(nn.Module):
    """Proper real spherical harmonics for angular basis functions."""
    def __init__(self, max_l: int):
        super().__init__()
        self.max_l = max_l
        self.num_spherical = (max_l + 1)**2  

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        """Compute real spherical harmonics for given vectors."""
        norm = torch.norm(vectors, dim=-1, keepdim=True)
        directions = vectors / (norm + 1e-10)

        x, y, z = directions[..., 0], directions[..., 1], directions[..., 2]

        result = torch.empty((*directions.shape[:-1], self.num_spherical),
                              dtype=directions.dtype, device=directions.device)

        result[..., 0] = 0.2820947917738781 

        if self.max_l >= 1:
            result[..., 1] = 0.4886025119029199 * y  
            result[..., 2] = 0.4886025119029199 * z  
            result[..., 3] = 0.4886025119029199 * x 

        if self.max_l >= 2:
            result[..., 4] = 0.5462742152960396 * x * y              
            result[..., 5] = 0.5462742152960396 * y * z              
            result[..., 6] = 0.6307831305050401 * (3*z*z - 1) / 2.0  
            result[..., 7] = 0.5462742152960396 * x * z              
            result[..., 8] = 0.5462742152960396 * (x*x - y*y) / 2.0  

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

        self.rbf = BesselBasisFunctions(rbf_dim, cutoff)
        self.sph = SphericalHarmonics(max_l)

        angular_dim = (max_l + 1)**2
        if max_l == 0:
            edge_feat_dim = rbf_dim + angular_dim 
        else:
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

        rbf_feats = self.rbf(dist)

        sph_feats = self.sph(rel_vec)
        cutoff_mask = (dist < self.cutoff).float().unsqueeze(-1)
        sph_feats = sph_feats * cutoff_mask

        if self.max_l == 0:
            edge_feats = torch.cat([rbf_feats, sph_feats], dim=-1)
        else:
            l0_feats = rbf_feats * sph_feats[..., 0:1]

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
            message_inputs = torch.cat([h_s[row], h_s[col], edge_attr], dim=-1)
            messages = message_fn(message_inputs)

            aggregated_messages = scatter_mean(messages, col, h_s.size(0))

            update_inputs = torch.cat([h_s, aggregated_messages], dim=-1)
            h_s_update = update_fn(update_inputs)
            h_s = h_s + h_s_update  # Residual connection

        return h_s