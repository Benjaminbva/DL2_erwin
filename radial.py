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