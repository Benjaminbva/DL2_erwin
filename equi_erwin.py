# @title Equivariant Erwin Transformer Implementation
# @markdown This code integrates GATr's E(3)-equivariant principles
# @markdown into the hierarchical structure of the Erwin transformer.
# @markdown It assumes a `gatr` library providing the necessary
# @markdown primitives like `EquiLinear`, `GeometricAttention`, etc.
# @markdown and interfaces like `embed_point`, `extract_point`.

import torch
import torch.nn as nn
from einops import rearrange, reduce
from typing import Literal, List, Optional
from dataclasses import dataclass

# Assume these imports from GATr library/primitives
from gatr.layers.linear import EquiLinear
from gatr.layers.attention.self_attention import SelfAttention
from gatr.layers.attention.config import SelfAttentionConfig
from gatr.layers.mlp.mlp import GeoMLP
from gatr.layers.mlp.config import MLPConfig
from gatr.layers.layer_norm import EquiLayerNorm
from gatr.interface import embed_point,embed_translation
from gatr.utils.tensors import construct_reference_multivector
import torch_cluster


# Assume balltree library is available
from balltree import build_balltree_with_rotations


# --- Data Structures ---

@dataclass
class EquivariantNode:
    """Dataclass to store hierarchical node information with multivectors."""
    x_mv: torch.Tensor               # Multivector features (..., C_mv, 16)
    pos_mv: torch.Tensor             # Multivector positions (..., 1, 16) - embedded points
    x_s: Optional[torch.Tensor] = None # Optional scalar features (..., C_s)
    pos_cartesian: Optional[torch.Tensor] = None # Original Cartesian coords (..., D), potentially needed for tree building
    batch_idx: Optional[torch.Tensor] = None     # Batch indices (N,)
    tree_idx_rot: Optional[torch.Tensor] = None  # Indices for rotated tree permutation
    children: Optional['EquivariantNode'] = None # Link to finer level node during unpooling

# --- Equivariant Modules ---
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
    """ Invariant Message Passing Neural Network using scalar distances. """
    def __init__(self, scalar_feature_dim: int, mp_steps: int, distance_dim: int = 1):
        super().__init__()
        self.mp_steps = mp_steps
        self.message_fns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * scalar_feature_dim + distance_dim, scalar_feature_dim),
                nn.GELU(),
                nn.LayerNorm(scalar_feature_dim)
            ) for _ in range(mp_steps)
        ])
        self.update_fns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * scalar_feature_dim, scalar_feature_dim), # hidden_dim + message_dim
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

        for message_fn, update_fn in zip(self.message_fns, self.update_fns):
            # Message computation
            message_inputs = torch.cat([h_s[row], h_s[col], edge_attr], dim=-1)
            messages = message_fn(message_inputs)

            # Aggregation
            aggregated_messages = scatter_mean(messages, col, h_s.size(0))

            # Update
            update_inputs = torch.cat([h_s, aggregated_messages], dim=-1)
            h_s_update = update_fn(update_inputs)
            h_s = h_s + h_s_update # Residual connection

        return h_s


class EquivariantErwinEmbedding(nn.Module):
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
        self.mpnn = InvMPNN(scalar_feature_dim=out_s_dim, mp_steps=mp_steps, distance_dim=1)

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

        # Configure GATr SelfAttention
        attn_config = SelfAttentionConfig(
            in_mv_channels=mv_dim,
            out_mv_channels=mv_dim,
            in_s_channels=s_dim,
            out_s_channels=s_dim,
            num_heads=num_heads,
            dropout_prob=dropout,
            pos_encoding=True, # Erwin handles structure via ball partitioning
            increase_hidden_channels=increase_hidden_channels_factor # ADD THIS LINE (or rely on GATr default)
            # Ensure all other necessary SelfAttentionConfig fields are appropriately set or defaulted
        )
        self.attention = SelfAttention(config=attn_config)
        self.sigma_att = nn.Parameter(-1 + 0.01 * torch.randn((1, num_heads, 1, 1)))

    @torch.no_grad()
    def create_attention_mask(self, pos: torch.Tensor):
        """ Distance-based attention bias (eq. 10). """
        pos = rearrange(pos, '(n m) d -> n m d', m=self.ball_size)
        return self.sigma_att * torch.cdist(pos, pos, p=2).unsqueeze(1)
    
    def forward(self, x_mv: torch.Tensor, x_s: Optional[torch.Tensor], pos_mv: torch.Tensor, pos_cartesian: torch.Tensor, batch_idx: torch.Tensor):
        # Reshape inputs for attention: (N, C, 16) -> (B, Items_in_batch, C, 16)
        # GATr attention expects shape (..., items, channels, 16/s_dim)
        # Here, 'items' corresponds to points within a ball.
        # We need to reshape the flat input (N = num_balls * ball_size)
        # into (num_balls, ball_size, C, 16)

        num_total_nodes = x_mv.shape[0]

        num_balls = num_total_nodes // self.ball_size

        x_mv_batched = rearrange(x_mv, '(b n) c val -> b n c val', b=num_balls, n=self.ball_size)
        x_s_batched = None
        if x_s is not None:
            x_s_batched = rearrange(x_s, '(b n) cs -> b n cs', b=num_balls, n=self.ball_size)

        # GATr SelfAttention expects (..., items, channels, 16)
        # Input shape is (batch=num_balls, items=ball_size, channels=c_mv, 16)
        out_mv, out_s = self.attention(x_mv_batched, scalars=x_s_batched,
                                       attention_mask = self.create_attention_mask(pos_cartesian)) # , attention_mask=?)

        # Reshape back to flat structure: (B, N, C, 16) -> (B*N, C, 16)
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

    def forward(self, node: EquivariantNode) -> EquivariantNode:
        if self.stride == 1: # No pooling
            # Still wrap in EquivariantNode for consistency
            return EquivariantNode(
                x_mv=node.x_mv, x_s=node.x_s, pos_mv=node.pos_mv,
                pos_cartesian=node.pos_cartesian, batch_idx=node.batch_idx, children=node
            )

        # Aggregate features using mean
        # Input shape: (N_fine, C, 16/s) where N_fine = N_coarse * stride
        # Output shape: (N_coarse, C, 16/s)
        with torch.no_grad():
            batch_idx = node.batch_idx[::self.stride]
            centers = reduce(node.pos_cartesian, "(n s) d -> n d", 'mean', s=self.stride)
            pos = rearrange(node.pos_cartesian, "(n s) d -> n s d", s=self.stride)
            rel_pos = pos - centers[:, None]

        x_mv = torch.cat(
            [rearrange(node.x_mv, "(n s) c m -> n (s c) m", s=self.stride),
             embed_translation(rel_pos)], dim = 1)
        x_s = torch.cat([rearrange(node.x_s, "(n s) c -> n (s c)", s=self.stride), 
                         torch.norm(rel_pos, dim = -1)], dim = 1)
        x_mv, x_s = self.proj(x_mv, x_s)
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
                               in_s_dim + stride, out_s_dim*stride)

    def forward(self, node: EquivariantNode) -> EquivariantNode:
        if self.stride == 1: # No pooling
            # Still wrap in EquivariantNode for consistency
            return EquivariantNode(
                x_mv=node.x_mv, x_s=node.x_s, pos_mv=node.pos_mv,
                pos_cartesian=node.pos_cartesian, batch_idx=node.batch_idx, children=node
            )

        # Aggregate features using mean
        # Input shape: (N_fine, C, 16/s) where N_fine = N_coarse * stride
        # Output shape: (N_coarse, C, 16/s)
        with torch.no_grad():
            rel_pos = rearrange(node.children.pos_cartesian, "(n m) d -> n m d", m=self.stride) - node.pos_cartesian[:, None]
        x_mv = torch.cat([node.x_mv, embed_translation(rel_pos)], dim = -2)
        x_s = torch.cat([node.x_s, torch.norm(rel_pos, dim = -1)], dim = 1)
        x_mv, x_s = self.proj(x_mv, x_s)
        node.children.x_mv = node.children.x_mv + rearrange(x_mv, "n (m d) s-> (n m) d s", m = self.stride)
        node.children.x_s = node.children.x_s + rearrange(x_s, "n (m d) -> (n m) d ", m = self.stride)
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

        hidden_mv_dim = int(mv_dim * mlp_ratio) # Or maybe fixed hidden dim? GATr MLPConfig allows tuple
        hidden_s_dim = int(s_dim * mlp_ratio) if s_dim is not None else None

        mlp_config = MLPConfig(
            mv_channels=(mv_dim, hidden_mv_dim, mv_dim),
            s_channels=(s_dim, hidden_s_dim, s_dim) if s_dim is not None else None,
            activation='gelu', # GATr uses scalar-gated activations
            dropout_prob=dropout
        )
        self.mlp = GeoMLP(config=mlp_config)

        # DropPath not directly applicable to multivectors in the same way.
        # GATr uses GradeDropout within layers if needed.
        # self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()

    def forward(self, node: EquivariantNode, reference_mv: torch.Tensor):
        x_mv, x_s, pos_mv, batch_idx = node.x_mv, node.x_s, node.pos_mv, node.batch_idx
        shortcut_mv, shortcut_s = x_mv, x_s

        # --- Attention Part ---
        x_mv_norm, x_s_norm = self.norm1(x_mv, x_s)
        attn_out_mv, attn_out_s = self.attn(x_mv_norm, x_s_norm, pos_mv, node.pos_cartesian ,batch_idx)

        # Apply dropout/drop path if applicable (GATr handles dropout internally in layers)
        # attn_out_mv, attn_out_s = self.drop_path(attn_out_mv, attn_out_s) # Needs adaptation for tuple

        x_mv = shortcut_mv + attn_out_mv
        if x_s is not None and attn_out_s is not None:
            x_s = shortcut_s + attn_out_s
        shortcut_mv, shortcut_s = x_mv, x_s # Update shortcut for next stage

        # --- MLP Part ---
        x_mv_norm, x_s_norm = self.norm2(x_mv, x_s)

        mlp_out_mv, mlp_out_s = self.mlp(x_mv_norm, scalars=x_s_norm, reference_mv=reference_mv)

        # Apply dropout/drop path
        # mlp_out_mv, mlp_out_s = self.drop_path(mlp_out_mv, mlp_out_s)

        x_mv = shortcut_mv + mlp_out_mv
        if x_s is not None and mlp_out_s is not None:
            x_s = shortcut_s + mlp_out_s

        # Update node features
        node.x_mv = x_mv
        node.x_s = x_s
        return node


class EquivariantBasicLayer(nn.Module):
    """ Equivariant version of Erwin's BasicLayer. """
    def __init__(
        self,
        direction: Literal['down', 'up', None], # down: encoder, up: decoder, None: bottleneck
        depth: int,
        stride: int,
        in_mv_dim: int, in_s_dim: Optional[int],
        out_mv_dim: int, out_s_dim: Optional[int], # Only relevant for pooling/unpooling projection
        hidden_mv_dim: int, hidden_s_dim: Optional[int], # Dim used within blocks
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
            # Unpooling needs skip connection dims, assume they match hidden dims for now
            # Input to unpooling proj is `in_mv_dim/in_s_dim` (from coarser level)
            # Output should match `hidden_mv_dim/hidden_s_dim` (finer level block dim)
            self.unpool = EquivariantBallUnpooling(
                 in_mv_dim, in_s_dim, # Input from previous (coarser) layer
                 out_mv_dim=hidden_mv_dim, out_s_dim=hidden_s_dim, # Output dim of projection
                 stride=stride, dimensionality=dimensionality
            )

    def forward(self, node: EquivariantNode, reference_mv: torch.Tensor) -> EquivariantNode:
        # 1. Unpooling (for decoder)
        node = self.unpool(node)

        # Handle rotation permutation logic from original Erwin
        tree_idx_rot_inv = None
        if len(self.rotate) > 1 and self.rotate[1]:
            assert node.tree_idx_rot is not None, "tree_idx_rot must be provided for rotation"
            tree_idx_rot_inv = torch.argsort(node.tree_idx_rot) # map from rotated to original

        # 2. Apply Blocks
        for use_rotation, blk in zip(self.rotate, self.blocks):
            if use_rotation:
                # Permute features according to rotated tree, process, permute back
                node_rotated = EquivariantNode(
                    x_mv = node.x_mv[node.tree_idx_rot],
                    x_s = node.x_s[node.tree_idx_rot] if node.x_s is not None else None,
                    #pos_mv = node.pos_mv[node.tree_idx_rot],
                    pos_mv = node.pos_mv[node.tree_idx_rot], 
                    pos_cartesian = node.pos_cartesian[node.tree_idx_rot] if node.pos_cartesian is not None else None,
                    batch_idx = node.batch_idx[node.tree_idx_rot] if node.batch_idx is not None else None,
                    # Pass other attributes if needed, children link might be invalid in rotated view
                )
                processed_node_rotated = blk(node_rotated, reference_mv)
                # Permute back and update original node
                node.x_mv = processed_node_rotated.x_mv[tree_idx_rot_inv]
                if node.x_s is not None:
                    node.x_s = processed_node_rotated.x_s[tree_idx_rot_inv]
                # Keep original node's pos, batch_idx etc.
            else:
                # Process with original ordering
                node = blk(node, reference_mv)

        # 3. Pooling (for encoder)
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

        self.rotate = rotate > 0 # Original Erwin used rotate=angle, here just bool
        self.decode = decode
        self.ball_sizes = ball_sizes
        self.strides = strides
        self.out_dim_scalar = out_dim_scalar
        self.out_dim_cartesian = out_dim_cartesian

        # Initial Embedding
        self.embed = EquivariantErwinEmbedding(mv_dim_in, mv_dims[0], 
                                               in_s_dim=mv_dim_in, out_s_dim=s_dims[0], mp_steps=mp_steps)

        num_layers = len(enc_depths) - 1 # last one is a bottleneck

        # Encoder
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

        # Bottleneck
        self.bottleneck = EquivariantBasicLayer(
            direction=None,
            depth=enc_depths[-1],
            stride=1, # No pooling/unpooling
            in_mv_dim=mv_dims[-1], in_s_dim=s_dims[-1],
            out_mv_dim=mv_dims[-1], out_s_dim=s_dims[-1], # Not used
            hidden_mv_dim=mv_dims[-1], hidden_s_dim=s_dims[-1],
            num_heads=enc_num_heads[-1],
            ball_size=ball_sizes[-1],
            mlp_ratio=mlp_ratio,
            rotate=self.rotate,
            dimensionality=dimensionality,
            dropout=dropout,
        )

        # Decoder
        if decode:
            self.decoder = nn.ModuleList()
            for i in range(num_layers - 1, -1, -1):
                self.decoder.append(
                    EquivariantBasicLayer(
                        direction='up',
                        depth=dec_depths[i],
                        stride=strides[i],
                        in_mv_dim=mv_dims[i+1], in_s_dim=s_dims[i+1], # Input from previous coarser layer
                        out_mv_dim=mv_dims[i], out_s_dim=s_dims[i], # Not directly used by unpool proj output
                        hidden_mv_dim=mv_dims[i], hidden_s_dim=s_dims[i], # Dim used within blocks (and target for unpool proj)
                        num_heads=dec_num_heads[i],
                        ball_size=ball_sizes[i], # Ball size at this finer level
                        mlp_ratio=mlp_ratio,
                        rotate=self.rotate,
                        dimensionality=dimensionality,
                        dropout=dropout,
                    )
                )

        self.in_dim = mv_dim_in
        self.out_dim = mv_dims[0]



    def forward(self, x_mv: torch.Tensor, # Initial features assumed scalar
                      node_positions_cartesian: torch.Tensor,
                      batch_idx: torch.Tensor,
                      x_s: Optional[torch.Tensor] = None,
                      edge_index: Optional[torch.Tensor] = None, # For initial MPNN
                      tree_idx: Optional[torch.Tensor] = None,   # Precomputed tree indices
                      tree_mask: Optional[torch.Tensor] = None, # Precomputed tree mask
                      radius: Optional[float] = None, # For building radius graph if edge_index not given
                      **kwargs):
        # 1. Build Ball Tree (using Cartesian coordinates as in original Erwin)
        # This part remains non-equivariant by design for Erwin's cross-ball mechanism
        if tree_idx is None or tree_mask is None:
            #print('making tree')
            tree_idx, tree_mask, tree_idx_rot_list = build_balltree_with_rotations(
                node_positions_cartesian, batch_idx, self.strides, self.ball_sizes, self.rotate)
            if tree_idx_rot_list is None: tree_idx_rot_list = [] # Ensure it's a list

        # Build radius graph if needed for initial MPNN
        if edge_index is None and self.embed.mp_steps > 0:
            if radius is None:
                 raise ValueError("Radius must be provided for MPNN if edge_index is not given.")
            if 'torch_cluster' not in globals():
                 raise RuntimeError("torch_cluster required for radius_graph.")
            edge_index = torch_cluster.radius_graph(node_positions_cartesian, radius, batch=batch_idx, loop=True)


        # 2. Initial Embedding (Scalar features -> MV/S features, Cartesian pos -> MV pos)
        # Construct a reference multivector (e.g., from mean position) for GeoMLP
        # Using 'data' might be slow, 'canonical' is faster but less adaptive.
        self.ref_mv_global = construct_reference_multivector('data', x_mv) # Or use positions?

        x_mv, x_s= self.embed(x_mv, x_s, node_positions_cartesian, edge_index)

        # 3. Prepare Initial Node for Hierarchy (using tree permutation)
        node = EquivariantNode(
            x_mv=x_mv[tree_idx],
            x_s=x_s[tree_idx] if x_s is not None else None,
            pos_mv=0,
            pos_cartesian=node_positions_cartesian[tree_idx], # Keep for potential CoM in pooling
            batch_idx=batch_idx[tree_idx],
            tree_idx_rot=None, # Will be populated in the encoder layers
        )

        # Store skip connections for decoder
        skip_connections = []

        # 4. Encoder Path
        rot_idx_iter = iter(tree_idx_rot_list)
        for i, layer in enumerate(self.encoder):
            node.tree_idx_rot = next(rot_idx_iter, None) # Get rotation indices for this level
            skip_connections.append(node) # Store node *before* processing and pooling
            node = layer(node, self.ref_mv_global)

        # 5. Bottleneck
        node.tree_idx_rot = next(rot_idx_iter, None)
        node = self.bottleneck(node, self.ref_mv_global)

        # 6. Decoder Path
        if self.decode:
            for i, layer in enumerate(self.decoder):
                # Pass skip connection state to the unpooling layer within BasicLayer
                skip_node = skip_connections.pop()
                node.children = skip_node # Link for unpooling to access skip features
                node = layer(node, self.ref_mv_global)

            # Final features are in `node.x_mv`, `node.x_s`
            # Need to un-permute using the tree_mask and tree_idx
            final_x_mv = node.x_mv[tree_mask]
            final_x_s = node.x_s[tree_mask] if node.x_s is not None else None
            original_indices = torch.argsort(tree_idx[tree_mask])
            final_x_mv = final_x_mv[original_indices]
            if final_x_s is not None:
                 final_x_s = final_x_s[original_indices]

            # 7. Final Output Processing
            output = {}

            output['final_mv'] = final_x_mv


            if final_x_s is not None:
                 if self.out_dim_scalar is not None:
                     output['final_s'] = self.final_proj_s(final_x_s)
                 else:
                     output['final_s'] = final_x_s
            return output

        else:
            # Return bottleneck features if not decoding
            return {'bottleneck_mv': node.x_mv, 'bottleneck_s': node.x_s, 'bottleneck_batch_idx': node.batch_idx}

# --- Helper Functions (If GATr not installed) ---
# Add dummy implementations or necessary primitives if needed
# e.g., torch_scatter if not available
