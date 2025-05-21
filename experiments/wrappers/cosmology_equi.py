import torch
import torch.nn as nn
from gatr.layers.linear import EquiLinear
from gatr.interface.point import embed_point, extract_point
from gatr.layers.mlp import GeoMLP, MLPConfig

class EquiEmbedding(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.pos_embedding = EquiLinear(1, out_dim, 1, out_dim)

    def forward(self, pos):
        return self.pos_embedding(embed_point(pos).unsqueeze(1))


class CosmologyEquiModel(nn.Module):
    def __init__(self, main_model):
        super().__init__()
        self.main_model = main_model
        self.embedding_model = EquiEmbedding(main_model.in_dim)

        mlp_config = MLPConfig(
            mv_channels=(main_model.out_dim, main_model.out_dim, 1),
            s_channels=(main_model.out_dim, main_model.out_dim, 1),
            activation='gelu',
            dropout_prob=0
        )

        self.pred_head = GeoMLP(config=mlp_config)

    def forward(self, node_positions, **kwargs):
        x_mv, x_s = self.embedding_model(node_positions)
        x_mv2, x_s2 = self.main_model(x_mv, node_positions, x_s = x_s,**kwargs)
        return self.pred_head(x_mv2, 
                              scalars = x_s2, 
                              reference_mv=self.main_model.ref_mv_global)


    def step(self, batch, prefix="train"):
        pred_mv, pred_s = self(batch["pos"], **batch)
        loss = ((extract_point(pred_mv).squeeze() - batch["target"]) ** 2).mean()
        return {f"{prefix}/loss": loss}

    def training_step(self, batch):
        return self.step(batch, "train")

    @torch.no_grad()
    def validation_step(self, batch):
        return self.step(batch, "val")
