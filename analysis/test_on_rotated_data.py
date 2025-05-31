import sys
import os
sys.path.append("../../")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.GATrErwin import EquivariantErwinTransformer
from experiments.wrappers.cosmology_equi import CosmologyEquiModel
import torch
import torch
import json
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from utils import check_rotated_data, get_cosmology_loader, get_error_results


if __name__ == "__main__":

    data_path = './data/cosmology'
    batch_size = 2
    split = 'test'
    num_samples = 512
    test_loader = get_cosmology_loader(data_path, batch_size, split, num_samples)

    GATrErwin_keys = ["mv_dim_in", "mv_dims", "s_dims", "enc_num_heads", "enc_depths", "dec_num_heads", "dec_depths", 
                    "strides", "ball_sizes", "rotate", "mp_steps", "rbf_dim", "max_l", "cutoff", "use_rad"]

    experiments = ['equi_smaller_rad', 'equi_smaller_rad_90r']
    seeds = [0,1,2]
    angles = [15.0, 45.0, 90.0, 160.0]
    rots = ['x', 'y', 'z']
    model_name = 'erwin'
    size = 'smaller'
    ckpt_dir = 'checkpoints'

    equi_res = get_error_results('rot_mse', experiments, seeds, angles, rots, model_name, size, ckpt_dir, check_rotated_data, GATrErwin_keys, EquivariantErwinTransformer, CosmologyEquiModel, test_loader)

    with open("rot_mse.json", "w", encoding="utf-8") as f:
        json.dump(equi_res, f, ensure_ascii=False, indent=2)