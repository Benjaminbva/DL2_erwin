import sys
import os
sys.path.append("../../")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import check_invariance, get_cosmology_loader, get_error_results
from models.erwin import ErwinTransformer
from experiments.wrappers.cosmology import CosmologyModel
import json
import sys
import os

if __name__ == "__main__":

    erwin_keys = ["c_in", "c_hidden", "enc_num_heads", "enc_depths", "dec_num_heads", 
                "dec_depths", "strides", "ball_sizes", "rotate", "mp_steps", "eq_9", "eq_12", "eq_13"]

    data_path = './data/cosmology'
    batch_size = 2
    split = 'test'
    num_samples = 512

    test_loader = get_cosmology_loader(data_path, batch_size, split, num_samples)

    experiments = ['MPNN_RotatingTree', 'InvMPNN_RotatingTree', 'inv_eq9', 
                'inv_eq9_eq12', 'inv_eq9_eq12_eq13']

    seeds = [0,1,2]
    angles = [15.0, 45.0, 90.0, 160.0]
    rots = ['x', 'y', 'z']
    model_name = 'erwin'
    size = 'smallish'
    ckpt_dir = 'checkpoints'

    inv_res = get_error_results('inv_res', experiments, seeds, angles, rots, model_name, size, ckpt_dir, check_invariance, erwin_keys, ErwinTransformer, CosmologyModel, test_loader)

    with open("inv_res.json", "w", encoding="utf-8") as f:
        json.dump(inv_res, f, ensure_ascii=False, indent=2)