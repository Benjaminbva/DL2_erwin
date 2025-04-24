import torch
from models import ErwinTransformer
from torch import sin, cos
import math
from balltree import build_balltree
import pandas as pd

torch.set_float32_matmul_precision("high")
df = pd.read_csv('erwin_stats.csv')

def invariance_error(model, x, pos, R, **kwargs):
    out = model(x, pos, **kwargs)
    out_r = model(x, pos@R, **kwargs)
    err = out - out_r
    return torch.norm(err)

model_name = 'eq. 13'
err_list = []
err_fixed_list = []
for i in range(100):
    config_w_pool = {
        "c_in": 16,
        "c_hidden": 16,
        "ball_sizes": [128, 128],
        "enc_num_heads": [1,1],
        "enc_depths": [1,1],
        "dec_num_heads": [4,],
        "dec_depths": [4,],
        "strides": [4,], # 0.25 coarsening
        "mp_steps": 0, # no MPNN
        "decode": True, # no decoder
        "dimensionality": 2, # for visualization
        "rotate": 0,
    }

    model = ErwinTransformer(**config_w_pool).cuda()

    bs = 1
    num_points = 1024
    angle = 45
    
    x = torch.randn(num_points * bs, config_w_pool["c_in"]).cuda()
    pos = torch.rand(num_points * bs, config_w_pool["dimensionality"]).cuda()
    batch_idx = torch.repeat_interleave(torch.arange(bs), num_points).cuda()
    angle = torch.tensor(math.radians(angle))
    R = torch.tensor([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]]).cuda()

    #dynamic
    kwargs = {
        "batch_idx": batch_idx,
        "radius": None,
        'rotate' : 0
    }
    err = invariance_error(model, x, pos, R, **kwargs)
    err_list.append(err)

    #fixed
    tree_idx, tree_mask = build_balltree(pos, batch_idx)

    kwargs = {
        "batch_idx": batch_idx,
        "radius": None,
        'tree_idx' : tree_idx,
        'tree_mask' : tree_mask,
        'rotate' : 0
    }
    err_fixed = invariance_error(model, x, pos, R, **kwargs)
    err_fixed_list.append(err_fixed)


errs = torch.tensor(err_list)
errs_fixed = torch.tensor(err_fixed_list)

df.loc[len(df)] = [model_name, errs.mean(), errs.std()]
df.loc[len(df)] = [model_name + '_fixed', errs_fixed.mean(), errs_fixed.std()]

df.to_csv('erwin_stats.csv', index=False)


'''tree_idx, tree_mask = build_balltree(pos, batch_idx)

kwargs = {
    "batch_idx": batch_idx,
    "radius": None,
    'tree_idx' : tree_idx,
    'tree_mask' : tree_mask,
}
err_fixed = invariance_error(model, x, pos, R, **kwargs)
print(err_fixed)'''