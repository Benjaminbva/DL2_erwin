
import torch 
import os
from gatr_utils.interface.point import extract_point
import torch.nn.functional as F
import math
from experiments.datasets.cosmology import CosmologyDataset
from torch.utils.data import DataLoader
import json

def load_model(model_architecture, wrapper, model_name, experiment, size, seed, ckpt_dir, model_keys):
    name = '_'.join([model_name, experiment, size, str(seed), 'best.pt'])
    ckpt_path = os.path.join(ckpt_dir, name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]

    model_config = {}
    for key in model_keys:
        model_config[key] = cfg[key]
    net = model_architecture(**model_config)
    model = wrapper(net).to(device)
    model = torch.compile(model)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    return model

def check_equivariance(model, test_loader, R):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R = R.to(device)

    numels = 0
    equi_loss = 0
    with torch.inference_mode():
        for data in test_loader:
            pos = data['pos'].to(device)

            batch = {'batch_idx': data['batch_idx'].to(device),
                    'edge_index': data['edge_index'].to(device)}
            
            batch_r = {'batch_idx': data['batch_idx'].to(device),
                    'edge_index': data['edge_index'].to(device)}

            pred, pred_s = model.forward(pos, **batch)
            r_pred, r_pred_s = model.forward(pos@R.T, **batch_r)

            pred = extract_point(pred).squeeze(1)
            r_pred = extract_point(r_pred).squeeze(1)

            loss_tensor = F.mse_loss(pred@R.T, r_pred, reduction="sum")
            equi_loss += loss_tensor
            numels += pred.numel()

            torch.cuda.empty_cache()
            del pred
            del r_pred
        return(equi_loss/numels)

def check_invariance(model, test_loader, R):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R = R.to(device)

    norms = 0
    numels = 0
    with torch.inference_mode():
        for data in test_loader:
            pos = data['pos'].to(device)
            
            batch = {'batch_idx': data['batch_idx'].to(device),
                    'edge_index': data['edge_index'].to(device),

                    }
            
            batch_r = {'batch_idx': data['batch_idx'].to(device),
                    'edge_index': data['edge_index'].to(device),

                    }

            pred = model.forward(pos, **batch)
            r_pred = model.forward(pos@R.T, **batch_r)

            diffs = pred - r_pred
            norm = torch.norm(diffs, dim=1)
            norms += norm.sum()
            numels += norm.numel()
            torch.cuda.empty_cache()
            del pred
            del r_pred
        return(norms/numels)

def get_rotations(angle_deg):
    θ = math.radians(angle_deg)

    cos_t = math.cos(θ)
    sin_t = math.sin(θ)

    # Rotation about Z (i.e. in the XY‐plane)
    R_z = torch.tensor([
        [ cos_t, -sin_t, 0.0],
        [ sin_t,  cos_t, 0.0],
        [   0.0,    0.0, 1.0],
    ], dtype=torch.float32)

    # Rotation about X (i.e. in the YZ‐plane)
    R_x = torch.tensor([
        [1.0,    0.0,     0.0],
        [0.0,  cos_t, -sin_t],
        [0.0,  sin_t,  cos_t],
    ], dtype=torch.float32)

    # Rotation about Y (i.e. in the ZX‐plane)
    R_y = torch.tensor([
        [ cos_t, 0.0, sin_t],
        [   0.0, 1.0,   0.0],
        [-sin_t, 0.0, cos_t],
    ], dtype=torch.float32)
    return [R_x, R_y, R_z]

def get_cosmology_loader(data_path, batch_size, split, num_samples):
    dataset = CosmologyDataset(
        task='node', 
        split=split, 
        num_samples=num_samples, 
        tfrecords_path=data_path, 
        knn=10,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=16,
    )
    return loader

def get_results(name, experiments, seeds, angles, rots, model_name, size, ckpt_dir, check_fn, model_keys, model_architecture, model_wrap, test_loader):
    equi_res = {}
    for experiment in experiments:
        equi_res[experiment] = {}
        for seed in seeds:
            equi_res[experiment][seed] = {}
            for angle in angles:
                equi_res[experiment][seed][angle] = {}
                for rot in rots:
                    equi_res[experiment][seed][angle][rot] = {}
                    
    for experiment in experiments:
        for seed in seeds:
            torch.manual_seed(seed)
            net = load_model(model_architecture, model_wrap, model_name, experiment, size, seed, ckpt_dir, model_keys)
            net.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            net.to(device)

            for angle_deg in angles:
                rotation_matrices = get_rotations(angle_deg)
                for R, rot in zip(rotation_matrices, rots):
                    res = check_fn(net, test_loader, R)
                    print(experiment, seed, angle_deg, rot, res.item())
                    equi_res[experiment][seed][angle_deg][rot] = res.item()

    with open(f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(equi_res, f, ensure_ascii=False, indent=2)

def round_to_kth_significant(x, k):
    if x == 0:
        return 0
    exp = math.floor(math.log10(abs(x)))
    digits = -exp + (k - 1)
    return round(x, digits)