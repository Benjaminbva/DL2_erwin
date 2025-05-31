
import sys
import os
sys.path.append("../../")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch 
import os
from gatr_utils.interface.point import extract_point
import torch.nn.functional as F
import math
from experiments.datasets.cosmology import CosmologyDataset
from torch.utils.data import DataLoader
import json
import numpy as np
import pandas as pd



def load_model(model_architecture, wrapper, model_name, experiment, size, seed, ckpt_dir, model_keys):
    """Load a compiled model from disk.
     
    Args:
        model_architecture: Function or class that instantiates the base network.
        wrapper: Callable that wraps the network (e.g. adds heads or normalisation).
        model_name: Base name of the model family (used to build the file name).
        experiment: Identifier of the training experiment/run.
        size: Model size descriptor (smallest, smallish, smaller, etc.).
        seed: Random seed corresponding to the checkpoint.
        ckpt_dir: Directory that contains all checkpoints.
        model_keys: Keys to pull from the stored config and forward into the
            "model_architecture" constructor (e.g. hidden size, depth).

    Returns:
        torch.nn.Module: The model loaded with weights and compiled for inference
        on the best available device (CUDA if available, else CPU).
    """
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

def check_equivariance(model, loader, R):
    """Measure rotation *equivariance* of the model's point predictions.

    For each mini-batch, predictions are computed on the original point cloud as
    well as on a rotated version. The Mean-Squared Error (MSE) between the
    rotated original prediction and the prediction of the rotated input is
    accumulated and normalised over all elements.

    Args:
        model: Model to be evaluated (must output per-point predictions).
        loader: "DataLoader" yielding dictionaries with at least "pos",
            "batch_idx" and "edge_index" keys.
        R: 3x3 rotation matrix applied to the input.

    Returns:
        torch.Tensor: Scalar tensor containing the average MSE equivariance error
        over the whole dataset
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R = R.to(device)

    numels = 0
    equi_loss = 0
    with torch.inference_mode():
        for data in loader:
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
    
def check_rotated_data(model, loader, R):
    """Evaluate prediction error when *both* input and target are rotated.

    compare the model's prediction on a rotated input cloud with the rotated 
    ground-truth target (accuracy under rotation).

    Args:
        model: Model under test.
        loader: "DataLoader" that yields point clouds and their targets (key
            "target") along with "batch_idx" and "edge_index".
        R: Rotation matrix 3x3 to apply.

    Returns:
        torch.Tensor: Average MSE between predicted and ground-truth rotated
        targets (lower values indicate better rotation accuracy).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R = R.to(device)

    numels = 0
    equi_loss = 0
    with torch.inference_mode():
        for data in loader:
            pos = data['pos'].to(device)

            batch = {'batch_idx': data['batch_idx'].to(device),
                    'edge_index': data['edge_index'].to(device)}
            
            batch_r = {'batch_idx': data['batch_idx'].to(device),
                    'edge_index': data['edge_index'].to(device)}

            r_pred, r_pred_s = model.forward(pos@R.T, **batch_r)

            r_pred = extract_point(r_pred).squeeze(1)

            loss_tensor = F.mse_loss(r_pred, data['target']@R.T, reduction="sum")
            equi_loss += loss_tensor
            numels += r_pred.numel()

            torch.cuda.empty_cache()
            del r_pred
        return(equi_loss/numels)

def check_invariance(model, loader, R):
    """Measure rotation invariance.

    Contrary to :func: check_equivariance , this function simply computes the
    *difference* between the model's latent outputs for the original and rotated
    point clouds. No ground-truth target is required.

    Args:
        model: Model under test.
        loader:  DataLoader  yielding point-cloud batches.
        R: Rotation matrix applied to inputs.

    Returns:
        torch.Tensor: Average *L2* norm of the difference between the two output
        sets (ideal rotation-invariant models yield 0).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R = R.to(device)

    norms = 0
    numels = 0
    with torch.inference_mode():
        for data in loader:
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
    """Return the three canonical 3-D rotation matrices for a given angle.

    The angle is interpreted as a right-handed rotation about the respective
    axis (X, Y, Z).  A negative angle therefore yields the opposite direction.

    Args:
        angle_deg: Rotation angle in degrees.

    Returns:
        list[torch.Tensor]: Three 3x3 rotation matrices  [R_x, R_y, R_z]  in
        *float32*.
    """
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
    """Create a  DataLoader  for the **CosmologyDataset**.

    Args:
        data_path: Path to the TFRecords directory.
        batch_size: Number of graphs per mini-batch.
        split: Dataset split to load ( 'train' ,  'val'  or  'test' ).
        num_samples: How many TFRecord samples to read.

    Returns:
        torch.utils.data.DataLoader: Ready-to-use data loader with fixed
         knn=10  and 16 worker processes.
    """
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

def get_error_results(name, experiments, seeds, angles, rots, model_name, size, ckpt_dir, check_fn, model_keys, model_architecture, model_wrap, test_loader):
    """Run the chosen *check function* for every experiment/seed/rotation combo.

    Args:
        name: Arbitrary label (unused by the function but left for compatibility).
        experiments: Collection of experiment identifiers.
        seeds: Seeds for which individual checkpoints exist.
        angles: Rotation angles *in degrees* that are passed to
            :func: get_rotations .
        rots: Names of the three axes ('x', 'y', 'z') — only used as dict keys.
        model_name: Base model name component.
        size: Size label of the trained model.
        ckpt_dir: Directory with the checkpoint files.
        check_fn: One of the check helpers (:func: check_equivariance , …).
        model_keys: List of config keys forwarded into the model constructor.
        model_architecture: Callable that builds the network given  model_keys .
        model_wrap: Callable that wraps the bare network into the full model.
        test_loader:  DataLoader  providing the evaluation data.

    Returns:
        dict: Nested dictionary  results[experiment][seed][angle][axis] → float 
        containing the error for every trial.
    """
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
    return equi_res

def round_to_kth_significant(x, k):
    if x == 0:
        return 0
    exp = math.floor(math.log10(abs(x)))
    digits = -exp + (k - 1)
    return round(x, digits)


def get_mse_results(results):
    """Aggregate tab-separated *experiment → MSE* log into mean ± std dataframe.

    Args:
        results: Multi-line string, each line formatted as  "<exp>\t<value>" .

    Returns:
        pandas.DataFrame: DataFrame with columns  ['experiment', 'mean', 'std'] 
        rounded to three and two significant digits, respectively.
    """

    exp2resultsperrun = {}
    exp2finalresults = {'experiment' : [], 'mean' : [], 'std' : []}
    for line in results.split('\n'):
        exp, res = line.split('\t')
        if exp not in exp2resultsperrun.keys():
            exp2resultsperrun[exp] = [float(res)]
        else:
            exp2resultsperrun[exp].append(float(res))
    for iexp, exp in enumerate(exp2resultsperrun):
        res = np.array(exp2resultsperrun[exp])
        final_res = float(res.mean())
        final_std = float(res.std())

        exp2finalresults['experiment'].append(exp)
        exp2finalresults['mean'].append(final_res)
        exp2finalresults['std'].append(final_std)
    df = pd.DataFrame.from_dict(exp2finalresults)
    df['mean'] = df['mean'].map(lambda x: format(x, '.3g'))
    df['std']  = df['std'].map(lambda x: format(x, '.2g'))
    return df

def get_avg_error_results(res):
    """Compute error averaged over *all* seeds and rotation axes per experiment.

    Args:
        res: Output dictionary produced by :func: get_error_results .

    Returns:
        dict:  avg[experiment][angle_deg] → error  with each value rounded to
        three significant digits. Missing combinations are ignored.
    """
    experiments = set()
    seeds = set()
    angles = set()
    rots = set()

    avg_res = {}
    for exp in res:
        avg_res[exp] = {}
        experiments.add(exp)
        for seed in res[exp]:
            seeds.add(seed)
            for angle in res[exp][seed]:
                angles.add(angle)
                for rot in res[exp][seed][angle]:
                    rots.add(rot)
                avg_res[exp][angle] = {}


    for experiment in experiments:
        for angle in angles:
            error = 0
            i = 0
            for seed in seeds:
                for rot in rots:
                    error += res[experiment][str(seed)][str(angle)][rot]
                    i+=1
            avg_res[experiment][angle] = round_to_kth_significant(error/i, 3)
    return(avg_res)