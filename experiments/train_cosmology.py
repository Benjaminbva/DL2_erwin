import sys
import os
sys.path.append("../../")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
torch.set_float32_matmul_precision("high")
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from training import fit
from models.GATrErwin import EquivariantErwinTransformer
from models import ErwinTransformer
from experiments.datasets import CosmologyDataset
from experiments.wrappers.cosmology_equi import CosmologyEquiModel
from experiments.wrappers.cosmology import CosmologyModel
import ast


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="erwin",
                        help="Model type (mpnn, pointtransformer, erwin)")
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--size", type=str, default="small",
                        choices=["egsmall","smallest", "smaller", "smallish", "small", "medium", "large"],
                        help="Model size configuration")
    parser.add_argument("--num-samples", type=int, default=8192,
                        help="Number of samples for training")
    parser.add_argument("--num-epochs", type=int, default=3000,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--use-wandb", action="store_true", default=True,
                        help="Whether to use Weights & Biases for logging")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate")
    parser.add_argument("--val-every-iter", type=int, default=500,
                        help="Validation frequency in iterations")
    parser.add_argument("--experiment", type=str, default="glx_node",
                        help="Experiment name")
    parser.add_argument("--test", action="store_true", default=True,
                        help="Whether to run testing")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mpsteps", type=int, default=0)
    parser.add_argument("--auxiliarympnn", action="store_true", default=False)    
    parser.add_argument("--embedt", action="store_true", default=False)    
    parser.add_argument("--rotate", type=int, default=0)
    parser.add_argument("--rad", action="store_true", default=False,
                        help="Use radial‐basis function message passing")
    parser.add_argument("--eq-9", action="store_true", default=False)  
    parser.add_argument("--eq-12", action="store_true", default=False)  
    parser.add_argument("--eq-13", action="store_true", default=False)  
    
    

    
    return parser.parse_args()


dynamic_configs = {
    "erwin": {
        "smallest": {
            "c_in": 8,
            "c_hidden": [8, 16],
            "enc_num_heads": [2, 4],
            "enc_depths": [2, 2],
            "dec_num_heads": [2],
            "dec_depths": [2],
            "strides": [2],
            "ball_sizes": [128, 128],
            "rotate": 0,
            "mp_steps":0
        },
        "smaller": {
            "c_in": 8,
            "c_hidden": [8, 16, 32],
            "enc_num_heads": [2, 4, 8],
            "enc_depths": [2, 2, 6],
            "dec_num_heads": [2, 4],
            "dec_depths": [2, 2],
            "strides": [2, 2],
            "ball_sizes": [128, 128, 128],
            "rotate": 0,
            "mp_steps":0
        },
        "small": {
            "c_in": 32,
            "c_hidden": [32, 64, 128, 256],
            "enc_num_heads": [2, 4, 8, 16],
            "enc_depths": [2, 2, 6, 2],
            "dec_num_heads": [2, 4, 8],
            "dec_depths": [2, 2, 2],
            "strides": [2, 2, 2],
            "ball_sizes": [256, 256, 256, 256],
            "rotate": 0,
            "mp_steps":3
        },
        "smallish": {
            "c_in": 16,
            "c_hidden": [16, 32, 64, 128],
            "enc_num_heads": [2, 4, 8, 16],
            "enc_depths": [2, 2, 6, 2],
            "dec_num_heads": [2, 4, 8],
            "dec_depths": [2, 2, 2],
            "strides": [2, 2, 2],
            "ball_sizes": [256, 256, 256, 256],
            "rotate": 90,
            "mp_steps":3
        },
        "medium": {
            "c_in": 64,
            "c_hidden": [64, 128, 256, 512],
            "enc_num_heads": [2, 4, 8, 16],
            "enc_depths": [2, 2, 6, 2],
            "dec_num_heads": [2, 4, 8],
            "dec_depths": [2, 2, 2],
            "strides": [2, 2, 2],
            "ball_sizes": [256, 256, 256, 256],
            "rotate": 0,
        },
        "large": {
            "c_in": 128,
            "c_hidden": [128, 256, 512, 1024],
            "enc_num_heads": [2, 4, 8, 16],
            "enc_depths": [2, 2, 6, 2],
            "dec_num_heads": [2, 4, 8],
            "dec_depths": [2, 2, 2],
            "strides": [2, 2, 2],
            "ball_sizes": [256, 256, 256, 256],
            "rotate": 0,
            "mp_steps":3
        },
    },
    "GATrErwin":{
        "smallest": {
            "mv_dim_in": 8,
            "mv_dims": [8, 16],
            "s_dims": [8, 16],
            "enc_num_heads": [2, 4],
            "enc_depths": [2, 2],
            "dec_num_heads": [2],
            "dec_depths": [2],
            "strides": [2],
            "ball_sizes": [128, 128],
            "rotate": 0,
            "mp_steps":0,
        },
        "small": {
            "mv_dim_in": 8,
            "mv_dims": [32, 64, 128, 256],
            "s_dims": [32, 64, 128, 256],
            "enc_num_heads": [2, 4, 8, 16],
            "enc_depths": [2, 2, 6, 2],
            "dec_num_heads": [2, 4, 8],
            "dec_depths": [2, 2, 2],
            "strides": [2, 2, 2],
            "ball_sizes": [256, 256, 256, 256],
            "rotate": 0,
            "mp_steps":0,
        },
    }
}

model_cls = {
    "erwin": ErwinTransformer,
    "GATrErwin": EquivariantErwinTransformer,
}

wrapper_cls = {
    "erwin": CosmologyModel,
    "GATrErwin": CosmologyEquiModel,
}


if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_dataset = CosmologyDataset(
        task='node', 
        split='train', 
        num_samples=args.num_samples, 
        tfrecords_path=args.data_path, 
        knn=10,
    )
    val_dataset = CosmologyDataset(
        task='node', 
        split='val', 
        num_samples=512, 
        tfrecords_path=args.data_path, 
        knn=10,
    )
    test_dataset = CosmologyDataset(
        task='node', 
        split='test', 
        num_samples=512, 
        tfrecords_path=args.data_path, 
        knn=10,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=16,
    )
    
    valid_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=train_dataset.collate_fn,
        num_workers=16,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=train_dataset.collate_fn,
        num_workers=16,
    )

    if args.model not in dynamic_configs.keys():
        raise ValueError(f"Unknown model type: {args.model}, choose between {dynamic_configs.keys()}")
    
    model_config = dynamic_configs[args.model][args.size]
    model_config['mp_steps'] = args.mpsteps
    model_config['rotate'] = args.rotate
    if args.model == 'erwin':
        model_config['eq_9'] = args.eq_9
        model_config['eq_12'] = args.eq_12
        model_config['eq_13'] = args.eq_13

    if args.model == 'GATrErwin':
        if not args.auxiliarympnn:
            model_config['s_dims'] = [None] * len(model_config['mv_dims'])
        model_config['embedt'] = args.embedt
        model_config['rad'] = args.rad
    dynamic_model = model_cls[args.model](**model_config)

    if args.model not in wrapper_cls.keys():
        print('model has no wrapper, dont be silly wrap your... model')
        model = dynamic_model.cuda()
    else:
        model = wrapper_cls[args.model](dynamic_model).cuda()

    model = torch.compile(model)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)

    config = vars(args)
    config.update(model_config)
    print("num_parameters", sum(p.numel() for p in model.parameters() if p.requires_grad))
    fit(config, model, optimizer, scheduler, train_loader, valid_loader, test_loader, 100, 200)