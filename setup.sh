#!/bin/bash

# make sure uv is installed, if not: 
# curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv erwin
source erwin/bin/activate

MINIMAL=false
if [[ "$*" == *"--minimal"* ]]; then
    MINIMAL=true
fi

if [ "$MINIMAL" = true ]; then
    echo "Installing minimal dependencies [Erwin]"
    uv pip install torch==2.6.0 # 2.5 required for torch-scatter, can be ommited for Erwin
    uv pip install torch-cluster -f https://data.pyg.org/whl/torch-2.6.0+cu124.html # CUDA 12.4
    uv pip install numpy 
    uv pip install einops
    uv pip install Cython
    uv pip install setuptools
    #GATr
    uv pip install xformers==0.0.29.post2
else
    echo "Installing all dependencies [Erwin + baselines + experiments]"

    # Erwin dependencies
    uv pip install torch==2.6.0 # 2.5 required for torch-scatter, can be ommited for Erwin
    uv pip install torch-cluster -f https://data.pyg.org/whl/torch-2.6.0+cu124.html # CUDA 12.4
    uv pip install numpy 
    uv pip install einops
    uv pip install Cython
    uv pip install setuptools

    # PointTransformer v3 dependencies
    uv pip install addict
    uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html # CUDA 12.4
    uv pip install spconv-cu120
    uv pip install timm

    #uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu126.html 

    # MD dependencies
    uv pip install h5py
    # cosmology dependencies
    uv pip install tensorflow
    # ShapeNet-Car dependencies
    # see https://github.com/ml-jku/UPT/blob/main/SETUP_DATA.md

    # misc dependencies
    uv pip install wandb
    uv pip install tqdm
    uv pip install matplotlib

    #GATr
    uv pip install xformers==0.0.29.post2
    uv pip install pandas
fi

# install c++ balltree implementation
cd balltree
python setup.py install

cd ..





