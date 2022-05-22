#!/usr/bin/env bash


export PATH=~/miniconda/bin:$PATH
wait
# create environment
conda create -n mmact python=3.8 -y
wait
eval "$(conda shell.bash hook)"
conda activate mmact
wait
conda install -y pytorch=1.9.0 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
wait
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
wait 
pip install -r requirements/build.txt
pip install -r requirements/runtime.txt
pip install -v -e .  # or "python setup.py develop"


