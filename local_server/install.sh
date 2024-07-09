#!/bin/bash
conda create -n SaprotHub python=3.10 --yes
source activate SaprotHub

mkdir -p /root/.cache/SaprotHub

# Install packages
git clone https://github.com/googlecolab/colabtools.git /root/.cache/SaprotHub/colabtools
pip install /root/.cache/SaprotHub/colabtools/
pip install traitlets==5.9.0
pip install pillow==10.4.0
pip install httplib2==0.22.0
pip install ipywidgets==7.7.1
pip install 'numpy<2'
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install pytorch-lightning==2.1.3
pip install loguru==0.7.2
pip install easydict
pip install colorama==0.4.6
pip install transformers==4.28.0
pip install peft==0.10.0
pip install lmdb==1.4.1
pip install biopython==1.83
pip install wandb==0.17.4
pip install matplotlib==3.9.1
pip install torchmetrics==0.9.3
pip install multiprocess

# Overwrite files for properly calling colab
colab_dir=$CONDA_PREFIX/lib/python3.10/site-packages/google/colab
cp ./data_table.py $colab_dir/data_table.py
cp ./_reprs.py $colab_dir/_reprs.py