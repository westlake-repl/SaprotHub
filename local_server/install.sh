#!/bin/bash
conda create -n SaprotHub python=3.10.15 --yes
source activate SaprotHub

mkdir -p /root/.cache/SaprotHub

apt-get -y install zip

# Install packages
git clone https://github.com/googlecolab/colabtools.git /root/.cache/SaprotHub/colabtools
pip install /root/.cache/SaprotHub/colabtools/
pip install -r ./requirements.txt

# Overwrite files to properly call colab
colab_dir=$CONDA_PREFIX/lib/python3.10/site-packages/google/colab
cp ./data_table.py $colab_dir/data_table.py
cp ./_reprs.py $colab_dir/_reprs.py