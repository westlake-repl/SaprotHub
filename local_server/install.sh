#!/bin/bash
conda create -n SaprotHub python=3.10.15 --yes
source activate SaprotHub

mkdir -p /root/.cache/SaprotHub

apt-get -y install zip

local_server_dir=$(pwd)

# Install packages
git clone https://github.com/googlecolab/colabtools.git /root/.cache/SaprotHub/colabtools
cd /root/.cache/SaprotHub/colabtools
git checkout e8519e12f553b0597c0e067cd9e4df821bdc6b2e
pip install /root/.cache/SaprotHub/colabtools/

cd $local_server_dir
pip install -r ./requirements.txt

# Overwrite files to properly call colab
colab_dir=$CONDA_PREFIX/lib/python3.10/site-packages/google/colab
cp ./data_table.py $colab_dir/data_table.py
cp ./_reprs.py $colab_dir/_reprs.py