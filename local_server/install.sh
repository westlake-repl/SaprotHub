#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_DIR=$(dirname "$SCRIPT_DIR")
CACHE_DIR="/root/.cache/SaprotHub"

# Find conda
for conda_sh in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh" "/opt/miniconda3/etc/profile.d/conda.sh"; do
    if [ -f "$conda_sh" ]; then
        source "$conda_sh"
        break
    fi
done
if ! command -v conda &>/dev/null; then
    echo "Error: conda not found. Please install Miniconda/Anaconda first."
    exit 1
fi

# Create/activate environment
if ! conda env list | grep -q '^SaprotHub '; then
    conda create -n SaprotHub python=3.10.15 -y
fi
conda activate SaprotHub

mkdir -p "$CACHE_DIR"

# Install zip on Linux if it is missing (optional, notebook shells may use it)
if [ "$(uname)" = "Linux" ] && ! command -v zip &>/dev/null; then
    apt-get update && apt-get -y install zip || true
fi

# Copy repository and notebook to the Jupyter notebook directory
rm -rf "$CACHE_DIR/SaprotHub"
cp -r "$REPO_DIR" "$CACHE_DIR/SaprotHub"
cp "$REPO_DIR/colab/SaprotHub_v2.ipynb" "$CACHE_DIR/"

# Install colabtools (provides google.colab)
if [ ! -d "$CACHE_DIR/colabtools/.git" ]; then
    rm -rf "$CACHE_DIR/colabtools"
    git clone https://github.com/googlecolab/colabtools.git "$CACHE_DIR/colabtools"
    cd "$CACHE_DIR/colabtools"
    git checkout e8519e12f553b0597c0e067cd9e4df821bdc6b2e
fi
cd "$CACHE_DIR/colabtools"
pip install .

# Install Python dependencies
pip install -r "$SCRIPT_DIR/requirements.txt"

# Install the saprot package
pip install "$CACHE_DIR/SaprotHub"

# Patch google.colab to work with local widgets
colab_dir=$CONDA_PREFIX/lib/python3.10/site-packages/google/colab
cp "$SCRIPT_DIR/data_table.py" "$colab_dir/data_table.py"
cp "$SCRIPT_DIR/_reprs.py" "$colab_dir/_reprs.py"

echo "Installation finished. Run 'bash run.sh' to start Jupyter."