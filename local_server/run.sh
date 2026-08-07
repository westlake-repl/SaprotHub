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

if ! conda env list | grep -q '^SaprotHub '; then
    echo "Error: SaprotHub environment not found. Please run install.sh first."
    exit 1
fi
conda activate SaprotHub

repo_cache="$CACHE_DIR/SaprotHub"

# Ensure the SaprotHub repository exists in the cache directory
if [ ! -d "$repo_cache" ]; then
    cp -r "$REPO_DIR" "$repo_cache"
fi

# Ensure the notebook exists in the cache directory
cp -f "$repo_cache/colab/SaprotHub_v2.ipynb" "$CACHE_DIR/"

# Ensure saprot is installed (idempotent)
pip install -q "$repo_cache"

# Patch google.colab to work with local widgets
colab_dir=$CONDA_PREFIX/lib/python3.10/site-packages/google/colab
cp "$SCRIPT_DIR/data_table.py" "$colab_dir/data_table.py"
cp "$SCRIPT_DIR/_reprs.py" "$colab_dir/_reprs.py"

# Switch to the cache root so notebook cells see the correct working directory
cd "$CACHE_DIR"

# Start Jupyter
jupyter notebook --config "$SCRIPT_DIR/jupyter_notebook_config.py" \
                 --notebook-dir "$CACHE_DIR" \
                 --allow-root