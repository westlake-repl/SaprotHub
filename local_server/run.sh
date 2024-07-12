#!/bin/bash

source activate SaprotHub

# Checkout the version in the cache
ori_dir=$(pwd)
if [ -d "/root/.cache/SaprotHub/SaprotHub" ];then
  cd /root/.cache/SaprotHub/SaprotHub
  local=$(git rev-parse HEAD)
  remote=$(git ls-remote https://github.com/westlake-repl/SaprotHub.git | grep HEAD)
  remote=(${remote//,/ }[0])
  if [ "$local" != "$remote" ]; then
      echo "The version is not the latest. Updating..."
      git fetch --all &&  git reset --hard origin/main && git pull
      pip install -r local_server/requirements.txt
      pip uninstall saprot --yes
  fi
fi

# Run the server
jupyter notebook --config $ori_dir/jupyter_notebook_config.py --allow-root