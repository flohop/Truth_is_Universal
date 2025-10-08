#!/usr/bin/env bash
set -e

if [ -d "/quickpod/Truth_is_Universal" ]; then
  mv /quickpod/Truth_is_Universal /workspace/Truth_is_Universal/
fi
cd ./Truth_is_Universal
git stash
git pull

conda activate truth_is_universal
python -m ipykernel install --user --name=truth_is_universal --display-name "Python (truth_is_universal)"