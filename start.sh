#!/usr/bin/env bash
set -e

mv /quickpod/Truth_is_Universal /workspace/Truth_is_Universal/
cd ./Truth_is_Universal
git stash
git pull

conda activate truth_is_universal
python -m ipykernel install --user --name=truth_is_universal --display-name "Python (truth_is_universal)"