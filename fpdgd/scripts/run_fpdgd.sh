#!/bin/bash

# prepare directory
script_fn="${PWD}/${0}"
script_dir=$(dirname $script_fn)
proj_dir="${script_dir}/../"

# prepare env
export PYTHONPATH=$proj_dir
eval "$(conda shell.bash hook)"
conda activate federated
cd $proj_dir

python run_fpdgd.py




