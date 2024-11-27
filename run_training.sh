#!/bin/bash
# strucures=("vector") # "scalar" "matrix"
# data=("two_moons" "mnist")

# for dataset in "${data[@]}"; do
#   for structure in "${structures[@]}"; do
#     sbatch slurm_launcher.slrm main.py \
#       --vfm_loss "Gaussian" \
#       --dataset "$dataset" \
#       --learned_structure "$structure"
#       # --learn_sigma true
#   done
# done

sbatch slurm_launcher.slrm main.py \
  --loss_fn "Gaussian" \
  --dataset "mnist" \
  --learned_structure "scalar"