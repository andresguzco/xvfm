#!/bin/bash
# structures=("vector" "scalar") # "matrix"
# data=("two_moons" "mnist")

# for dataset in "${data[@]}"; do
#   for structure in "${structures[@]}"; do
#     sbatch slurm_launcher.slrm main.py \
#       --num_epochs 1000 \
#       --log_interval 100 \
#       --loss_fn "Gaussian" \
#       --dataset "${dataset}" \
#       --learned_structure "${structure}"
#   done
# done

data=("two_moons" "mnist")

for dataset in "${data[@]}"; do
  sbatch slurm_launcher.slrm main.py \
    --num_epochs 1000 \
    --log_interval 100 \
    --loss_fn "Gaussian" \
    --dataset "${dataset}"
done

# sbatch slurm_launcher.slrm main.py \
#   --num_epochs 1000 \
#   --log_interval 100 \
#   --loss_fn "Gaussian" \
#   --dataset "mnist" \
#   --learned_structure "scalar"