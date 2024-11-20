#!/bin/bash
sbatch slurm_launcher.slrm main.py \
  --model_type "vfm" \
  --num_epochs 5000 \
  --batch_size 256 \
  --vfm_loss "Gaussian" \
  --dataset "two_moons" \
  # --learn_sigma