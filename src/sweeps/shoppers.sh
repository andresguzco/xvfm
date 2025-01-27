#!/bin/bash
#SBATCH --partition=rtx6000
#SBATCH --qos=m4

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --array=1-16%5
#SBATCH --output=slurm/slurm-%A_%a.out

echo "[DEBUG] Host name: " `hostname`

source  ~/miniforge3/etc/profile.d/conda.sh
conda activate eval

wandb agent --count 1 andresguzco/XVFM/aojvbd3l