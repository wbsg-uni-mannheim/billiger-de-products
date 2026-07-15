#!/bin/bash
#SBATCH --job-name=run_hiergat
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=110:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --output=slurm_runs/logs/run_hiergat%j.out
#SBATCH --error=slurm_runs/logs/run_hiergat%j.err

source /home/aasteine/miniconda3/etc/profile.d/conda.sh
conda activate hier_env

# Force conda's newer libstdc++ to be found first
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

which python

echo "CUDA devices available:"
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
nvidia-smi

python -u src/models/hiergat/all_runs.py