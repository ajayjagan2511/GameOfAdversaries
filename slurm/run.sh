#!/bin/bash
#SBATCH --job-name=goa
#SBATCH --output=small_10_10_%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=15:00:00

module purge
module load WebProxy
export http_proxy=http://10.73.132.63:8080
export https_proxy=http://10.73.132.63:8080

PROJECT_DIR="/scratch/user/aaupadhy/college/projects/GameOfAdversaries"
cd $PROJECT_DIR
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

export HUGGINGFACE_TOKEN="PLEASE ENTER API KEY"

COMPUTE_NODE=$(hostname -s)
echo "ssh -N -L 8787:${COMPUTE_NODE}:8787 aaupadhy@grace.hprc.tamu.edu"

source ~/.bashrc
conda activate ML

export XDG_CACHE_HOME=/scratch/user/aaupadhy/.cache
export HF_HOME=$XDG_CACHE_HOME/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HUB_CACHE
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets

echo "Job started at $(date)"
echo "Running on ${COMPUTE_NODE}.grace.hprc.tamu.edu"
nvidia-smi

python main.py --attack --visualize \
               --max_prompts 20 \
               --max_steps 20 \
               --checkpoint ./checkpoints/safety_head.pt \
               --run_id small_10_10
echo "Job finished at $(date)"
