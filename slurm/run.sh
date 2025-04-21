#!/bin/bash
#SBATCH --job-name=goa
#SBATCH --output=%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=140G
#SBATCH --time=15:00:00

module purge
module load WebProxy
export http_proxy=http://10.73.132.63:8080
export https_proxy=http://10.73.132.63:8080

PROJECT_DIR="/scratch/user/aaupadhy/college/projects/GameOfAdversaries"
cd $PROJECT_DIR
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"


COMPUTE_NODE=$(hostname -s)
echo "ssh -N -L 8787:${COMPUTE_NODE}:8787 aaupadhy@grace.hprc.tamu.edu"

source ~/.bashrc
conda activate ML

echo "Job started at $(date)"
echo "Running on ${COMPUTE_NODE}.grace.hprc.tamu.edu"
nvidia-smi

python main.py --data data/harmful_prompts_20000_diverse.csv
echo "Job finished at $(date)"