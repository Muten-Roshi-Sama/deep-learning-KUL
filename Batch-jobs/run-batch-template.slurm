#!/bin/bash -l
#SBATCH --account=lp_deeplearning_gt_2026b
#SBATCH --cluster=wice
#SBATCH --partition=gpu_v100
#SBATCH --nodes=1 --ntasks=1
#SBATCH --gres=shard:1
#SBATCH --time=10:00
#SBATCH –output=logs-%j.out
#SBATCH –error=slurm-%j.err
module load PyTorch/2.9.1-foss-2025a-CUDA-12.8.0-whl
cd $VSC_DATA/dl-lab1
python lab-session-1-part1.py --skip-demos --skip-plots