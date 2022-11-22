#!/usr/bin/env bash
#SBATCH --job-name=dynamic
#SBATCH --output=results/log/dynamic%j.log
#SBATCH --error=results/err/dynamic%j.err
#SBATCH --mail-user=coello@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
echo "Hello"
        

srun anaconda3/bin/python conda/SRP_cluster/main_stud_dynamic.py        # python jobs require the srun command to work
echo "Running"
