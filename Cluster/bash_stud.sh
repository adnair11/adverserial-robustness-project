#!/usr/bin/env bash
#SBATCH --job-name=adv_stud
#SBATCH --output=adv_stud%j.log
#SBATCH --error=adv_stud%j.err
#SBATCH --mail-user=coello@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
echo "Hello"
        

srun anaconda3/bin/python conda/SRP_cluster/main_stud.py        # python jobs require the srun command to work
echo "Running"
