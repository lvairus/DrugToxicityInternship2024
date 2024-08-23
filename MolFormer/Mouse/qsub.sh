#!/bin/bash -l
#PBS -N regmulticlass_tox
#PBS -l select=1
#PBS -m abe
#PBS -M lvairus@anl.gov
#PBS -l walltime=3:00:00
#PBS -q preemptable
#PBS -l filesystems=home:eagle
#PBS -A datascience 
#PBS -o logs/regmulticlass_tox.OU
#PBS -e logs/regmulticlass_tox.ER

module use /soft/modulefiles
module load conda
conda activate /lus/eagle/projects/datascience/avasan/envs/sst_tf216 

cd /lus/eagle/projects/datascience/lvairus/Pharmacokinetic_Modeling/ModelTraining/Molformer_Mouse

# change sweep_id
CUDA_VISIBLE_DEVICES=0 wandb agent 'lvairusorg/Mouse/b89ngao1' & 
CUDA_VISIBLE_DEVICES=1 wandb agent 'lvairusorg/Mouse/b89ngao1' &
CUDA_VISIBLE_DEVICES=2 wandb agent 'lvairusorg/Mouse/b89ngao1' &
CUDA_VISIBLE_DEVICES=3 wandb agent 'lvairusorg/Mouse/b89ngao1' &

wait

# CUDA_VISIBLE_DEVICES=0 python runscript_multiclass.py

