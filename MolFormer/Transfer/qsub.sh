#!/bin/bash -l
#PBS -N mouse_pretrain
#PBS -l select=1
#PBS -m abe
#PBS -M lvairus@anl.gov
#PBS -l walltime=5:00:00
#PBS -q preemptable
#PBS -l filesystems=home:eagle
#PBS -A datascience 
#PBS -o logs/mouse_pretrain.OU
#PBS -e logs/mouse_pretrain.ER

module use /soft/modulefiles
module load conda
conda activate /lus/eagle/projects/datascience/avasan/envs/sst_tf216 

cd /lus/eagle/projects/datascience/lvairus/Pharmacokinetic_Modeling/ModelTraining/Molformer_Transfer

# change sweep_id
CUDA_VISIBLE_DEVICES=0 wandb agent 'lvairusorg/Transfer/l169bnwc' & 
CUDA_VISIBLE_DEVICES=1 wandb agent 'lvairusorg/Transfer/l169bnwc' &
CUDA_VISIBLE_DEVICES=2 wandb agent 'lvairusorg/Transfer/l169bnwc' &
CUDA_VISIBLE_DEVICES=3 wandb agent 'lvairusorg/Transfer/l169bnwc' &

wait

# # single run
# CUDA_VISIBLE_DEVICES=0 python runscript_multiclass.py

