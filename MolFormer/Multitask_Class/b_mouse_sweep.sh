#!/bin/bash -l
#PBS -N t_mouse_10
#PBS -l select=1
#PBS -m abe
#PBS -M lvairus@anl.gov
#PBS -l walltime=2:00:00
#PBS -q preemptable
#PBS -l filesystems=home:eagle
#PBS -A datascience 
#PBS -o logs/t_mouse_10.OU
#PBS -e logs/t_mouse_10.ER

module use /soft/modulefiles
module load conda
conda activate /lus/eagle/projects/datascience/lvairus/envs/sst   

cd /lus/eagle/projects/datascience/lvairus/Pharmacokinetic_Modeling/ModelTraining/MolFormer_MultiTask_Class

# change sweep_id
CUDA_VISIBLE_DEVICES=0 wandb agent 'lvairusorg/Multitask_Class_Oral/ko2ytrbn' & 
CUDA_VISIBLE_DEVICES=1 wandb agent 'lvairusorg/Multitask_Class_Oral/ko2ytrbn' &
CUDA_VISIBLE_DEVICES=2 wandb agent 'lvairusorg/Multitask_Class_Oral/ko2ytrbn' &
CUDA_VISIBLE_DEVICES=3 wandb agent 'lvairusorg/Multitask_Class_Oral/ko2ytrbn'

