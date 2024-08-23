#!/bin/bash -l
#PBS -N mtask
#PBS -l select=1
#PBS -m abe
#PBS -M lvairus@anl.gov
#PBS -l walltime=0:20:00
#PBS -q debug
#PBS -l filesystems=home:eagle
#PBS -A datascience 
#PBS -o logs/mtask.OU
#PBS -e logs/mtask.ER

module use /soft/modulefiles
module load conda
conda activate /lus/eagle/projects/datascience/lvairus/envs/sst   

cd /lus/eagle/projects/datascience/lvairus/Pharmacokinetic_Modeling/ModelTraining/MolFormer_MultiTask_Class

# Copy in the sweep ID
CUDA_VISIBLE_DEVICES=0 wandb agent "lvairusorg/Multitask Class Oral Test/3fkewbhl" & 
CUDA_VISIBLE_DEVICES=1 wandb agent "lvairusorg/Multitask Class Oral Test/3fkewbhl" & 
CUDA_VISIBLE_DEVICES=2 wandb agent "lvairusorg/Multitask Class Oral Test/3fkewbhl" & 
CUDA_VISIBLE_DEVICES=3 wandb agent "lvairusorg/Multitask Class Oral Test/3fkewbhl"
