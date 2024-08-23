#!/bin/bash -l
#PBS -N cardio5
#PBS -l select=1
#PBS -m abe
#PBS -M lvairus@anl.gov
#PBS -l walltime=0:30:00
#PBS -q debug
#PBS -l filesystems=home:eagle
#PBS -A datascience 
#PBS -o logs/cardio5.OU
#PBS -e logs/cardio5.ER

module use /soft/modulefiles
module load conda
conda activate /lus/eagle/projects/datascience/lvairus/envs/sst   

cd /lus/eagle/projects/datascience/lvairus/Pharmacokinetic_Modeling/ModelTraining/MolFormer_Toxric

# change sweep_id
CUDA_VISIBLE_DEVICES=0 wandb agent 'lvairusorg/Toxric/fmt0brk0' & 
CUDA_VISIBLE_DEVICES=1 wandb agent 'lvairusorg/Toxric/fmt0brk0' &
CUDA_VISIBLE_DEVICES=2 wandb agent 'lvairusorg/Toxric/fmt0brk0' &
CUDA_VISIBLE_DEVICES=3 wandb agent 'lvairusorg/Toxric/fmt0brk0' &

# Wait for all background processes to finish
wait
