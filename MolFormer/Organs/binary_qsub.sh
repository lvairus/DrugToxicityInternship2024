#!/bin/bash -l
#PBS -N cardio1
#PBS -l select=1
#PBS -m abe
#PBS -M lvairus@anl.gov
#PBS -l walltime=0:30:00
#PBS -q debug
#PBS -l filesystems=home:eagle
#PBS -A datascience 
#PBS -o logs/cardio1.OU
#PBS -e logs/cardio1.ER

module use /soft/modulefiles
module load conda
conda activate /lus/eagle/projects/datascience/lvairus/envs/sst   

cd /lus/eagle/projects/datascience/lvairus/Pharmacokinetic_Modeling/ModelTraining/MolFormer_Toxric

export CUDA_VISIBLE_DEVICES=0,1,2,3

python binary_runcscript.py
