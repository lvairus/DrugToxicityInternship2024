#!/bin/bash -l
#PBS -N b_mouse
#PBS -l select=1
#PBS -m abe
#PBS -M lvairus@anl.gov
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -l filesystems=home:eagle
#PBS -A datascience 
#PBS -o logs/b_mouse.OU
#PBS -e logs/b_mouse.ER

module use /soft/modulefiles
module load conda
conda activate /lus/eagle/projects/datascience/lvairus/envs/sst   

cd /lus/eagle/projects/datascience/lvairus/Pharmacokinetic_Modeling/ModelTraining/MolFormer_MultiTask_Class


CUDA_VISIBLE_DEVICES=0 python run_script_lv_mouse_bi.py

