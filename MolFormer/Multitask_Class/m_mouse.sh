#!/bin/bash -l
#PBS -N m_mouse89
#PBS -l select=1
#PBS -m abe
#PBS -M lvairus@anl.gov
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -l filesystems=home:eagle
#PBS -A datascience 
#PBS -o logs/m_mouse.OU
#PBS -e logs/m_mouse.ER

module use /soft/modulefiles
module load conda
conda activate /lus/eagle/projects/datascience/lvairus/envs/sst   

cd /lus/eagle/projects/datascience/lvairus/Pharmacokinetic_Modeling/ModelTraining/MolFormer_MultiTask_Class


CUDA_VISIBLE_DEVICES=0 python run_script_lv_multi.py -y m_mouse0.yaml &
CUDA_VISIBLE_DEVICES=1 python run_script_lv_multi.py -y m_mouse1.yaml 

