#!/bin/bash -l
#PBS -N stask1
#PBS -l select=1
#PBS -m abe
#PBS -M lvairus@anl.gov
#PBS -l walltime=0:30:00
#PBS -q debug
#PBS -l filesystems=home:eagle
#PBS -A datascience 
#PBS -o logs/stask1.OU
#PBS -e logs/stask1.ER

module use /soft/modulefiles
module load conda
conda activate /lus/eagle/projects/datascience/lvairus/envs/sst   

cd /lus/eagle/projects/datascience/lvairus/Pharmacokinetic_Modeling/ModelTraining/MolFormer_MultiTask_Class

export CUDA_VISIBLE_DEVICES=0 python run_script_lv_multi_sweep.py -d cat -s SMILES -l EPACategoryIndex -t 0.2 -E 10
export CUDA_VISIBLE_DEVICES=1 python run_script_lv_multi_sweep.py -d dog -s SMILES -l EPACategoryIndex -t 0.2 -E 10
export CUDA_VISIBLE_DEVICES=2 python run_script_lv_multi_sweep.py -d man -s SMILES -l EPACategoryIndex -t 0.2 -E 10
export CUDA_VISIBLE_DEVICES=3 python run_script_lv_multi_sweep.py -d woman -s SMILES -l EPACategoryIndex -t 0.2 -E 10

