#!/bin/bash -l
#PBS -N mtask
#PBS -l select=1
#PBS -m abe
#PBS -M lvairus@anl.gov
#PBS -l walltime=2:00:00
#PBS -q preemptable
#PBS -l filesystems=home:eagle
#PBS -A datascience 
#PBS -o logs/mtask.OU
#PBS -e logs/mtask.ER

module use /soft/modulefiles
module load conda
conda activate /lus/eagle/projects/datascience/lvairus/envs/sst   

cd /lus/eagle/projects/datascience/lvairus/Pharmacokinetic_Modeling/ModelTraining/MolFormer_MultiTask_Class

CUDA_VISIBLE_DEVICES=0 python run_script_lv_multi.py -y multi0.yaml &
CUDA_VISIBLE_DEVICES=1 python run_script_lv_multi.py -y multi1.yaml &
CUDA_VISIBLE_DEVICES=2 python run_script_lv_multi.py -y multi2.yaml &
CUDA_VISIBLE_DEVICES=3 python run_script_lv_multi.py -y multi3.yaml
