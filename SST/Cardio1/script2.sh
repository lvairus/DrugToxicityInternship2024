#!/bin/bash -l
#PBS -N sst_training
#PBS -l select=1
#PBS -m abe
#PBS -M lvairus@anl.gov
#PBS -l walltime=0:10:00
#PBS -q preemptable
#PBS -l filesystems=grand
#PBS -A datascience 

module use /soft/modulefiles
module load conda
conda activate /lus/eagle/projects/datascience/lvairus/envs/sst   

cd /lus/eagle/projects/datascience/lvairus/SST_Pytorch

export CUDA_VISIBLE_DEVICES=0,1,2,3
python sst_reg_train_opt.py -t ../data/Acute_Toxicity_mouse_intraperitoneal_LD50.csv -v ../data/Acute_Toxicity_mouse_intraperitoneal_LD50_val.csv -s Canonical_SMILES -l Toxicity_Value -c config_mod.json -e 10 -b 128
