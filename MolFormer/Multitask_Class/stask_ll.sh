#!/bin/bash -l
#PBS -N stask_ll
#PBS -l select=1
#PBS -m abe
#PBS -M lvairus@anl.gov
#PBS -l walltime=0:10:00
#PBS -q debug
#PBS -l filesystems=home:eagle
#PBS -A datascience 
#PBS -o logs/stask_ll.OU
#PBS -e logs/stask_ll.ER

module use /soft/modulefiles
module load conda
conda activate /lus/eagle/projects/datascience/lvairus/envs/sst   

cd /lus/eagle/projects/datascience/lvairus/Pharmacokinetic_Modeling/ModelTraining/MolFormer_MultiTask_Class

if [ -f sweep_id.txt ]; then
    rm sweep_id.txt
fi

CUDA_VISIBLE_DEVICES=0 python run_script_lv_multi_sweep.py -d gpig -s SMILES -l EPACategoryIndex -t 0.2 -E 10 &

# Wait until sweep_id.txt is updated
while [ ! -f sweep_id.txt ]; do
    sleep 1
done
# Read the sweep_id from the file
sweep_id=$(cat sweep_id.txt)

CUDA_VISIBLE_DEVICES=1 wandb agent "lvairusorg/Multitask_Class_Oral/${sweep_id}" & 
CUDA_VISIBLE_DEVICES=2 wandb agent "lvairusorg/Multitask_Class_Oral/${sweep_id}" &
CUDA_VISIBLE_DEVICES=3 wandb agent "lvairusorg/Multitask_Class_Oral/${sweep_id}"
