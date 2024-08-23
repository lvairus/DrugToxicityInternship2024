#!/bin/bash -l
#PBS -N stask_ll
#PBS -l select=1
#PBS -m abe
#PBS -M lvairus@anl.gov
#PBS -l walltime=0:20:00
#PBS -q debug
#PBS -l filesystems=home:eagle
#PBS -A datascience 
#PBS -o logs/stask_ll.OU
#PBS -e logs/stask_ll.ER

module use /soft/modulefiles
module load conda
conda activate /lus/eagle/projects/datascience/lvairus/envs/sst   

cd /lus/eagle/projects/datascience/lvairus/Pharmacokinetic_Modeling/ModelTraining/MolFormer_MultiTask_Class


# # Create the sweep and capture the output
# output=$(wandb sweep sweep.yaml)

# # Extract the sweep ID from the output
# sweep_id=$(echo "$output" | awk '/Creating sweep with ID:/{print $5}')

# # Print the sweep ID
# echo "Sweep ID: $sweep_id"

# Use the sweep ID as needed
CUDA_VISIBLE_DEVICES=0 wandb agent lvairusorg/Multitask_Class_Oral/pzwwa3qr & 
CUDA_VISIBLE_DEVICES=1 wandb agent lvairusorg/Multitask_Class_Oral/pzwwa3qr & 
CUDA_VISIBLE_DEVICES=2 wandb agent lvairusorg/Multitask_Class_Oral/pzwwa3qr & 
CUDA_VISIBLE_DEVICES=3 wandb agent lvairusorg/Multitask_Class_Oral/pzwwa3qr
