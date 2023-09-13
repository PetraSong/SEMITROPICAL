#!/bin/bash
#SBATCH --job-name=vit
#SBATCH --partition=cpu
#SBATCH --array=0-999
#SBATCH --cpus-per-task=2
#SBATCH --time=60:00:00
#SBATCH --output=./logs_fe/%a.log
#SBATCH --mem=200000M

echo '----------------------------'
echo ' JOB ID: '$SLURM_ARRAY_JOB_ID
echo ' CURRENT TASK ID: '$SLURM_JOB_ID
echo ' CURRENT TASK NUMBER: '$SLURM_ARRAY_TASK_ID
echo '----------------------------'
echo ' MIN TASK ID: '$SLURM_ARRAY_TASK_MIN
echo ' MAX TASK ID: '$SLURM_ARRAY_TASK_MAX
echo ' TOTAL NUMBER OF TASKS: '$SLURM_ARRAY_TASK_COUNT
echo '----------------------------'

eval "$(conda shell.bash hook)"

conda  activate /hpc/dhl_ec/VirtualSlides/cglastonbury/wsi
python3 extract_features.py  -index=$SLURM_ARRAY_TASK_ID -num_tasks=$SLURM_ARRAY_TASK_COUNT -features_extraction_checkpoint /hpc/dhl_ec/fcisternino/checkpoints/checkpoint_ViT_AT.pth