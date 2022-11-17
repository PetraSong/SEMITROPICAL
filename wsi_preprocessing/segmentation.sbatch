#!/bin/bash
#SBATCH --job-name=segmentation
#SBATCH --partition=cpu
#SBATCH --array=0-9
#SBATCH --cpus-per-task=1
#SBATCH --time=40:00:00
#SBATCH --output=./logs_segm/%a.log
#SBATCH --mem=250000M


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

python3 segmentation.py  -index=$SLURM_ARRAY_TASK_ID -num_tasks=$SLURM_ARRAY_TASK_COUNT