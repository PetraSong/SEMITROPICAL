#!/bin/bash
#
#SBATCH --job-name=evg_segmentation # set the name of the job
#SBATCH --partition=cpu # set the partition for the job
#SBATCH --array=0-9 # set the number of array jobs to be executed; when there are 2 workers and 1000 slides, 2 slides will be processed per array job, so --array=0-499
#SBATCH --cpus-per-task=1 # set the number of cores
#SBATCH --time=40:00:00 # set the total run time limit (HH:MM:SS)
#SBATCH --output=/directory_where_the_wsi_are/logs_evg_segm/%a.log # make sure to create the logs_evg_segm folder first
#SBATCH --error=/directory_where_the_wsi_are/logs_evg_segm/%a.errors # make sure to create the logs_evg_segm folder first
#SBATCH --mem=250000M # set the memory limit
#SBATCH --mail-type=ALL # select which email types will be sent (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=your_email@domain.com # set destination email
#
# Description: Script to run segmentation on STAINED slides
echo '----------------------------'
echo ' JOB ID: '$SLURM_ARRAY_JOB_ID
echo ' CURRENT TASK ID: '$SLURM_JOB_ID
echo ' CURRENT TASK NUMBER: '$SLURM_ARRAY_TASK_ID
echo '----------------------------'
echo ' MIN TASK ID: '$SLURM_ARRAY_TASK_MIN
echo ' MAX TASK ID: '$SLURM_ARRAY_TASK_MAX
echo ' TOTAL NUMBER OF TASKS: '$SLURM_ARRAY_TASK_COUNT
echo '----------------------------'

CONVOCALS="/directory_where_this_repo_is/CONVOCALS"
PROJECT_DIR="/directory_where_the_wsi_are"

# Load conda environment
# make sure to create the proper environment first, and next activate it
eval "$(conda shell.bash hook)"
conda activate convocals

# edit the following line to run your script for the given paths and stain
python3 $CONVOCALS/wsi_preprocessing/segmentation.py \
--index=$SLURM_ARRAY_TASK_ID --num_tasks=$SLURM_ARRAY_TASK_COUNT \
--slide_dir="${PROJECT_DIR}" \
--output_dir="${PROJECT_DIR}/PROCESSED" \
--masks_dir="${PROJECT_DIR}/PROCESSED/masks/" \
--model $CONVOCALS/wsi_preprocessing/PathProfiler/tissue_segmentation/checkpoint_ts.pth