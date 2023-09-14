#!/bin/bash
#
#SBATCH --job-name=evg_segmentation
#SBATCH --partition=cpu
#SBATCH --array=0-9
#SBATCH --cpus-per-task=1
#SBATCH --time=40:00:00
#SBATCH --output=/hpc/dhl_ec/VirtualSlides/EVG/beautiful_plaques/logs_evg_segm/%a.log
#SBATCH --error=/hpc/dhl_ec/VirtualSlides/EVG/beautiful_plaques/logs_evg_segm/%a.errors
#SBATCH --mem=250000M
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s.w.vanderlaan-2@umcutrecht.nl
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

# Load conda environment
# make sure to create the proper environment first, and next activate it
eval "$(conda shell.bash hook)"
conda activate convocals

# edit the following line to run your script for the given paths and stain
python3 CONVOCALS/wsi_preprocessing/segmentation.py \
--index=$SLURM_ARRAY_TASK_ID --num_tasks=$SLURM_ARRAY_TASK_COUNT \
--slide_dir="VirtualSlides" \
--output_dir="VirtualSlides/PROCESSED" \
--masks_dir="VirtualSlides/PROCESSED_TEST/masks/" \
--model CONVOCALS/wsi_preprocessing//PathProfiler/tissue_segmentation/checkpoint_ts.pth