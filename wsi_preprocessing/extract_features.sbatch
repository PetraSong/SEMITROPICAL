#!/bin/bash
#
#SBATCH --job-name=evg_feat # set the name of the job
#SBATCH --partition=cpu # set the partition for the job
#SBATCH --array=0-999 # set the number of array jobs to be executed; when there are 2 workers and 1000 slides, 2 slides will be processed per array job, so --array=0-499
#SBATCH --cpus-per-task=2 # set the number of cores
#SBATCH --time=60:00:00 # set the total run time limit (HH:MM:SS)
#SBATCH --output=/directory_where_the_wsi_are/logs_evg_segm/%a.%j.%N.feat.log # make sure to create the logs_evg_segm folder first
#SBATCH --error=/directory_where_the_wsi_are/logs_evg_segm/%a.%j.%N.feat.errors # make sure to create the logs_evg_segm folder first
#SBATCH --mem=200G # set the memory limit
#SBATCH --mail-type=ALL # select which email types will be sent (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=your_email@domain.com # set destination email


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
# conda  activate /hpc/dhl_ec/VirtualSlides/cglastonbury/wsi
conda activate /hpc/local/Rocky8/dhl_ec/software/mambaforge3/envs/convocals
# conda activate convocals

# edit the following line to set the path to the wsi_preprocessing folder
CONVOCALS_PREPROCESSING_DIR="/hpc/dhl_ec/VirtualSlides/Projects/CONVOCALS/wsi_preprocessing"  

# edit the following lines to set the paths to the h5 files, the slides, and the output folder
H5_DIR="" # with slash at the end
SLIDE_DIR="" # without slash at the end
OUTPUT_DIR="" # with slash at the end

# edit the following line to set the path to the checkpoint
CHECKPOINT="/hpc/dhl_ec/VirtualSlides/Projects/CONVOCALS/AtheroExpressCLAM/checkpoints/checkpoint_ViT_AT.pth"

# edit the following line to run your script for the given paths and stain
python3 $CONVOCALS_PREPROCESSING_DIR/extract_features.py \
-index=$SLURM_ARRAY_TASK_ID \
-num_tasks=$SLURM_ARRAY_TASK_COUNT \
-h5_data=$H5_DIR \
-slide_folder=$SLIDE_DIR \
-features_extraction_checkpoint=$CHECKPOINT \
-output_dir=$OUTPUT_DIR 