#!/bin/bash
#
#SBATCH --job-name=iph_heatmap # set the name of the job
#SBATCH --partition=cpu # set the partition for the job
#SBATCH --cpus-per-task=2 # set the number of cores
#SBATCH --time=24:00:00 # set the total run time limit (HH:MM:SS)
#SBATCH --output=./logs/heatmap/%a.%j.%N.evg.feat.log # make sure to create the logs_evg_segm folder first
#SBATCH --error=./logs/heatmap/%a.%j.%N.evg.feat.errors # make sure to create the logs_evg_segm folder first
#SBATCH --mem=16G # set the memory limit

echo '----------------------------'
echo ' JOB ID: '$SLURM_ARRAY_JOB_ID
echo ' CURRENT TASK ID: '$SLURM_JOB_ID
echo ' CURRENT TASK NUMBER: '$SLURM_ARRAY_TASK_ID
echo '----------------------------'
echo ' MIN TASK ID: '$SLURM_ARRAY_TASK_MIN
echo ' MAX TASK ID: '$SLURM_ARRAY_TASK_MAX
echo ' TOTAL NUMBER OF TASKS: '$SLURM_ARRAY_TASK_COUNT
echo '----------------------------'

source /home/dhl_ec/tpeters/.bashrc

eval "$(conda shell.bash hook)"
# conda  activate /hpc/dhl_ec/VirtualSlides/cglastonbury/wsi
# conda activate /hpc/local/Rocky8/dhl_ec/software/mambaforge3/envs/convocals
conda activate convocals

# CONVOCALS_DIR="/hpc/dhl_ec/VirtualSlides/Projects/CONVOCALS/AtheroExpressCLAM"
CONVOCALS_DIR="/hpc/dhl_ec/tpeters/projects/AtheroExpressCLAM"
H5_DIR="/hpc/dhl_ec/VirtualSlides/HE/PROCESSED/features_imagenet/h5_files/"
OUTPUT_DIR="/hpc/dhl_ec/VirtualSlides/HE/heatmaps/iph_test/"
CHECKPOINT="/hpc/dhl_ec/fcisternino/CLAM/results/HE_IPH_AtheroExpress_SB_sum_s1/s_7_checkpoint.pt"
CSV_IN="/hpc/dhl_ec/VirtualSlides/HE/20230809.AEDB_HE_IPH_with_path_test.csv"
CSV_OUT="/hpc/dhl_ec/VirtualSlides/HE/heatmaps/IPH_area_samples_test.csv"

python3 $CONVOCALS_DIR/iph.py \
--save_img \
--h5_dir=$H5_DIR \
--csv_in=$CSV_IN \
--csv_out=$CSV_OUT \
--out_dir=$OUTPUT_DIR \
--model_checkpoint=$CHECKPOINT 
