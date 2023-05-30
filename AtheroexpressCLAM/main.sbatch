#!/bin/bash
#SBATCH --job-name=vit
#SBATCH --partition=gpu
#SBATCH --array=0-0
#SBATCH --cpus-per-task=2
#SBATCH --time=30:00:00
#SBATCH --output=./main_sum.log
#SBATCH --mem=20000M
'# --gres=gpu:1'


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

CUDA_VISIBLE_DEVICES=0,1 python main.py --drop_out --early_stopping --lr 1e-4 --k 1 --label_frac 1.0 --exp_code atheroexpress_classification_binary --model_size dino_version --bag_loss ce --task wsi_classification_binary --model_type clam_mb --log_data --subtyping --data_root_dir /hpc/dhl_ec/fcisternino/ATHEROEXPRESS_PROCESSED/features_imagenet --no_inst_cluster --weighted_sample