**Multiple Instance Learning for AtheroExpress histopathology samples**

This repository allows to train Multiple Instance Learning models as the one in CLAM paper (https://arxiv.org/abs/2004.09666)

Before running it, it is necessary to:
- have the csv file for the train/val/test splits as in the _splits_ folder
- store the features of every sample in a .h5 or .pt file


Training run command:

CUDA_VISIBLE_DEVICES=0,1 python main.py --drop_out --early_stopping --lr 1e-4 --k 1 --label_frac 1.0 --exp_code atheroexpress_classification_code --model_size dino_version --bag_loss ce --task wsi_classification_binary --model_type clam_mb --log_data --subtyping --data_root_dir /path/to/features

