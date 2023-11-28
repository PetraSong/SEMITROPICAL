#!/usr/bin/env python3
# -*- coding: utf-8 -*-
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("                                       Plotting HeatMaps: attention maps for whole slide classification")
print("")
print("* Version          : v1.0.0")
print("")
print("* Last update      : 2023-10-03")
print("* Written by       : Francesco Cisternino & Yipei (Petra) Song")
print("* Edite by         : Craig Glastonbury | Sander W. van der Laan | Clint L. Miller | Yipei Song | Francesco Cisternino.")
print("")
print("* Description      : Main modeling.")
print("")
print("                     [1] https://github.com/MaryamHaghighat/PathProfiler")
print("")
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

import argparse
from models.model_clam_sum import CLAM_MB, CLAM_SB
import torch
from utils.eval_utils import initiate_model
from vis_utils.heatmap_utils import slide_to_scaled_pil_image
import h5py
import openslide
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PIL

SCALE_FACTOR = 16 

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for Heatmap Creation')
# data_root_dir => see files organization in README.md
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_mb', 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--model_size', type=str, choices=['small', 'big', 'dino_version','imagenet'], default='imagenet', help='size of model, does not affect mil')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping', 'wsi_classification', 'wsi_classification_binary', 'wsi_classification_binary_eq'], default='wsi_classification_binary_eq')
parser.add_argument('--subtyping', action='store_true', default=True, 
                     help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=1.0,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=96, help='number of positive/negative patches to sample for clam')
parser.add_argument('--drop_out', action='store_true', default=True, help='enabel dropout (p=0.25)')
args = parser.parse_args()
    
args.n_classes = 2


def create_model(args):
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    # Model type
    if args.model_type == 'clam' and args.subtyping:
        model_dict.update({'subtyping': True})
    # Model size
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    # Model type 2: single branch, multiple branch
    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        # B parameter for clustering
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        ## CREATION OF THE MODEL
        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict)
        else:
            raise NotImplementedError
        
    return model

model =  initiate_model(args, '/hpc/dhl_ec/VirtualSlides/SMA/scripts/CONVOCALS-master/AtheroexpressCLAM/results/SMA_IPH_classification_binary_no_inst_k10_balanced_s1/s_0_checkpoint.pt')
model.eval()

f = h5py.File('/hpc/dhl_ec/VirtualSlides/SMA/features_512/h5_files/AE1987.SMA.h5')
coords = f['coords'][:]
features = f['features'][:] 

# model returns logits, Y_prob, Y_hat, A_raw, results_dict
res = model(torch.from_numpy(features))
attention_normalized_scores = res[3][1]


slide = openslide.open_slide('/hpc/dhl_ec/VirtualSlides/SMA/_ndpi/_images/AE1987.SMA.ndpi')
# get downscaled version of the slide
downscaled_img, _ = slide_to_scaled_pil_image(slide, SCALE_FACTOR=SCALE_FACTOR)
downscaled_img.save('./AE1987_raw.png')
# matrix to store logits
logits = np.zeros((downscaled_img.size[1], downscaled_img.size[0]))
# mask to store tissue locations
mask = np.zeros((downscaled_img.size[1], downscaled_img.size[0]))


for coord, p in zip(coords, attention_normalized_scores):
        y, x = map(int, coord)
        x //= SCALE_FACTOR
        y //= SCALE_FACTOR
        xspan, yspan = (slice(x, x+512//SCALE_FACTOR), slice(y, y+512//SCALE_FACTOR))
        logits[xspan, yspan] += p.detach().numpy().item()
        mask[xspan, yspan] += 1


arr_min = np.min(logits[mask >= 1])
arr_max = np.max(logits[mask >= 1])
arr = (logits - arr_min) / (arr_max - arr_min)
# Convert the logits array into heatmap
cmap = plt.get_cmap('coolwarm')
rgba = cmap(arr, bytes=True)

# Convert the 3D array to a PIL image
img = Image.fromarray(rgba, 'RGBA')
mask = mask > 0
alpha = 170 * mask.astype(np.uint8)
img.putalpha(Image.fromarray(alpha, 'L'))
# Overlay heatmap to raw image
overlaid_img = Image.alpha_composite(downscaled_img.convert('RGBA'), img).convert('RGB')

overlaid_img.save('./AE1987_vis.png')

print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("+ The MIT License (MIT)                                                                                               +")
print("+ Copyright (c) 2023 Francesco Cisternino | Craig Glastonbury | Sander W. van der Laan | Clint L. Miller | Yipei Song +")
print("+                                                                                                                     +")
print("+ Permission is hereby granted, free of charge, to any person obtaining a copy of this software and                   +")
print("+ associated documentation files (the \"Software\"), to deal in the Software without restriction, including           +")
print("+ without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell             +")
print("+ copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the            +")
print("+ following conditions:                                                                                               +")
print("+                                                                                                                     +")
print("+ The above copyright notice and this permission notice shall be included in all copies or substantial                +")
print("+ portions of the Software.                                                                                           +")
print("+                                                                                                                     +")
print("+ THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT             +")
print("+ LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO           +")
print("+ EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER           +")
print("+ IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR             +")
print("+ THE USE OR OTHER DEALINGS IN THE SOFTWARE.                                                                          +")
print("+                                                                                                                     +")
print("+ Reference: http://opensource.org.                                                                                   +")
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")