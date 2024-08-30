import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import os
import pandas as pd
from utils.utils import *
import PIL
from PIL import Image
from math import floor
import matplotlib.pyplot as plt
from datasets.wsi_dataset import Wsi_Region
import h5py
from wsi_core.WholeSlideImage import WholeSlideImage
from scipy.stats import percentileofscore
import math
from utils.file_utils import save_hdf5
from scipy.stats import percentileofscore

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def score2percentile(score, ref):
    percentile = percentileofscore(ref, score)
    return percentile

def drawHeatmap(scores, coords, slide_path=None, wsi_object=None, vis_level = -1, **kwargs):
    if wsi_object is None:
        wsi_object = WholeSlideImage(slide_path)
        print(wsi_object.name)
    
    wsi = wsi_object.getOpenSlide()
    if vis_level < 0:
        vis_level = wsi.get_best_level_for_downsample(32)
    
    heatmap = wsi_object.visHeatmap(scores=scores, coords=coords, vis_level=vis_level, **kwargs)
    return heatmap

def initialize_wsi(wsi_path, seg_mask_path=None, seg_params=None, filter_params=None):
    wsi_object = WholeSlideImage(wsi_path)
    if seg_params['seg_level'] < 0:
        best_level = wsi_object.wsi.get_best_level_for_downsample(32)
        seg_params['seg_level'] = best_level

    wsi_object.segmentTissue(**seg_params, filter_params=filter_params)
    wsi_object.saveSegmentation(seg_mask_path)
    return wsi_object

def compute_from_patches(wsi_object, features, coords, clam_pred=None, model=None, feature_extractor=None, batch_size=512,  
    attn_save_path=None, ref_scores=None, feat_save_path=None, **wsi_kwargs):    
    top_left = wsi_kwargs['top_left']
    bot_right = wsi_kwargs['bot_right']
    patch_size = wsi_kwargs['patch_size']
    
#    roi_dataset = Wsi_Region(wsi_object, **wsi_kwargs)
#    roi_loader = get_simple_loader(roi_dataset, batch_size=batch_size, num_workers=8)
#    print('total number of patches to process: ', len(roi_dataset))
#    num_batches = len(roi_loader)
#    print('number of batches: ', len(roi_loader))
#    mode = "w"
#    breakpoint()

    mode='w'

    if attn_save_path is not None:
        A = model(features, attention_only=True)
           
        if A.size(0) > 1: #CLAM multi-branch attention
            A = A[clam_pred]

        A = A.view(-1, 1).cpu().detach().numpy()

        if ref_scores is not None:
            for score_idx in range(len(A)):
                A[score_idx] = score2percentile(A[score_idx], ref_scores)

        asset_dict = {'attention_scores': A, 'coords': coords}
        save_path = save_hdf5(attn_save_path, asset_dict, mode=mode)
    

    if feat_save_path is not None:
        asset_dict = {'features': features.cpu().numpy(), 'coords': coords}
        save_hdf5(feat_save_path, asset_dict, mode=mode)

        mode = "a"
    return attn_save_path, feat_save_path, wsi_object

def slide_to_scaled_pil_image(slide, SCALE_FACTOR=32):
    """
    Convert a WSI slide to a scaled-down PIL image.
    Args:
        slide: An OpenSlide object.
    Returns:
        Tuple consisting of scaled-down PIL image, original width, original height, new width, and new height.
    """
    large_w, large_h = slide.dimensions
    new_w = math.floor(large_w / SCALE_FACTOR)
    new_h = math.floor(large_h / SCALE_FACTOR)
    level = slide.get_best_level_for_downsample(SCALE_FACTOR)
    whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
    whole_slide_image = whole_slide_image.convert("RGB")
    img = whole_slide_image.resize((new_w, new_h), PIL.Image.BILINEAR)
    return img, (large_w, large_h, new_w, new_h)