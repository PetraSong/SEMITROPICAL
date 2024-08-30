
import argparse
from models.model_clam_sum import CLAM_MB, CLAM_SB
import torch
from utils.eval_utils import initiate_model
from vis_utils.heatmap_utils import slide_to_scaled_pil_image
from sklearn import preprocessing as pre
import h5py
import openslide
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import pandas as pd
from PIL import ImageDraw, ImageFont

SCALE_FACTOR = 16 

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for Heatmap Creation')
# data_root_dir => see files organization in README.md
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb', 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--model_size', type=str, choices=['small', 'big', 'dino_version'], default='dino_version', help='size of model, does not affect mil')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping', 'wsi_classification', 'wsi_classification_binary', 'wsi_classification_binary_symptoms'], default='wsi_classification_binary')
parser.add_argument('--subtyping', action='store_true', default=True, 
                     help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=1.0,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=96, help='number of positive/negative patches to sample for clam')
parser.add_argument('--drop_out', action='store_true', default=True, help='enabel dropout (p=0.25)')
args = parser.parse_args()
    
args.n_classes = 2

SAVE_IMAGES = False

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

# Normalize between -1 and 1
def normalize_array(array):
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = -1 + 2 * (array - min_val) / (max_val - min_val)
    return normalized_array

def compute_iph(case_id, filename, out_df, gt_label):
    f = h5py.File(f'/hpc/dhl_ec/VirtualSlides/HE/PROCESSED/features_imagenet/h5_files/{case_id}.h5')
    coords = f['coords'][:]
    features = f['features'][:] 

    with torch.no_grad():
        # model returns logits, Y_prob, Y_hat, A_raw, results_dict
        res = model(torch.from_numpy(features))

    attention_scores = res[3][0].cpu().numpy()
    contribution_array = res[5][:,0].cpu().numpy()

    print(f"mean: {np.median(contribution_array)} -- min: {np.min(contribution_array)} -- max: {np.max(contribution_array)}")

    # Clip contribution array at positive and negative percentiles
    clipped_contribution_positive = np.percentile(contribution_array, 98)
    # clipped_contribution_negative = np.percentile(contribution_array, 1)
    
    contribution_array_clipped = np.clip(contribution_array, a_min=None, a_max=clipped_contribution_positive)
    # contribution_array_clipped = contribution_array

    # Normalize arrays
    attention_scores = (attention_scores - np.min(attention_scores)) / (np.max(attention_scores) - np.min(attention_scores))
    # attention_scores = normalize_array(attention_scores)
    # attention_scores = pre.MinMaxScaler((-1, 1)).fit_transform(attention_scores.reshape(-1, 1))[:,0]
    contribution_norm = (contribution_array_clipped - np.min(contribution_array_clipped)) / (np.max(contribution_array_clipped) - np.min(contribution_array_clipped))
    # contribution_norm = normalize_array(contribution_array_clipped)
    # contribution_norm = pre.MinMaxScaler((-1, 1)).fit_transform(contribution_array_clipped.reshape(-1, 1))[:,0]

    # Calculate combined importance scores
    importance_scores = attention_scores * contribution_norm
    importance_scores = (importance_scores - np.min(importance_scores)) / (np.max(importance_scores) - np.min(importance_scores))
    # importance_scores = normalize_array(importance_scores)
    # importance_scores = pre.MinMaxScaler((-1, 1)).fit_transform(importance_scores.reshape(-1, 1))[:,0]


    # attention_scores = pre.MinMaxScaler().fit_transform(attention_scores.reshape(-1, 1))[:,0]
    # attention_scores = torch.from_numpy(attention_scores)
    # atten_test = pre.MinMaxScaler().fit_transform(atten_test.reshape(-1, 1))[:,0]
    # attention_scores = (attention_scores - np.min(attention_scores)) / (np.max(attention_scores) - np.min(attention_scores))
    # attention_scores = torch.from_numpy(attention_scores)
    # atten_test = np.sqrt((atten_test - np.min(atten_test)) / (np.max(atten_test) - np.min(atten_test)))
    
    # adjusted_attn = torch.clone(attention_scores) * atten_test
    # adjusted_attn = torch.clone(attention_scores).cpu().numpy()
    # arr_min = np.min(adjusted_attn)
    # arr_max = np.max(adjusted_attn)
    # adjusted_attn = (adjusted_attn - arr_min) / (arr_max - arr_min)
    # adjusted_attn *= atten_test
    # adjusted_attn = torch.from_numpy(adjusted_attn)
    # adjusted_attn[atten_test <= 0] *= -1
    print(res[3].shape)
    print(res[5].shape)
    print(res[5][:,0].shape)
    print(f'Probability of IPH: {round(res[1][0].item(), 4)}')
    # area = (attention_scores > 0).sum() / attention_scores.shape[0]
    # area_test = (importance_scores > 0).sum() / importance_scores.shape[0]
    area = (attention_scores > 0.5).sum() / attention_scores.shape[0]
    area_test = (importance_scores > 0.5).sum() / importance_scores.shape[0]
    print(f'Proxy IPH area: {area}')
    print(f'Proxy IPH area (test): {area_test}')
    out_df = out_df.append({'case_id': case_id, 'IPH': (res[1][0] > 0.5).cpu().numpy(), 'iph_prob': round(res[1][0].item(), 4), 'gt': gt_label == 'yes', 
                            # 'area': ((attention_scores > 0).sum() / attention_scores.shape[0]), 
                            'area': ((attention_scores > 0.5).sum() / attention_scores.shape[0]),
                            'area_mean': np.median(attention_scores), 
                            # 'min': np.min(attention_scores), 
                            # 'max': np.max(attention_scores),
                            # 'area (test)': ((importance_scores > 0).sum() / importance_scores.shape[0]),
                            'area (test)': ((importance_scores > 0.5).sum() / importance_scores.shape[0]),
                            'area_mean (test)': np.median(importance_scores), 
                            # 'min (test)': np.min(importance_scores), 
                            # 'max (test)': np.max(importance_scores),
                            'coords_amount': len(coords),
                            }, ignore_index=True)
    # out_df.to_csv('./IPH_area_samples_contrib_90.csv')
    # out_df.to_csv('./IPH_area_samples_contrib.csv')
    # out_df.to_csv('./IPH_area_samples_contrib_full.csv')
    out_df.to_csv('./IPH_area_samples_contrib_testr.csv')

    if SAVE_IMAGES:
        #out_df.to_csv('IPH_area_samples.csv')
        slide = openslide.open_slide(filename)
        # get downscaled version of the slide
        downscaled_img, _ = slide_to_scaled_pil_image(slide, SCALE_FACTOR=SCALE_FACTOR)
        downscaled_img.save(f'./heatmaps/contrib_test/{case_id}_raw.png')
        # matrix to store logits
        logits = np.zeros((downscaled_img.size[1], downscaled_img.size[0]))
        # mask to store tissue locations
        mask = np.zeros((downscaled_img.size[1], downscaled_img.size[0]))


        for coord, p in zip(coords, attention_scores):
                y, x = map(int, coord)
                x //= SCALE_FACTOR
                y //= SCALE_FACTOR
                xspan, yspan = (slice(x, x+512//SCALE_FACTOR), slice(y, y+512//SCALE_FACTOR))
                logits[xspan, yspan] += p.item()
                mask[xspan, yspan] += 1

        logits /= (mask + 1e-9)
        arr_min = np.min(logits[mask >= 1])
        arr_max = np.max(logits[mask >= 1])
        print(f'max: {arr_max}')
        print(f'min: {arr_min}')
        arr = (logits - arr_min) / (arr_max - arr_min)
        # arr = pre.MinMaxScaler((-1, 1)).fit_transform(logits)
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
        overlaid_img.save(f'./heatmaps/contrib_test/{case_id}_heat.png')

        # get downscaled version of the slide
        downscaled_img, _ = slide_to_scaled_pil_image(slide, SCALE_FACTOR=SCALE_FACTOR)
        # matrix to store logits
        logits = np.zeros((downscaled_img.size[1], downscaled_img.size[0]))
        # mask to store tissue locations
        mask = np.zeros((downscaled_img.size[1], downscaled_img.size[0]))

        for coord, p in zip(coords, importance_scores):
                y, x = map(int, coord)
                x //= SCALE_FACTOR
                y //= SCALE_FACTOR
                xspan, yspan = (slice(x, x+512//SCALE_FACTOR), slice(y, y+512//SCALE_FACTOR))
                logits[xspan, yspan] += p.item()
                mask[xspan, yspan] += 1

        logits /= (mask + 1e-9)
        arr_min = np.min(logits[mask >= 1])
        arr_max = np.max(logits[mask >= 1])

        print(f'max (test): {arr_max}')
        print(f'min (test): {arr_min}')

        arr = (logits - arr_min) / (arr_max - arr_min)
        # arr = pre.MinMaxScaler((-1, 1)).fit_transform(logits)
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
        overlaid_img.save(f'./heatmaps/contrib_test/{case_id}_heat_test.png')
    return out_df



model =  initiate_model(args, '/hpc/dhl_ec/fcisternino/CLAM/results/HE_IPH_AtheroExpress_SB_sum_s1/s_0_checkpoint.pt')
model.eval()


out_df = pd.DataFrame()

df = pd.read_csv('/hpc/dhl_ec/tpeters/projects/AtheroExpressCLAM/AtheroExpress_HE_WSI_dataset_IPH_with_path_sub.csv')
# df = pd.read_csv('/hpc/dhl_ec/tpeters/projects/AtheroExpressCLAM/AtheroExpress_HE_WSI_dataset_IPH_with_path.csv')
# df = pd.read_csv('/hpc/dhl_ec/tpeters/projects/AtheroExpressCLAM/AtheroExpress_HE_WSI_dataset_IPH_with_path_test.csv')
#df = pd.read_csv('samples.txt', sep = ' ', header=None)
for idx, line in df.iterrows():
    case_id, sample_id, filename, gt_label = line
    #sample_id, filename = line
    print(sample_id)
    out_df = compute_iph(sample_id, filename, out_df, gt_label)


# out_df.to_csv('HE_samples_IPH_sum.csv')