import subprocess
import glob
import pandas as pd
import os
import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score

Image.MAX_IMAGE_PIXELS = None

# Directories and paths
annot_dir = "/hpc/dhl_ec/VirtualSlides/ANNOTATIONS/Annotations/export_v2/"
heatmap_dir = "/hpc/dhl_ec/VirtualSlides/ANNOTATIONS/Annotations/heatmap/iph/"
arr_dir = "/hpc/dhl_ec/VirtualSlides/ANNOTATIONS/Annotations/heatmap/iph/"  # Assuming .npy arrays are stored here
raw_dir = heatmap_dir  # Assuming raw images are in the same directory as heatmaps
overlay_dir = "/hpc/dhl_ec/VirtualSlides/ANNOTATIONS/Annotations/heatmap/overlay_iph_new/"
resized_annot_dir = "/hpc/dhl_ec/VirtualSlides/ANNOTATIONS/Annotations/heatmap/resized_annots_new/" 
overlay_heatmap_dir = "/hpc/dhl_ec/VirtualSlides/ANNOTATIONS/Annotations/heatmap/overlay_heatmap_new/"
csv_path = "/hpc/dhl_ec/VirtualSlides/ANNOTATIONS/Annotations/heatmap/analysis_results.csv"

# Ensure the directories exist
os.makedirs(overlay_dir, exist_ok=True)
os.makedirs(resized_annot_dir, exist_ok=True)
os.makedirs(overlay_heatmap_dir, exist_ok=True)

# Function to resize images using PIL
def resize_image_with_pillow(input_path, output_path, target_width, target_height):
    with Image.open(input_path) as img:
        resized_img = img.resize((target_width, target_height), Image.ANTIALIAS)
        resized_img.save(output_path)

# Function to calculate metrics, now using binary masks
def calculate_metrics(binary_mask, expert_binary_mask):
    """
    Calculate sensitivity, specificity, ROC AUC, and IoU.
    
    :param binary_mask: Numpy array of binary predicted labels.
    :param expert_binary_mask: Numpy array of binary ground truth labels (expert annotations).
    :return: Dictionary containing calculated metrics and IoU.
    """
    # Normalize binary masks to have values in {0, 1}
    binary_mask = (binary_mask>0).astype(int)
    expert_binary_mask = (expert_binary_mask>0).astype(int)

    # Flatten arrays to convert them into 1D arrays for metric calculation
    binary_mask_flat = binary_mask.flatten()
    expert_binary_mask_flat = expert_binary_mask.flatten()

    # Check if there are both classes present
    if len(np.unique(expert_binary_mask_flat)) == 1:
        # Only one class present, cannot calculate ROC AUC
        roc_auc = float('nan')
        sensitivity = float('nan')
        specificity = float('nan')
        iou = float('nan')
    else:
        # Calculate ROC AUC
        roc_auc = roc_auc_score(expert_binary_mask_flat, binary_mask_flat)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(expert_binary_mask_flat, binary_mask_flat).ravel()

        # Calculate sensitivity and specificity
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        # Calculate Intersection over Union (IoU)
        intersection = np.logical_and(binary_mask, expert_binary_mask).sum()
        union = np.logical_or(binary_mask, expert_binary_mask).sum()
        iou = intersection / union if union > 0 else 0.0
    
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ROC_AUC": roc_auc,
        "IoU": iou
    }

# Function to overlay masks on raw images
def overlay_masks_on_raw(raw_path, binary_mask, output_path):
    # Read the raw image
    raw_img = cv2.imread(raw_path)
    
    # Convert binary mask to boolean
    mask_bool = binary_mask.astype(bool)
    
    # Convert the raw image to HSV (Hue, Saturation, Value) color space
    raw_hsv = cv2.cvtColor(raw_img, cv2.COLOR_BGR2HSV)
    
    # Set the hue to 80 (greenish hue) and saturation to 255 (full color saturation) where mask is true
    raw_hsv[mask_bool, 0] = 80  # HUE
    raw_hsv[mask_bool, 1] = 255  # SATURATION
    
    # Convert back to BGR color space
    overlay_img_bgr = cv2.cvtColor(raw_hsv, cv2.COLOR_HSV2BGR)

    # Save the overlay image
    cv2.imwrite(output_path, overlay_img_bgr)

# Function to overlay masks on heatmaps with pseudo-color
def overlay_masks_on_heatmap(heatmap_path, binary_mask, output_path):
    heatmap_img = cv2.imread(heatmap_path)
    
    # Convert binary mask to boolean
    mask_bool = binary_mask.astype(bool)
    
    # Convert the heatmap to HSV color space
    heatmap_hsv = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2HSV)
    
    # Set the hue to 80 (greenish hue) and saturation to 255 (full color saturation) where mask is true
    heatmap_hsv[mask_bool, 0] = 80  # HUE
    heatmap_hsv[mask_bool, 1] = 255  # SATURATION
    
    # Convert back to BGR color space
    overlay_img_bgr = cv2.cvtColor(heatmap_hsv, cv2.COLOR_HSV2BGR)

    cv2.imwrite(output_path, overlay_img_bgr)

# Initialize dataframe for results
df_results = pd.DataFrame(columns=["case_id", "sensitivity", "specificity", "ROC_AUC", "IoU", "annotation_overlay_iph_path", "annotation_overlay_heatmap_path"])

# Iterate through heatmap directory to process each case
for heatmap_path in glob.glob(heatmap_dir + "*_overlay.png"):
    case_id = os.path.basename(heatmap_path).split('.')[0].replace("_test", "")
    annot_path = os.path.join(annot_dir, f"{case_id}.HE_iph_label.png")
    arr_path = os.path.join(arr_dir, f"{case_id}.HE_iph_binary_prediction.npy")
    raw_path = os.path.join(raw_dir, f"{case_id}.HE_raw.png")
    
    if not (os.path.exists(annot_path) and os.path.exists(arr_path)):
        continue  # Skip if corresponding annotation or .npy array does not exist
    
    # Load binary mask from .npy file
    binary_mask = np.load(arr_path)
    print("predicted binary mask shape:", binary_mask.shape)
    
    # Resize annotations using Pillow and load the resized expert binary mask
    heatmap = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)
    print("predicted heatmap shape:", heatmap.shape)

    target_height, target_width = binary_mask.shape[:2]
    resized_annot_path = os.path.join(resized_annot_dir, f"{case_id}.HE_iph_label_resized.png")
    resize_image_with_pillow(annot_path, resized_annot_path, target_width, target_height)
    resized_expert_binary_mask = cv2.imread(resized_annot_path, cv2.IMREAD_GRAYSCALE)

    # Skip steps if the expert binary mask is pure black
    if np.sum(resized_expert_binary_mask) == 0:
        print(f"Skipping {case_id} due to pure black mask.")
        continue

    # Calculate metrics including IoU
    metrics = calculate_metrics(binary_mask, resized_expert_binary_mask)

    # Overlay on raw and heatmap
    overlay_iph_path = os.path.join(overlay_dir, f"{case_id}.HE_overlay_iph.png")
    overlay_heatmap_path = os.path.join(overlay_heatmap_dir, f"{case_id}.HE_overlay_heatmap.png")
    overlay_masks_on_raw(raw_path, resized_expert_binary_mask, overlay_iph_path)
    overlay_masks_on_heatmap(heatmap_path, resized_expert_binary_mask, overlay_heatmap_path)

    # Append results to dataframe
    df_results = df_results.append({
        "case_id": case_id,
        "sensitivity": metrics["sensitivity"] if not np.isnan(metrics["sensitivity"]) else "N/A",
        "specificity": metrics["specificity"] if not np.isnan(metrics["specificity"]) else "N/A",
        "ROC_AUC": metrics["ROC_AUC"] if not np.isnan(metrics["ROC_AUC"]) else "N/A",
        "IoU": metrics["IoU"] if not np.isnan(metrics["IoU"]) else "N/A",
        "annotation_overlay_iph_path": overlay_iph_path,
        "annotation_overlay_heatmap_path": overlay_heatmap_path
    }, ignore_index=True)

# Save results to CSV
df_results.to_csv(csv_path, index=False)

print("Analysis complete. Results saved to:", csv_path)
