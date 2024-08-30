import subprocess
import glob
import pandas as pd
import os
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc


# Directories and paths
annot_dir = "/hpc/dhl_ec/VirtualSlides/ANNOTATIONS/Annotations/export/"
heatmap_dir = "/hpc/dhl_ec/VirtualSlides/ANNOTATIONS/Annotations/heatmap/iph/"
arr_dir = "/hpc/dhl_ec/VirtualSlides/ANNOTATIONS/Annotations/heatmap/iph/"  # Assuming .npy arrays are stored here
raw_dir = heatmap_dir  # Assuming raw images are in the same directory as heatmaps
overlay_dir = "/hpc/dhl_ec/VirtualSlides/ANNOTATIONS/Annotations/heatmap/overlay_iph/"
resized_annot_dir = "/hpc/dhl_ec/VirtualSlides/ANNOTATIONS/Annotations/heatmap/resized_annots/" 
overlay_heatmap_dir = "/hpc/dhl_ec/VirtualSlides/ANNOTATIONS/Annotations/heatmap/overlay_heatmap/"
csv_path = "/hpc/dhl_ec/VirtualSlides/ANNOTATIONS/Annotations/heatmap/analysis_results.csv"

# Ensure the directories exist
os.makedirs(overlay_dir, exist_ok=True)
os.makedirs(resized_annot_dir, exist_ok=True)
os.makedirs(overlay_heatmap_dir, exist_ok=True)

# Function to resize images using ImageMagick
def resize_image_with_imagemagick(input_path, output_path, target_width, target_height):
    cmd = ['magick', 'convert', input_path, '-resize', f'{target_width}x{target_height}!', output_path]
    subprocess.run(cmd, check=True)

# Function to calculate metrics (might need adjustments based on .npy data)
def calculate_metrics(binary_mask, predictions):
    """
    Calculate sensitivity, specificity, ROC AUC, and find the optimal threshold.
    
    :param binary_mask: Numpy array of ground truth binary labels.
    :param predictions: Numpy array of predicted probabilities.
    :return: Dictionary containing calculated metrics and the optimal threshold.
    """
    # Flatten arrays to convert them into 1D arrays for metric calculation
    binary_mask_flat = binary_mask.flatten()
    predictions_flat = predictions.flatten()
    
    # Calculate ROC AUC
    roc_auc = roc_auc_score(binary_mask_flat, predictions_flat)
    
    # Calculate ROC curve and find the optimal threshold
    fpr, tpr, thresholds = roc_curve(binary_mask_flat, predictions_flat)
    optimal_idx = np.argmax(tpr - fpr)  # Maximize sensitivity (TPR) - FPR
    optimal_threshold = thresholds[optimal_idx]
    
    # Apply the optimal threshold to predictions to get binary output
    optimal_predictions = (predictions_flat >= optimal_threshold).astype(int)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(binary_mask_flat, optimal_predictions).ravel()
    
    # Calculate sensitivity and specificity
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ROC_AUC": roc_auc,
        "optimal_threshold": optimal_threshold
    }

# Function to overlay masks on raw images (remains unchanged)
def overlay_masks_on_raw(raw_path, binary_mask, output_path):
    # Read the raw image
    raw_img = cv2.imread(raw_path)
    
    # Convert the binary mask to a 3-channel image for overlay
    mask_colored = cv2.applyColorMap(cv2.convertScaleAbs(binary_mask, alpha=(255.0/255.0)), cv2.COLORMAP_JET)
    
    # Create the overlay image
    overlay_img = cv2.addWeighted(raw_img, 0.6, mask_colored, 0.4, 0)
    
    # Save the overlay image
    cv2.imwrite(output_path, overlay_img)

# New function to overlay masks on heatmaps with pseudo-color
def overlay_masks_on_heatmap(heatmap_path, binary_mask, output_path):
    heatmap_img = cv2.imread(heatmap_path)
    mask_colored = cv2.applyColorMap(binary_mask * 255, cv2.COLORMAP_JET)
    overlay_img = cv2.addWeighted(heatmap_img, 0.7, mask_colored, 0.3, 0)
    cv2.imwrite(output_path, overlay_img)


# New function to overlay enhanced information
def overlay_enhanced_information(heatmap_path, arr, binary_mask, output_path):
    # Example function to overlay additional information from arr on heatmap
    # This is a placeholder for how you might implement such a function

# Initialize dataframe for results
df_results = pd.DataFrame(columns=["case_id", "sensitivity", "specificity", "ROC_AUC", "optimal_threshold", "annotation_overlay_iph_path", "annotation_overlay_heatmap_path"])

# Iterate through heatmap directory to process each case
for heatmap_path in glob.glob(heatmap_dir + "*_test.png"):
    case_id = os.path.basename(heatmap_path).split('.')[0].replace("_test", "")
    annot_path = os.path.join(annot_dir, f"{case_id}.HE_iph_label.png")
    arr_path = os.path.join(arr_dir, f"{case_id}.HE_arr.npy")
    raw_path = os.path.join(raw_dir, f"{case_id}.HE_raw.png")
    
    if not (os.path.exists(annot_path) and os.path.exists(arr_path)):
        continue  # Skip if corresponding annotation or .npy array does not exist
    
    # Load .npy array
    arr = np.load(arr_path)
    
    # Resize annotations using ImageMagick and load the resized binary mask
    heatmap = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)
    target_height, target_width = heatmap.shape[:2]
    resized_annot_path = os.path.join(resized_annot_dir, f"{case_id}.HE_iph_label_resized.png")
    resize_image_with_imagemagick(annot_path, resized_annot_path, target_width, target_height)
    resized_binary_mask = cv2.imread(resized_annot_path, cv2.IMREAD_GRAYSCALE)
    
    # Skip steps if the binary mask is pure black
    if np.sum(resized_binary_mask) == 0:
        print(f"Skipping {case_id} due to pure black mask.")
        continue

    enhanced_metric = calculate_metrics(resized_binary_mask, arr)

    # Overlay on raw and heatmap
    overlay_iph_path = os.path.join(overlay_dir, f"{case_id}.HE_overlay_iph.png")
    overlay_heatmap_path = os.path.join(overlay_heatmap_dir, f"{case_id}.HE_overlay_heatmap.png")
    overlay_masks_on_raw(raw_path, resized_binary_mask, overlay_iph_path)
    overlay_masks_on_heatmap(heatmap_path, resized_binary_mask, overlay_heatmap_path)

    # Append results to dataframe
    df_results = df_results.append({
        "case_id": case_id,
        "sensitivity": metrics["sensitivity"],
        "specificity": metrics["specificity"],
        "ROC_AUC": metrics["ROC_AUC"],
        "optimal_threshold": metrics["optimal_threshold"],
        "annotation_overlay_iph_path": overlay_iph_path,
        "annotation_overlay_heatmap_path": overlay_heatmap_path
    }, ignore_index=True)
# Save results to CSV
df_results.to_csv(csv_path, index=False)

print("Analysis complete. Results saved to:", csv_path)
