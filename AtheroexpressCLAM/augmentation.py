from openslide import open_slide
import h5py
import torch
import numpy as np
import random
# k = 800

# slide = open_slide ('/hpc/dhl_ec/fcisternino/CLAM/heatmaps/demo/slides/AE490.UMC.HE.ndpi')
# h5_path = '/hpc/dhl_ec/fcisternino/CLAM/heatmaps/heatmap_raw_results/HEATMAP_OUTPUT_BINARY/Symptomatic/AE490.UMC.HE/AE490.UMC.HE.h5'
# file = h5py.File(h5_path, "r")
# features = torch.tensor(file['features'][:])
# coords = torch.tensor(file['coords'][:])

# f = features[k]
# c = coords[k]
# norm = torch.norm(features - f.unsqueeze(0), dim=1)
# val, min_idx = torch.kthvalue(norm, 2)

# qc = coords[int(min_idx)]
# img1 = slide.read_region((int(c[0]), int(c[1])), 0, (512, 512)).convert('RGB')
# img1.save('./patch_query.jpg')
# img2 = slide.read_region((int(qc[0]), int(qc[1])), 0, (512, 512)).convert('RGB')
# img2.save('./patch_retrieved.jpg')

def drop_features(tensor, indices):
    mask = torch.ones(tensor.shape, dtype=torch.bool)
    mask[indices] = False
    return tensor[mask].reshape(-1, 1000)

def bag_augmentation(features):

    choices = ['no_aug', 'drop', 'replace', 'interpolate']
    drop_idx = []

    for i in range(features.shape[0]):
        
        features_i = features[i]

        choice = random.choices(choices, weights = [0.6, 0.15, 0.13, 0.12])[0]
     #   print(choice)

        if choice == 'no_aug':
            continue

        elif choice == 'drop':
            drop_idx.append(i)
        
        norm = torch.norm(features - features_i.unsqueeze(0), dim=1)
        val, min_idx = torch.kthvalue(norm, 2)
        features_closest_i = features[int(min_idx)]

        if choice == 'interpolate':
            l = round(random.random(), 1)
            features[i] = l * features_i + (1-l) * features_closest_i
        
        elif choice == 'replace':
            features[i] = features_closest_i

    features = drop_features(features, drop_idx)
    return features

        


# h5_path = '/hpc/dhl_ec/fcisternino/CLAM/heatmaps/heatmap_raw_results/HEATMAP_OUTPUT_BINARY/Symptomatic/AE490.UMC.HE/AE490.UMC.HE.h5'
# file = h5py.File(h5_path, "r")
# features = torch.tensor(file['features'][:])
# res = bag_augmentation(features)

# print(res.shape)