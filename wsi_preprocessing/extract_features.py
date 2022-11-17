import os
import torch
import argparse
import datetime
import numpy as np
import glob
import openslide
import math
from segmentation_utils import get_coords_h5
from segmentation_utils import save_hdf5, extract_multi_view
from model import FeaturesExtraction
from dataset import Tiles_Bag
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import shutil
import os
import h5py


def get_args_parser():
    parser = argparse.ArgumentParser('segmentation', add_help=False)

    # JOB INDEX
    parser.add_argument('-index', type=str, default=0, help='index of actual job')  
    
    # NUMBER OF TASKS
    parser.add_argument('-num_tasks', type=str, default=1, help='number of tasks')
    
    # DATA DIRECTORY
    parser.add_argument('-h5_data', type=str, default="/hpc/dhl_ec/fcisternino/ATHEROEXPRESS_PROCESSED/patches/", 
                                                        help='path to directory containing h5 coordinates files')

    # DATA DIRECTORY
    parser.add_argument('-slide_folder', type=str, default="/hpc/dhl_ec/VirtualSlides/HE/", 
                                                        help='path to directory containing the slides')


    # FEATURES FILES SAVE DIR
    parser.add_argument('-output_dir', type=str, default="/hpc/dhl_ec/fcisternino/ATHEROEXPRESS_PROCESSED/features_512/", help='path to features .h5/.pt storage')


    # FEATURES EXTRACTION CHECKPOINT
    parser.add_argument('-features_extraction_checkpoint', type=str, default="/home/f.cisternino/WSIproj/experiments/dinoExperiments/multiview_1/best_checkpoint.pt",
                    help='Checkpoint for the tiles features extraction model')

    # BATCH SIZE (tiles level)
    parser.add_argument('-batch_size', type=int, default=1,
                        help='batch size (tiles)')
   # TILE SIZE
    parser.add_argument('-tile_size', type=int, default=512,
                        help='batch size (related to the number of tiles that are processed by the network)')

   
    # SAVE FEATURES
    parser.add_argument('-save_features', default=False , help='whether to save segmentation thumbnails')

    # SAVE TILES
    parser.add_argument('-save_tiles', default=True , help='whether to save tiles images')

    return parser




def extract_features(args, chunk):
    print('Jobs starting...')
    # === DATA === #
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # === FEATURES EXTRACTION MODEL === #
    features_extraction_class = FeaturesExtraction(args)
    features_extraction = features_extraction_class.model
    features_extraction.eval()
    features_extraction.to(DEVICE)


    
    os.makedirs(args.output_dir, exist_ok=True)

    temp_dir = os.path.join(args.output_dir, 'temp')
    
    features_output_dir = os.path.join(args.output_dir)
    features_dir_h5 = os.path.join(features_output_dir, 'h5_files')
    features_dir_pt = os.path.join(features_output_dir, 'pt_files')


    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(features_output_dir, exist_ok=True)
    os.makedirs(features_dir_h5, exist_ok=True)
    os.makedirs(features_dir_pt, exist_ok=True)

    # tile size
    tile_size = args.tile_size

    transform = features_extraction_class.transform
    count = 0
    # Iteration over list of slidenames
    for slidename in chunk:
        
        print(f'Slide {chunk.index(slidename) + 1}/{len(chunk)}', flush = True)
        
        start = datetime.datetime.now()

        slidename_noext = slidename.split('/')[-1].rsplit('.', 1)[0]

        if os.path.exists(os.path.join(features_dir_h5, slidename_noext + '.h5')) and os.path.exists(os.path.join(features_dir_pt , slidename_noext + '.pt')):
            continue
        
        # Open the slide
        slide = openslide.open_slide(slidename)
        # Slidename without extension
        # h5 file
        
        h5_file = os.path.join(args.h5_data, slidename_noext + '.coords.h5')
        # print(h5_file)
        coords = get_coords_h5(h5_file)

        dataset = Tiles_Bag(slide=slide, tile_size = args.tile_size, transform=transform, h5=h5_file)
        loader = DataLoader(dataset = dataset, batch_size = 1, num_workers=2)
    
        mode = 'w'
        for tile, coords in loader:
            col, row = coords
      #      img_256, img_512, img_1024 = extract_multi_view(slide, (col, row), tile_size=tile_size)
      #      img_256 = transform(img_256).unsqueeze(0)
      #      img_512 = transform(img_512).unsqueeze(0)
      #      img_1024 = transform(img_1024).unsqueeze(0)
      #      multi_view_input = torch.concat((img_256, img_512, img_1024), dim = 0).unsqueeze(0).to(DEVICE)
            features = features_extraction_class.extractFeatures(tile, device=DEVICE)

     #      cls_token = features[:, 0]
     #       features_v1 = np.mean(features[:, 1: 1+64], axis=1)
     #       features_v2 = np.mean(features[:, 65: 65+64], axis=1)
     #       features_v3 = np.mean(features[:, 129: 129+64], axis=1)
     #       asset_dict = {'features_cls': cls_token, 'features_v1': features_v1,
     #                   'features_v2': features_v2, 'features_v3': features_v3, 
     #                   'coords':  np.array([[int(col), int(row)]])}
            
            asset_dict = {'features': features, 'coords':  np.array([[int(col), int(row)]])}
            output_path = os.path.join(temp_dir, slidename_noext + '.h5')
            save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)

            mode = 'a'
            
        file = h5py.File(output_path, "r")
        
        features = file['features'][:]
        torch.save(torch.from_numpy(np.array(features)), os.path.join(temp_dir , slidename_noext + '.pt'))

        end = datetime.datetime.now()

        print(f'Time required: {end - start}', flush=True)

        shutil.move(os.path.join(temp_dir, slidename_noext + '.h5'), os.path.join(features_dir_h5 , slidename_noext + '.h5'))
        shutil.move(os.path.join(temp_dir, slidename_noext + '.pt'), os.path.join(features_dir_pt , slidename_noext + '.pt'))                            

if __name__ == "__main__":
    parser = argparse.ArgumentParser('features-extraction', parents=[get_args_parser()])
    args = parser.parse_args()
    files = glob.glob(os.path.join(args.slide_folder, '_tif/*.TIF')) + glob.glob(os.path.join(args.slide_folder, '_ndpi/*.ndpi'))

    num_tasks = int(args.num_tasks)
    i = int(args.index)
    print('Number of files:', len(files), flush=True)
    files_per_job = math.ceil(len(files)/num_tasks)
    chunks = [files[x:x+ files_per_job] for x in range(0, len(files), files_per_job )]
    if i < len(chunks):
        chunk = chunks[i]
        print(f'Chunk {i}: {len(chunk)} slides', flush= True)
    else:
        chunk = []
    extract_features(args, chunk)
