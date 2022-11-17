import os
import argparse
import datetime
from PIL import Image
import numpy as np
import openslide
from PIL import ImageDraw
from segmentation_utils import select_random_tiles, get_chunk_AE
from segmentation_utils import slide_to_scaled_pil_image, save_hdf5, get_chunk
from dataset import Whole_Slide_Bag
from torch.utils.data import DataLoader

def get_args_parser():
    parser = argparse.ArgumentParser('segmentation', add_help=False)

    # JOB INDEX
    parser.add_argument('-index', type=str, default=0, help='index of actual job')  
    
    # NUMBER OF TASKS
    parser.add_argument('-num_tasks', type=str, default=1, help='number of tasks')

    # SEGMENTATION OUTPUT DIRECTORY
    parser.add_argument('-output_dir', type=str, default="/hpc/dhl_ec/fcisternino/ATHEROEXPRESS_PROCESSED/", 
                        help='root dir containing all the output results (to be created)')

    # MASKS DIR
    parser.add_argument('-masks_dir', type=str, default="/hpc/dhl_ec/fcisternino/ATHEROEXPRESS_PROCESSED/masks/", 
                        help='directory where the black and white masks will be stored')

    # BATCH SIZE (tiles level)
    parser.add_argument('-batch_size_tiling', type=int, default=512, help='batch size')

    # SAVE THUMBNAILS
    parser.add_argument('-save_thumbnails', default=True , help='whether to save segmentation thumbnails')

    return parser



def tile_slides(args, chunk):
    print('Jobs starting...')
    # === DATA === #
    slides = chunk

    h5_dir = os.path.join(args.output_dir, 'patches_512')
    os.makedirs(h5_dir, exist_ok=True)

    if args.save_thumbnails:
        thumbnails_dir = os.path.join(args.output_dir, 'thumbnails')
        os.makedirs(thumbnails_dir, exist_ok=True)

    # tile size
    tile_size = args.tile_size
    # scale factor
    SCALE = 32
    
    count = 0
    # Iteration over list of slidenames
    for slidename in slides:
        print(slidename, flush = True)
        
        # Open the slide
        slide = openslide.open_slide(slidename)
        # Slide name without extension
        slidename_noext = slidename.split('/')[-1].rsplit('.',1)[0]
        # Tissue
        tissue_name  = slidename.split('/')[-2]
        # Read the mask
        mask = Image.open(os.path.join(args.masks_dir, tissue_name, slidename_noext + '.jpg'))
        mask = np.array(mask)
        # Whole Slide Bag Dataset
        slide_dataset = Whole_Slide_Bag(slide, tile_size = args.tile_size, mask = mask)
        # Tiles Loader
        tiles_loader = DataLoader(dataset= slide_dataset, batch_size=300, num_workers=4)
        # Slidename without file extension
        tissue_name = slidename.split('/')[-2]

        # Downscaled version of the slide 
        downscaled_img, _ = slide_to_scaled_pil_image(slide, SCALE_FACTOR=SCALE)
        draw = ImageDraw.Draw(downscaled_img)
        coords = []
        start = datetime.datetime.now()
        print(slidename_noext)

        mode = 'w'
        for (col, row), res in tiles_loader:

            tissue_indexes = (res[:] > 0.3).nonzero(as_tuple=False)

            for t_idx in list(tissue_indexes):

                coords = np.array([int(col[t_idx]), int(row[t_idx])])
                asset_dict = {'coords': np.array([coords])}
                output_path = os.path.join(h5_dir, slidename_noext + '.coords.h5')
                save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)

                mode = 'a'

                if args.save_thumbnails:
                        s = (int(coords[0]/SCALE), int(coords[1]/SCALE))
                        draw.rectangle(((s[0], s[1]), (s[0] + tile_size/SCALE, s[1] + tile_size/SCALE)), fill=None, outline="green", width=2)
                    

        end = datetime.datetime.now()
        print(f'Time required for slide {slidename}: {end - start}', flush=True)
        if args.save_thumbnails:
            downscaled_img.save(os.path.join(thumbnails_dir,  slidename_noext + '_segm.png'))
        count +=1
        print(f'Tiling performed for {count}/{len(chunk)} slides', flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('segmentation', parents=[get_args_parser()])
    args = parser.parse_args()
    chunk = get_chunk('/group/glastonbury/GTEX-subset/', int(args.index), int(args.num_tasks))
    tile_slides(args, chunk)
