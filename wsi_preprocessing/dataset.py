from html.entities import html5
from torch.utils.data import Dataset
import numpy as np
from segmentation_utils import get_coords_h5
import math
import glob
import os

class Whole_Slide_Bag(Dataset):

    def __init__(self, slide, tile_size, mask):
        
        # slide : openslide WSI
        # tile_size : dimension of each squared patch (e.g. 256 or 512)
        # transform : transform function for the tiles classification model
        # fe_transf : transform function for the features extraction model


        
        self.slide = slide
        self.dimensions = slide.dimensions
        self.cols = np.arange(0, int(self.dimensions[0]/tile_size) * tile_size, tile_size)
        self.rows = np.arange(0, int(self.dimensions[1]/tile_size) * tile_size, tile_size)
        self.length = len(self.cols) * len(self.rows)
        self.tile_size = tile_size
        self.mask = mask

        self.scale_factor = round(self.dimensions[0] / self.mask.shape[1])

    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):

        
        row = self.rows[math.floor(idx/len(self.cols))]
        col = self.cols[idx % len(self.cols)]

        tile_row = row // self.scale_factor
        tile_col = col // self.scale_factor
        tile_mask = self.mask[tile_row: tile_row+ self.tile_size // self.scale_factor, tile_col: tile_col + self.tile_size // self.scale_factor]
        res = (tile_mask == 255).sum() / (tile_mask.shape[0]*tile_mask.shape[1])

        return (col, row), res



class Tiles_Bag(Dataset):

    def __init__(self, slide, tile_size, transform, h5):
        
        # slide : openslide WSI
        # tile_size : dimension of each squared patch (e.g. 256 or 512)
        # transform : transform function for the tiles classification model


        
        self.slide = slide
        self.tile_size = tile_size
        self.transform = transform
        self.coords = get_coords_h5(h5)

    
    def __len__(self):
        return self.coords.shape[0]
    
    def __getitem__(self, idx):

        col, row = self.coords[idx]
        img = self.slide.read_region((col, row), 0, (self.tile_size, self.tile_size)).convert('RGB')

        tile = self.transform(img)
        return tile, (col, row)


class WSI_Dataset(Dataset):

    def __init__(self, datafolder):
        
        # datafolder
        
        self.slides = glob.glob(os.path.join(datafolder, '*.svs'))
    
    def __len__(self):
        return len(self.slides)
    
    def __getitem__(self, idx):
        return self.slides[idx]