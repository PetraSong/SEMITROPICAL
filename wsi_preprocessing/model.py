import torch
import numpy as np
from torchvision.transforms import transforms
import timm
import torch.nn as nn
import dino_model.vision_transformer as vits
import torchvision.models as models
from torchvision.models import convnext_base, ConvNeXt_Base_Weights



class FeaturesExtraction:

    def __init__(self, args):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            #transforms.Normalize((0.8035626,  0.50688577, 0.63166803), (0.14852792, 0.20197499, 0.16831921)),
        ])
        state_dict = torch.load(args.features_extraction_checkpoint, map_location="cpu")
        args.checkpoint_key = 'teacher'  # one of 'student' or 'teacher'
        args.arch = 'vit_base'
        args.patch_size = 16
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict {args.features_extraction_checkpoint}", flush = True)
            state_dict = state_dict[args.checkpoint_key]

        self.model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = self.model.load_state_dict(state_dict, strict=False)
        for p in self.model.parameters():
            p.requires_grad = False
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.features_extraction_checkpoint, msg), flush = True)
        self.model.eval()


    def extractFeatures(self, img, device):
        with torch.no_grad():
            feats = self.model(img.to(device)).clone()
            feats = feats.cpu().detach().numpy()
        return feats

class FeaturesExtraction_IMAGENET:

    def __init__(self, args):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.model = convnext_base(weights = ConvNeXt_Base_Weights.IMAGENET1K_V1)
        self.model.eval()


    def extractFeatures(self, img, device):
        with torch.no_grad():
            feats = self.model(img.to(device)).clone()
            feats = feats.cpu().detach().numpy()
        return feats