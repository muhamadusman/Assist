# Check Pytorch installation
import torch, torchvision
from torch import nn
print(torch.__version__, torch.cuda.is_available())
import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import pandas as pd
import nibabel as nib
import imageio
import os

from monai import metrics
from monai.utils.enums import MetricReduction
from monai.metrics import compute_roc_auc

# Check MMSegmentation installation
import mmseg
print(mmseg.__version__)

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

swin_model_file = "Path_to_Proj/assist-vision/data/experiment_runs/work_dirs/swin_base_brats2020_100_001/best.pth"
#device = "cpu"
swin_config_file = "Path_to_Proj/assist-vision/data/experiment_runs/work_dirs/swin_base_brats2021/swin_base_brats2020.py"
vit_config_file = "Path_to_Proj/assist-vision/data/experiment_runs/work_dirs/vit-b16_ln_mln_brats2021/vit-b16_ln_mln_brats2020.py"
vit_model_file = "Path_to_Proj/assist-vision/data/experiment_runs/work_dirs/vit-b16_ln_mln_brats2021/best.pth"
swin_model = init_segmentor(swin_config_file, swin_model_file, device=device)
swin_tr_params = sum(p.numel() for p in swin_model.parameters() if p.requires_grad)
print(f"Swin-Transformer {swin_tr_params}")
vit_model = init_segmentor(vit_config_file, vit_model_file, device=device)
vit_tr_params = sum(p.numel() for p in vit_model.parameters() if p.requires_grad)
print(f"ViT-Transformer {vit_tr_params}")

vit_config_file = "Path_to_Proj/assist-vision/data/experiment_runs/work_dirs/vit-b16_ln_mln_brats2021/vit-b16_ln_mln_brats2021.py"
vit_model_file = "Path_to_Proj/x_muhak/assist-vision/data/experiment_runs/work_dirs/vit-b16_ln_mln_brats2021/best.pth"
vit_model = init_segmentor(vit_config_file, vit_model_file, device=device)
vit_tr_params = sum(p.numel() for p in vit_model.parameters() if p.requires_grad)
print(f"ViT-Transformer {vit_tr_params}")
