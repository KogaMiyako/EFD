from skimage.metrics import peak_signal_noise_ratio as F_psnr
from skimage.metrics import structural_similarity as F_ssim
import lpips
import torch
import numpy as np
from skimage import io
import glob

# path
SR_paths = glob.glob('./checkpoints/diffsr_div2k_rrdb32_dot_senet/results_0_Urban100/SR/*.png')
HR_paths = glob.glob('./checkpoints/diffsr_div2k_rrdb32_dot_senet/results_0_Urban100/HR/*.png')

n_samples = len(SR_paths)
count = 1
_psnr = 0.0

for i in range(n_samples):

    print("cur psnr: ")
