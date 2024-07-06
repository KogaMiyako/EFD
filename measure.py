import glob
import torchvision.transforms as transforms
from PIL import Image
import torch
import numpy as np
from utils.utils import Measure

# trans = transforms.ToTensor()
# Ms = Measure()
# # _psnr = 0.0
# _ssim = 0.0
# _lpips = 0.0
# _lr_psnr = 0.0
#
# HR_path = './cpexpres/1-1/0GT.png'
# SR_paths = glob.glob(f'./cpexpres/1-1/*.png')
# SR_paths.remove(HR_path)
# LR_paths = glob.glob(f'./cpexpres/LR/BSD'+'_LR/*.png')
#
#
# for SR_path in SR_paths:
#     filename = SR_path.split('/')[-1]
#
#     SR = Image.open(SR_path)
#     HR = Image.open(HR_path)
#     LR = Image.open(LR_paths[0])
#
#     SR = trans(SR)
#     HR = trans(HR)
#     LR = trans(LR)
#
#     SR = SR * 2 - 1
#     HR = HR * 2 - 1
#     LR = LR * 2 - 1
#
#     s = Ms.measure(SR, HR, LR, 4)
#
#     _psnr = s['psnr']
#     _ssim = s['ssim']
#     _lpips = s['lpips']
#     # _lr_psnr += s['lr_psnr']
#
#     print(f'{filename}: psnr: {_psnr}, '
#           f'ssim: {_ssim}, '
#           f'lpips: {_lpips}, ')
#     # f'lr_psnr: {_lr_psnr / count}')


trans = transforms.ToTensor()
# SR_paths = glob.glob('./checkpoints/diffsr_div2k_rrdb32_enhanceEdgeNew/results_0_DIV2K/SR/*.png')
# HR_paths = glob.glob('./checkpoints/diffsr_div2k_rrdb32_enhanceEdgeNew/results_0_DIV2K/HR/*.png')
# LR_paths = glob.glob('./checkpoints/diffsr_div2k_rrdb32_enhanceEdgeNew/results_0_DIV2K/LR/*.png')

model_names = ['EDSR', 'ESRGAN', 'Real-ESRGAN', 'LDM', 'Bicubic', 'EFD', 'RRDB', 'ResShift']
model_name = model_names[7]
data_names = ['DIV2K', 'Set5', 'Set14', 'BSD', 'Urban100', 'manga109', 'lsun']
data_num = 5
data_name = data_names[data_num]

SR_paths = glob.glob(f'./cpexpres/{model_name}/{data_name}/*.png')
HR_paths = glob.glob(f'./cpexpres/HR-New/{data_name}/*.png')
LR_paths = glob.glob(f'./cpexpres/LR/{data_name}'+'_LR/*.png')

_psnr = 0.0
_ssim = 0.0
_lpips = 0.0
_lr_psnr = 0.0

n_samples = len(SR_paths)
Ms = Measure()

count = 1
for i in range(n_samples):
    SR = Image.open(SR_paths[i])
    HR = Image.open(HR_paths[i])
    LR = Image.open(LR_paths[i])

    SR = trans(SR)
    HR = trans(HR)
    LR = trans(LR)

    # Resize HR to match the size of SR
    HR = torch.nn.functional.interpolate(HR.unsqueeze(0), size=SR.shape[1:], mode='bicubic',
                                                 align_corners=False).squeeze(0)

    SR = SR * 2 - 1
    HR = HR * 2 - 1
    LR = LR * 2 - 1

    s = Ms.measure(SR, HR, LR, 4)

    _psnr += s['psnr']
    _ssim += s['ssim']
    _lpips += s['lpips']
    #_lr_psnr += s['lr_psnr']

    print(f'psnr: {_psnr / count}, '
          f'ssim: {_ssim / count}, '
          f'lpips: {_lpips / count}, ')
          #f'lr_psnr: {_lr_psnr / count}')
    count += 1

print(f'psnr: {_psnr / n_samples:.3f}, '
      f'ssim: {_ssim / n_samples:.3f}, '
      f'lpips: {_lpips / n_samples:.3f}, ')
      #f'lr_psnr: {_lr_psnr / n_samples:.3f}')
