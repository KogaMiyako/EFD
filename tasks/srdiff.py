import os.path

import torch
from models.diffsr_modules import Unet, RRDBNet
from models.diffusion import GaussianDiffusion
from tasks.trainer import Trainer
from utils.hparams import hparams
from utils.utils import load_ckpt


class SRDiffTrainer(Trainer):
    def build_model(self):
        hidden_size = hparams['hidden_size']
        dim_mults = hparams['unet_dim_mults']
        dim_mults = [int(x) for x in dim_mults.split('|')]

        ### todo 1 动刀处： cond_dim = feat + 'edge feat'

        denoise_fn = Unet(
            hidden_size, out_dim=3, cond_dim=hparams['rrdb_num_feat'], dim_mults=dim_mults)
        # denoise_fn = Unet(
        #     hidden_size, out_dim=3, cond_dim=hparams['rrdb_num_feat'] + 1, dim_mults=dim_mults)

        ### todo 1 end

        if hparams['use_rrdb']:
            rrdb = RRDBNet(3, 3, hparams['rrdb_num_feat'], hparams['rrdb_num_block'],
                           hparams['rrdb_num_feat'] // 2)
            if hparams['rrdb_ckpt'] != '' and os.path.exists(hparams['rrdb_ckpt']):
                load_ckpt(rrdb, hparams['rrdb_ckpt'])
        else:
            rrdb = None
        self.model = GaussianDiffusion(
            denoise_fn=denoise_fn,
            rrdb_net=rrdb,
            timesteps=hparams['timesteps'],
            loss_type=hparams['loss_type']
        )
        self.global_step = 0
        return self.model

    # def split_image_into_patches(self, image, patch_size):
    #     batch_size, num_channels, height, width = image.shape
    #     stride = patch_size  # Stride for overlapping patches
    #
    #     num_patches_h = (height + stride - 1) // stride
    #     num_patches_w = (width + stride - 1) // stride
    #
    #     # Split the image into patches
    #     patches = []
    #     for h in range(num_patches_h):
    #         for w in range(num_patches_w):
    #
    #             start_h = h * stride
    #             start_w = w * stride
    #
    #             if start_w + stride > width:
    #                 start_w = width - stride
    #
    #             if start_h + stride > height:
    #                 start_h = height - stride
    #
    #             patch = image[:, :, start_h:start_h + patch_size, start_w:start_w + patch_size]
    #             patches.append(patch)
    #
    #     return patches
    #
    # def merge_patches_into_image(self, patches, image_shape):
    #     _, num_channels, height, width = image_shape
    #     patch_size = patches[0].shape[2]
    #     stride = patch_size
    #
    #     num_patches_h = (height + stride - 1) // stride
    #     num_patches_w = (width + stride - 1) // stride
    #
    #     merged_image = torch.zeros(1, num_channels, height, width)
    #
    #     for i, patch in enumerate(patches):
    #         h = i // num_patches_w
    #         w = i % num_patches_w
    #
    #         start_h = h * stride
    #         start_w = w * stride
    #
    #         if start_w + stride > width:
    #             start_w = width - stride
    #
    #         if start_h + stride > height:
    #             start_h = height - stride
    #
    #         merged_image[:, :, start_h:start_h + patch_size, start_w:start_w + patch_size] = patch
    #
    #     return merged_image

    # def sample_and_test(self, sample):
    #     ret = {k: 0 for k in self.metric_keys}
    #     ret['n_samples'] = 0
    #     ### todo ?
    #     img_hr = sample['img_hr']
    #     img_hr_patches = self.split_image_into_patches(img_hr, 128)
    #
    #     img_lr = sample['img_lr']
    #     img_lr_patches = self.split_image_into_patches(img_lr, 32)
    #
    #     img_lr_up = sample['img_lr_up']
    #     img_lr_up_patches = self.split_image_into_patches(img_lr_up, 128)
    #
    #     assert len(img_hr_patches) == len(img_lr_patches)
    #
    #     img_sr_patches = []
    #     rrdb_out_patches = []
    #     for i in range(len(img_hr_patches)):
    #         img_sr_p, rrdb_out_p = self.model.sample(img_lr_patches[i], img_lr_up_patches[i], img_hr_patches[i].shape)
    #         img_sr_patches.append(img_sr_p)
    #         rrdb_out_patches.append(rrdb_out_p)
    #
    #     img_sr = self.merge_patches_into_image(img_sr_patches, img_hr.shape)
    #     rrdb_out = self.merge_patches_into_image(rrdb_out_patches, img_hr.shape)
    #
    #     for b in range(img_sr.shape[0]):
    #         s = self.measure.measure(img_sr[b], img_hr[b], img_lr[b], hparams['sr_scale'])
    #         ret['psnr'] += s['psnr']
    #         ret['ssim'] += s['ssim']
    #         ret['lpips'] += s['lpips']
    #         ret['lr_psnr'] += s['lr_psnr']
    #         ret['n_samples'] += 1
    #     return img_sr, rrdb_out, ret

    def sample_and_test(self, sample):
        ret = {k: 0 for k in self.metric_keys}
        ret['n_samples'] = 0

        img_hr = sample['img_hr']
        img_lr = sample['img_lr']
        img_lr_up = sample['img_lr_up']
        img_sr, rrdb_out = self.model.sample(img_lr, img_lr_up, img_hr.shape)

        for b in range(img_sr.shape[0]):
            s = self.measure.measure(img_sr[b], img_hr[b], img_lr[b], hparams['sr_scale'])
            ret['psnr'] += s['psnr']
            ret['ssim'] += s['ssim']
            ret['lpips'] += s['lpips']
            # ret['lr_psnr'] += s['lr_psnr']
            ret['n_samples'] += 1
        return img_sr, rrdb_out, ret

    # def sample_and_test(self, sample):
    #     ret = {k: 0 for k in self.metric_keys}
    #     ret['n_samples'] = 0
    #     img_hr = sample['img_hr']
    #     img_lr_up = sample['img_lr_up']
    #     img_lr = sample['img_lr']
    #
    #     # 获取输入图片的尺寸
    #     _, _, H_hr, W_hr = img_hr.size()
    #     _, _, H_lr_up, W_lr_up = img_lr_up.size()
    #     _, _, H_lr, W_lr = img_lr.size()
    #
    #     # 计算分割的尺寸
    #     split_size = (128, 128)
    #     stride = (split_size[0] // 4, split_size[1] // 4)
    #     n_rows = (H_hr - split_size[0]) // stride[0] + 1
    #     n_cols = (W_hr - split_size[1]) // stride[1] + 1
    #
    #     split_size_lr = (32, 32)
    #
    #     img_sr_list = []
    #     rrdb_out_list = []
    #
    #     # 对图片进行循环分割和处理
    #     for i in range(n_rows):
    #         for j in range(n_cols):
    #             # 计算当前分割的起始位置
    #             start_row = i * stride[0]
    #             start_col = j * stride[1]
    #
    #             # 从原始图片中提取当前分割区域
    #             img_hr_patch = img_hr[:, :, start_row:start_row + split_size[0], start_col:start_col + split_size[1]]
    #             img_lr_up_patch = img_lr_up[:, :, start_row // 4:start_row // 4 + split_size[0] // 4,
    #                               start_col // 4:start_col // 4 + split_size[1] // 4]
    #             img_lr_patch = img_lr[:, :, start_row // 4:start_row // 4 + split_size[0] // 4,
    #                            start_col // 4:start_col // 4 + split_size[1] // 4]
    #
    #             # 调用模型进行处理
    #             img_sr_patch, rrdb_out_patch, ret_patch = self.model.sample(img_lr_patch, img_lr_up_patch,
    #                                                                         img_hr_patch.shape)
    #
    #             # 累加度量指标
    #             ret['psnr'] += ret_patch['psnr']
    #             ret['ssim'] += ret_patch['ssim']
    #             ret['lpips'] += ret_patch['lpips']
    #             ret['lr_psnr'] += ret_patch['lr_psnr']
    #             ret['n_samples'] += ret_patch['n_samples']
    #
    #             # 将处理结果添加到列表中
    #             img_sr_list.append(img_sr_patch)
    #             rrdb_out_list.append(rrdb_out_patch)
    #
    #     # 合并分割结果
    #     img_sr = torch.cat(img_sr_list, dim=2)
    #     img_sr = torch.cat(torch.split(img_sr, H_hr, dim=2), dim=3)
    #     rrdb_out = torch.cat(rrdb_out_list, dim=2)
    #     rrdb_out = torch.cat(torch.split(rrdb_out, H_hr, dim=2), dim=3)
    #
    #     return img_sr, rrdb_out, ret

    def build_optimizer(self, model):
        params = list(model.named_parameters())
        if not hparams['fix_rrdb']:
            params = [p for p in params if 'rrdb' not in p[0]]
        params = [p[1] for p in params]
        return torch.optim.Adam(params, lr=hparams['lr'])

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, hparams['decay_steps'], gamma=0.5)

    def training_step(self, batch):
        img_hr = batch['img_hr']
        img_lr = batch['img_lr']
        img_lr_up = batch['img_lr_up']
        losses, _, _ = self.model(img_hr, img_lr, img_lr_up)
        total_loss = sum(losses.values())
        return losses, total_loss
