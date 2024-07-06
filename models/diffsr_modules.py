import functools
import torch
from torch import nn
import torch.nn.functional as F
from utils.hparams import hparams
from .module_util import make_layer, initialize_weights
from .commons import Mish, SinusoidalPosEmb, RRDB, Residual, Rezero, LinearAttention
from .commons import ResnetBlock, Upsample, Block, Downsample
from .commons import SENetBlock
# from .openaimodel import SpatialTransformer


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if hparams['sr_scale'] == 8:
            self.upconv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, get_fea=False):
        feas = []
        x = (x + 1) / 2
        fea_first = fea = self.conv_first(x)
        for l in self.RRDB_trunk:
            fea = l(fea)
            feas.append(fea)
        trunk = self.trunk_conv(fea)
        fea = fea_first + trunk
        feas.append(fea)

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        if hparams['sr_scale'] == 8:
            fea = self.lrelu(self.upconv3(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea_hr = self.HRconv(fea)
        out = self.conv_last(self.lrelu(fea_hr))
        out = out.clamp(0, 1)
        out = out * 2 - 1
        if get_fea:
            return out, feas
        else:
            return out


# class EdgeBatchNorm2d(nn.Module):
#     def __init__(self, num_features, num_conditions):
#         super(EdgeBatchNorm2d, self).__init__()
#         self.num_features = num_features
#         self.num_conditions = num_conditions
#
#         self.bn = nn.BatchNorm2d(num_features)
#         self.gamma_conv = nn.Conv2d(num_conditions, num_features, kernel_size=1)
#         self.beta_conv = nn.Conv2d(num_conditions, num_features, kernel_size=1)
#
#     def forward(self, x, conditions):
#         # 计算批标准化的均值和方差
#         bn_mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
#         bn_var = torch.var(x, dim=(0, 2, 3), keepdim=True)
#
#         # 标准化
#         x_normalized = (x - bn_mean) / torch.sqrt(bn_var + 1e-5)
#
#         # 根据条件输入调整缩放因子和偏移量
#         gamma = self.gamma_conv(conditions)
#         beta = self.beta_conv(conditions)
#
#         # 缩放和平移
#         out = gamma * x_normalized + beta
#
#         # 添加残差连接
#         out = out + x
#
#         return out


# class FlattenAndMap(nn.Module):
#     def __init__(self):
#         super(FlattenAndMap, self).__init__()
#         self.flatten = nn.Flatten()
#         self.linear = nn.Linear(128*128*1, 1000)
#
#     def forward(self, x):
#         x = self.flatten(x)
#         x = self.linear(x)
#         return x


class Unet(nn.Module):
    def __init__(self, dim, out_dim=None, dim_mults=(1, 2, 4, 8), cond_dim=32):
        super().__init__()

        # self.EBN = EdgeBatchNorm2d(320, 1)

        dims = [3, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        groups = 0

        ### todo 2.5 dim1 +16

        # self.cond_proj = nn.ConvTranspose2d(cond_dim * ((hparams['rrdb_num_block'] + 1) // 3),
        #                                     dim, hparams['sr_scale'] * 2, hparams['sr_scale'],
        #                                     hparams['sr_scale'] // 2)

        self.cond_proj = nn.ConvTranspose2d(cond_dim * ((hparams['rrdb_num_block'] + 1) // 3) * 2,
                                            dim, hparams['sr_scale'] * 2, hparams['sr_scale'],
                                            hparams['sr_scale'] // 2)

        ### todo 2.5 end

        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # todo n2
        # for ind, (dim_in, dim_out) in enumerate(in_out):
        #     is_last = ind >= (num_resolutions - 1)
        #     self.downs.append(nn.ModuleList([
        #         ResnetBlock(dim_in, dim_out, time_emb_dim=dim, groups=groups),
        #         ResnetBlock(dim_out, dim_out, time_emb_dim=dim, groups=groups),
        #         Downsample(dim_out) if not is_last else nn.Identity()
        #     ]))

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=dim, groups=groups),
                ResnetBlock(dim_out, dim_out, time_emb_dim=dim, groups=groups),
                SENetBlock(dim_out, dim_out, time_emb_dim=dim, groups=groups),
                # SENetBlock(dim_out, dim_out, time_emb_dim=dim, groups=groups),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        # self.sptran_d1 = SpatialTransformer(192, 1, 192, depth=1, context_dim=1000)
        # self.sptran_d2 = SpatialTransformer(256, 1, 256, depth=1, context_dim=1000)
        # self.sptran_m1 = SpatialTransformer(256, 1, 256, depth=1, context_dim=1000)
        # self.sptran_u1 = SpatialTransformer(192, 1, 192, depth=1, context_dim=1000)
        # self.sptran_u2 = SpatialTransformer(128, 1, 128, depth=1, context_dim=1000)
        # #
        # self.flat_map = FlattenAndMap()

        # todo n2
        # mid_dim = dims[-1]
        # self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, groups=groups)
        # if hparams['use_attn']:
        #     self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        # self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, groups=groups)

        mid_dim = dims[-1]
        self.mid_block1 = SENetBlock(mid_dim, mid_dim, time_emb_dim=dim, groups=groups)
        if hparams['use_attn']:
            self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = SENetBlock(mid_dim, mid_dim, time_emb_dim=dim, groups=groups)

        # for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
        #     is_last = ind >= (num_resolutions - 1)
        #
        #     self.ups.append(nn.ModuleList([
        #         ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim, groups=groups),
        #         ResnetBlock(dim_in, dim_in, time_emb_dim=dim, groups=groups),
        #         Upsample(dim_in) if not is_last else nn.Identity()
        #     ]))

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim, groups=groups),
                ResnetBlock(dim_in, dim_in, time_emb_dim=dim, groups=groups),
                SENetBlock(dim_in, dim_in, time_emb_dim=dim, groups=groups),
                # SENetBlock(dim_in, dim_in, time_emb_dim=dim, groups=groups),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            Block(dim, dim, groups=groups),
            nn.Conv2d(dim, out_dim, 1)
        )

        if hparams['res'] and hparams['up_input']:
            self.up_proj = nn.Sequential(
                nn.ReflectionPad2d(1), nn.Conv2d(3, dim, 3),
            )
        if hparams['use_wn']:
            self.apply_weight_norm()
        if hparams['weight_init']:
            self.apply(initialize_weights)

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                # print(f"| Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def forward(self, x, time, cond, img_lr_up):

        ### todo 2.4
        cond, feature_tensor = cond

        # _ft = feature_tensor.to(torch.float32)
        # _cond = self.flat_map(_ft)
        # _cond = torch.unsqueeze(_cond, 1)

        feature_tensor = feature_tensor / 255.0
        # feature_tensor = torch.cat([feature_tensor] * 16, 1)

        cond = torch.cat(cond[2::3], 1)
        feature_tensor = F.interpolate(feature_tensor, scale_factor=0.25, mode='bilinear', align_corners=False)

        ### todo 2.4.0 cat
        n = 10
        kernel_size = 2 * n + 1
        kernel = torch.ones(1, 1, kernel_size, kernel_size, dtype=torch.float32)
        kernel = kernel.to('cuda:0')
        feature_tensor = F.conv2d(feature_tensor, kernel, padding=n)

        cond_f = cond * feature_tensor
        cond = torch.cat([cond, cond_f], dim=1)
        ### todo 2.4.1 edge batch normalization
        # 64 320 32 32 + 64 16 32 32
        # cond = self.EBN(cond, feature_tensor)
        cond = self.cond_proj(cond)

        ### todo 2.4 end

        t = self.time_pos_emb(time)
        t = self.mlp(t)

        h = []
        for i, (resnet, resnet2, senet,  downsample) in enumerate(self.downs):
            x = resnet(x, t)
            x = resnet2(x, t)
            x = senet(x, t)
            # x = senet2(x, t)
            if i == 0:
                x = x + cond
                if hparams['res'] and hparams['up_input']:
                    x = x + self.up_proj(img_lr_up)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        if hparams['use_attn']:
            x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2,senet, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = senet(x, t)
            # x = senet2(x, t)
            x = upsample(x)

        return self.final_conv(x)
        # cond = self.cond_proj(torch.cat(cond[2::3], 1))
        # for i, (resnet, resnet2, downsample) in enumerate(self.downs):
        #     x = resnet(x, t)
        #     x = resnet2(x, t)
        #     if i == 0:
        #         x = x + cond
        #         if hparams['res'] and hparams['up_input']:
        #             x = x + self.up_proj(img_lr_up)
        #     if i == 2:
        #         x = self.sptran_d1(x, _cond)
        #     if i == 3:
        #         x = self.sptran_d2(x, _cond)
        #     h.append(x)
        #     x = downsample(x)
        #
        # x = self.mid_block1(x, t)
        # x = self.sptran_m1(x, _cond)
        #
        # if hparams['use_attn']:
        #     x = self.mid_attn(x)
        # x = self.mid_block2(x, t)
        #
        # for i, (resnet, resnet2, upsample) in enumerate(self.ups):
        #     x = torch.cat((x, h.pop()), dim=1)
        #     x = resnet(x, t)
        #     x = resnet2(x, t)
        #     if i == 0:
        #         x = self.sptran_u1(x, _cond)
        #     if i == 1:
        #         x = self.sptran_u2(x, _cond)
        #     x = upsample(x)
        #
        # return self.final_conv(x)

    def make_generation_fast_(self):
        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(remove_weight_norm)
