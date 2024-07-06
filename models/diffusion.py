from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from .module_util import default
from utils.sr_utils import SSIM, PerceptualLoss
from utils.hparams import hparams

import cv2


# gaussian diffusion trainer class
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas


def get_beta_schedule(num_diffusion_timesteps, beta_schedule='linear', beta_start=0.0001, beta_end=0.02):
    if beta_schedule == 'quad':
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == 'linear':
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'warmup10':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == 'warmup50':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == 'const':
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)

# 定义Canny边缘检测函数
def canny_transform(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.uint8)  # 将图像转换为8位无符号整数类型
    img = cv2.Canny(img, threshold1=100, threshold2=200)
    img = np.expand_dims(img, axis=0)
    return torch.from_numpy(img)


class GaussianDiffusion(nn.Module):
    def __init__(self, denoise_fn, rrdb_net, timesteps=1000, loss_type='l1'):
        super().__init__()
        self.denoise_fn = denoise_fn
        # condition net
        self.rrdb = rrdb_net
        self.ssim_loss = SSIM(window_size=11)
        if hparams['aux_percep_loss']:
            self.percep_loss_fn = [PerceptualLoss()]

        if hparams['beta_schedule'] == 'cosine':
            betas = cosine_beta_schedule(timesteps, s=hparams['beta_s'])
        if hparams['beta_schedule'] == 'linear':
            betas = get_beta_schedule(timesteps, beta_end=hparams['beta_end'])
            if hparams['res']:
                betas[-1] = 0.999

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        self.sample_tqdm = True

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, noise_pred, clip_denoised: bool):
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    def forward(self, img_hr, img_lr, img_lr_up, t=None, *args, **kwargs):
        """
        :param img_hr: hr image
        :param img_lr: lr image
        :param img_lr_up: lr image expand to hr

            we get res from img_lr_up and img_hr as input
                to de_noise
        """
        x = img_hr
        b, *_, device = *x.shape, x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long() \
            if t is None else torch.LongTensor([t]).repeat(b).to(device)

        ### todo 2.1 得到边缘条件 use img_hr

        ### todo 2.1.1 use img_lr to get edge 64

        # img_hr_temp = img_lr
        # img_hr_temp_cpu = img_hr_temp.cpu()
        #
        # # 应用变换操作得到特征Tensor
        # feature_tensor = torch.stack(
        #     [canny_transform(np.transpose(img_hr_temp_cpu[i].numpy(), (1, 2, 0))) for i in range(x.shape[0])])
        #
        # # feature_tensor = F.interpolate(feature_tensor, size=(128, 128), mode='bilinear', align_corners=False)
        # feature_tensor = feature_tensor.to(device)
        # del img_hr_temp_cpu

        ###  2.1.0 use img_hr to get edge

        img_hr_temp = img_hr
        img_hr_temp_cpu = img_hr_temp.cpu()

        # 应用变换操作得到特征Tensor
        feature_tensor = torch.stack(
            [canny_transform(np.transpose(img_hr_temp_cpu[i].numpy(), (1, 2, 0))) for i in range(x.shape[0])])

        feature_tensor = feature_tensor.to(device)
        del img_hr_temp_cpu

        ### todo 2.1 end

        if hparams['use_rrdb']:
            if hparams['fix_rrdb']:
                self.rrdb.eval()
                with torch.no_grad():
                    rrdb_out, cond = self.rrdb(img_lr, True)

                ### todo 2.2 fix_rrdb: true 连接边缘条件

                cond = [cond, feature_tensor]

                ### todo 2.2 end

            else:
                rrdb_out, cond = self.rrdb(img_lr, True)
        else:
            rrdb_out = img_lr_up
            cond = img_lr
        x = self.img2res(x, img_lr_up)
        ### todo 3.0
        p_losses, x_tp1, noise_pred, x_t, x_t_gt, x_0 = self.p_losses(x, t, cond, img_lr_up, feature_tensor, *args, **kwargs)
        # p_losses, x_tp1, noise_pred, x_t, x_t_gt, x_0 = self.p_losses(x, t, cond, img_lr_up,  *args, **kwargs)
        ### todo 3.0 end
        ret = {'q': p_losses}
        if not hparams['fix_rrdb']:
            if hparams['aux_l1_loss']:
                ret['aux_l1'] = F.l1_loss(rrdb_out, img_hr)
            if hparams['aux_ssim_loss']:
                ret['aux_ssim'] = 1 - self.ssim_loss(rrdb_out, img_hr)
            if hparams['aux_percep_loss']:
                ret['aux_percep'] = self.percep_loss_fn[0](img_hr, rrdb_out)
        # x_recon = self.res2img(x_recon, img_lr_up)
        x_tp1 = self.res2img(x_tp1, img_lr_up)
        x_t = self.res2img(x_t, img_lr_up)
        x_t_gt = self.res2img(x_t_gt, img_lr_up)
        return ret, (x_tp1, x_t_gt, x_t), t

    def p_losses(self, x_start, t, cond, img_lr_up, edge_feature, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_tp1_gt = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_t_gt = self.q_sample(x_start=x_start, t=t - 1, noise=noise)
        noise_pred = self.denoise_fn(x_tp1_gt, t, cond, img_lr_up)

        ### todo 2.3
        cond, _ = cond
        ### todo 2.3

        x_t_pred, x0_pred = self.p_sample(x_tp1_gt, t, cond, img_lr_up, noise_pred=noise_pred)

        if self.loss_type == 'l1':
            loss = (noise - noise_pred).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, noise_pred)
        elif self.loss_type == 'ssim':
            loss = (noise - noise_pred).abs().mean()
            loss = loss + (1 - self.ssim_loss(noise, noise_pred))
        else:
            raise NotImplementedError()

        ### todo 3.0 apply edge loss

        device = x0_pred.device

        x0_pred_temp = x0_pred
        x0_pred_temp = self.res2img(x0_pred_temp, img_lr_up)
        x0_pred_temp_cpu = x0_pred_temp.cpu()

        # 应用变换操作得到特征Tensor
        feature_tensor = torch.stack(
            [canny_transform(np.transpose(x0_pred_temp_cpu[i].numpy(), (1, 2, 0))) for i in range(x_start.shape[0])])

        feature_tensor = feature_tensor.to(device)
        del x0_pred_temp_cpu

        assert feature_tensor.shape == edge_feature.shape

        feature_tensor = feature_tensor / 255.0
        edge_feature = edge_feature / 255.0

        feature_tensor = feature_tensor.view(-1)
        edge_feature = edge_feature.view(-1)

        loss_edge = F.binary_cross_entropy(feature_tensor, edge_feature)

        loss = loss + 0.01 * loss_edge

        ### todo 3.0 end

        return loss, x_tp1_gt, noise_pred, x_t_pred, x_t_gt, x0_pred

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        t_cond = (t[:, None, None, None] >= 0).float()
        t = t.clamp_min(0)
        return (
                       extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                       extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
               ) * t_cond + x_start * (1 - t_cond)

    @torch.no_grad()
    def p_sample(self, x, t, cond, img_lr_up, noise_pred=None, clip_denoised=True, repeat_noise=False):
        if noise_pred is None:
            noise_pred = self.denoise_fn(x, t, cond=cond, img_lr_up=img_lr_up)
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x0_pred = self.p_mean_variance(
            x=x, t=t, noise_pred=noise_pred, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0_pred

    @torch.no_grad()
    def sample(self, img_lr, img_lr_up, shape, save_intermediate=False):
        device = self.betas.device
        # device = torch.device("cuda", 1)

        b = shape[0]
        if not hparams['res']:
            t = torch.full((b,), self.num_timesteps - 1, device=device, dtype=torch.long)
            img = self.q_sample(img_lr_up, t)
        else:
            img = torch.randn(shape, device=device)
        if hparams['use_rrdb']:
            rrdb_out, cond = self.rrdb(img_lr, True)
        else:
            rrdb_out = img_lr_up
            cond = img_lr

        ### todo 3.1 cond 变化
        ### todo 3.1.1 cond 变化 先lredge再放大

        # img_hr_temp = img_lr
        # img_hr_temp_cpu = img_hr_temp.cpu()
        #
        # # 应用变换操作得到特征Tensor
        # feature_tensor = torch.stack(
        #     [canny_transform(np.transpose(img_hr_temp_cpu[i].numpy(), (1, 2, 0))) for i in range(b)])
        # feature_tensor = F.interpolate(feature_tensor, size=(128, 128), mode='bilinear', align_corners=False)
        #
        # feature_tensor = feature_tensor.to(device)
        # del img_hr_temp_cpu
        #
        # cond = [cond, feature_tensor]

        img_hr_temp = img_lr_up
        img_hr_temp_cpu = img_hr_temp.cpu()

        # 应用变换操作得到特征Tensor
        feature_tensor = torch.stack(
            [canny_transform(np.transpose(img_hr_temp_cpu[i].numpy(), (1, 2, 0))) for i in range(b)])

        feature_tensor = feature_tensor.to(device)
        del img_hr_temp_cpu

        cond = [cond, feature_tensor]

        ### todo 3.1 end

        it = reversed(range(0, self.num_timesteps))
        if self.sample_tqdm:
            it = tqdm(it, desc='sampling loop time step', total=self.num_timesteps)
        images = []
        for i in it:
            img, x_recon = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long), cond, img_lr_up)
            if save_intermediate:
                img_ = self.res2img(img, img_lr_up)
                x_recon_ = self.res2img(x_recon, img_lr_up)
                images.append((img_.cpu(), x_recon_.cpu()))
        img = self.res2img(img, img_lr_up)
        if save_intermediate:
            return img, rrdb_out, images
        else:
            return img, rrdb_out

    @torch.no_grad()
    def interpolate(self, x1, x2, img_lr, img_lr_up, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)
        if hparams['use_rrdb']:
            rrdb_out, cond = self.rrdb(img_lr, True)
        else:
            cond = img_lr

        assert x1.shape == x2.shape

        x1 = self.img2res(x1, img_lr_up)
        x2 = self.img2res(x2, img_lr_up)

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img, x_recon = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long), cond, img_lr_up)

        img = self.res2img(img, img_lr_up)
        return img

    def res2img(self, img_, img_lr_up, clip_input=None):
        if clip_input is None:
            clip_input = hparams['clip_input']
        if hparams['res']:
            if clip_input:
                img_ = img_.clamp(-1, 1)
            img_ = img_ / hparams['res_rescale'] + img_lr_up
        return img_

    def img2res(self, x, img_lr_up, clip_input=None):
        if clip_input is None:
            clip_input = hparams['clip_input']
        if hparams['res']:
            x = (x - img_lr_up) * hparams['res_rescale']
            if clip_input:
                x = x.clamp(-1, 1)
        return x
