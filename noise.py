import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from functools import partial
from PIL import Image
from torchvision import transforms
import analysis
from tqdm import tqdm
import pandas as pd
from modules import Trans
from torchvision.utils import save_image

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

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class ResidualDiffusion(nn.Module):
    def __init__(self, 
                 timesteps=24, 
                 main_weight=1.0, 
                 lpips_weight=0.3, 
                 lpips_pow=2.0, 
                 beta_schedule='cosine', 
                 beta_s=0.008, 
                 beta_end=0.999, 
                 comp=1.0,
                 scale=1.0,
                 use_lipis=False):
        super().__init__()

        self.timesteps = timesteps
        self.use_lipis = use_lipis
        self.alpha = main_weight
        self.beta = lpips_weight
        self.lpips_pow = lpips_pow
        self.comp = comp
        self.scale = scale


        if beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps, s=beta_s)
        if beta_schedule == 'linear':
            betas = get_beta_schedule(timesteps, beta_end=beta_end)
            betas[-1] = 0.999

        alphas = 1. - betas
        alphas = self.scale * alphas
        alphas = np.clip(alphas, a_min=0, a_max=1)
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

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
    
    def noise_adder(self, x_start, t, res, noise=None, compensate=None):

        noise = default(noise, lambda: torch.randn_like(x_start))
        if isinstance(t, int):
            # 如果 t 是整数，创建对应的张量
            t_tensor = torch.full((1,), self.timesteps - t - 1, device=x_start.device, dtype=torch.long)
            t_tensor = t_tensor.clamp_min(0)
            t_cond = (t_tensor[:, None, None, None] >= 0).float()
        if compensate is not None:
            res_c = res * compensate + 0.3
            res = res_c
        #a = extract(self.sqrt_alphas_cumprod, t_tensor, x_start.shape)
        #b = extract(self.sqrt_one_minus_alphas_cumprod, t_tensor, x_start.shape)
        a = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.001]
        b = [0, 0.13, 0.16, 0.19, 0.22, 0.25, 0.28, 0.3]
        return (x_start +b[t] * noise * res) * t_cond + x_start * (1 - t_cond)
    
    def forward(self, x, t, res, noise=None, compensate=None):
        return self.noise_adder(x, t, res, noise, compensate)


if __name__ == '__main__':
    path = r"M:\github\database\DIV2K_valid_HR_1080_square\0604_cropped.png"
    image = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    timesteps = 8
    experiment_dir = analysis.create_experiment_dir()
    resdiffusion = ResidualDiffusion()
    save_dir = experiment_dir
    histogram_dir = os.path.join(experiment_dir, 'clean')
    residual_dir = os.path.join(experiment_dir, 'noise')
    os.makedirs(residual_dir, exist_ok=True) 
    os.makedirs(histogram_dir, exist_ok=True)

    hr_img = transform(image)
    hr_img = hr_img
    diff_imgs, res_imgs = analysis.worker(hr_img, timesteps, sp_delta=125, save_dir=histogram_dir, mode='Linear')
    for i, img in tqdm(enumerate(diff_imgs), total=len(diff_imgs)):
        # 确保去除批次维度

        img = img.squeeze(0)
        
        save_image(img, os.path.join(histogram_dir, f'noise_{i}.png'))

    noise_imgs = [] 
    for t in range(timesteps):
        noise_img = resdiffusion(diff_imgs[t], t, res_imgs[t], noise=None, compensate=20)
        noise_imgs.append(noise_img)
    for i, img in tqdm(enumerate(noise_imgs), total=len(noise_imgs)):
        # 确保去除批次维度

        img = img.squeeze(0)
        
        save_image(img, os.path.join(residual_dir, f'noise_{i}.png'))