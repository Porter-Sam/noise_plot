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
import math
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


import matplotlib.pyplot as plt
import numpy as np

def plot_res_imgs_distribution(res_imgs):
    """
    绘制res_imgs中所有图像的值分布直方图
    """
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 计算子图布局
    n_images = len(res_imgs)
    cols = min(4, n_images)  # 最多4列
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    all_values = []
    
    for i in range(n_images):
        # 将tensor转换为numpy数组并展平
        values = res_imgs[i].cpu().numpy().flatten()
        all_values.extend(values)
        
        # 绘制单个图像的直方图
        axes[i].hist(values, bins=100, alpha=0.7, density=True, edgecolor='black')
        axes[i].set_title(f'Res Image {i} Distribution\nRange: [{values.min():.4f}, {values.max():.4f}]')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Density')
        
        # 添加垂直线表示均值
        axes[i].axvline(values.mean(), color='red', linestyle='--', label=f'Mean: {values.mean():.4f}')
        axes[i].legend()
    
    # 隐藏多余的子图
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # 绘制整体分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(all_values, bins=200, alpha=0.7, density=True, edgecolor='black')
    plt.title('Overall Distribution of All Res Images Values')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.axvline(np.mean(all_values), color='red', linestyle='--', label=f'Mean: {np.mean(all_values):.4f}')
    plt.axvline(np.median(all_values), color='green', linestyle='--', label=f'Median: {np.median(all_values):.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 打印统计信息
    print(f"Total number of images: {n_images}")
    print(f"Value range across all images: [{min(all_values):.6f}, {max(all_values):.6f}]")
    print(f"Mean value: {np.mean(all_values):.6f}")
    print(f"Median value: {np.median(all_values):.6f}")
    print(f"Standard deviation: {np.std(all_values):.6f}")
    print(f"Values < 0: {sum(1 for v in all_values if v < 0)}")
    print(f"Values > 0: {sum(1 for v in all_values if v > 0)}")

def process_patches_with_threshold(image_tensor, patch_size=16):
    """
    将图像均分成256个patch，统计每patch内值为1的像素数量，
    如果大于一半就把整个patch内的像素都置1，否则置0
    
    Args:
        image_tensor: 输入图像张量 (C, H, W) 或 (H, W)
        patch_size: 每个patch的大小，默认16x16 (16*16=256 patches for 64x64 image)
    
    Returns:
        processed_tensor: 处理后的图像张量
    """
    # 确保输入是torch张量
    if not isinstance(image_tensor, torch.Tensor):
        image_tensor = torch.tensor(image_tensor)
    
    # 处理单通道图像
    if len(image_tensor.shape) == 2:
        C = 1
        H, W = image_tensor.shape
        image_tensor = image_tensor.unsqueeze(0)  # 添加通道维度
    elif len(image_tensor.shape) == 3:
        C, H, W = image_tensor.shape
    else:
        raise ValueError("输入张量应为 (H, W) 或 (C, H, W) 格式")
    
    # 计算patch数量
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    
    if num_patches_h * num_patches_w != 256:
        print(f"警告: 图像尺寸 ({H}, {W}) 无法精确分割成256个{patch_size}x{patch_size}的patch")
        print(f"实际patch数量: {num_patches_h * num_patches_w}")
    
    # 初始化输出张量
    output_tensor = torch.zeros_like(image_tensor)
    
    # 遍历每个patch
    for c in range(C):  # 遍历每个通道
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                # 计算当前patch的坐标范围
                start_h = i * patch_size
                end_h = min(start_h + patch_size, H)  # 防止越界
                start_w = j * patch_size
                end_w = min(start_w + patch_size, W)  # 防止越界
                
                # 提取当前patch
                patch = image_tensor[c, start_h:end_h, start_w:end_w]
                
                # 统计值为1的像素数量
                ones_count = (patch == 1).sum().item()
                total_pixels = patch.numel()
                
                # 判断是否超过一半
                if ones_count > total_pixels / 2:
                    # 将整个patch设置为1
                    output_tensor[c, start_h:end_h, start_w:end_w] = 1
                else:
                    # 将整个patch设置为0
                    output_tensor[c, start_h:end_h, start_w:end_w] = 0
    
    # 如果原始输入是2D，则返回2D
    if len(output_tensor.shape) == 3 and C == 1:
        output_tensor = output_tensor.squeeze(0)
    
    return output_tensor


import torch
import cv2
import numpy as np

def dense_mask_connect(
    tensor,
    normalize_range=(0, 1),  # 输入归一化范围：(min, max)，默认(0,1)，可选(-1,1)
    mask_threshold=0.5,      # 归一化后遮罩判定阈值（>此值视为1，否则0）
    kernel_size=5,           # 膨胀核大小（奇数）
    dilate_iter=2,           # 膨胀次数
    min_dense_area=500,      # 最小密集区面积（过滤零星1值）
    close_kernel_size=5      # 闭运算核大小（填充空隙）
):
    """
    0/1遮罩专用：归一化BCHW张量密集区连片函数（输入输出均为0/1，不引入255）
    
    核心特点：
    - 输入：归一化BCHW张量（仅0/1遮罩，值在[0,1]或[-1,1]）
    - 输出：BCHW张量（纯0/1遮罩，[batch_size, 1, height, width]）
    - 功能：过滤零星1值（直接转为0），密集1值连成片，0值区域（黑色）完全不变
    
    参数说明：
    - tensor: 输入4维BCHW张量 [batch_size, channels, height, width]，0/1遮罩已归一化
              支持通道：1（单通道遮罩）、3/4通道（自动转为单通道处理）
    - normalize_range: 输入的归一化范围，可选 (0,1) 或 (-1,1)
    - mask_threshold: 归一化后遮罩判定阈值（如0.5：>0.5视为1，否则0，确保纯0/1）
    - kernel_size: 膨胀核大小（奇数，如3/5/7，控制1值连接范围）
    - dilate_iter: 膨胀次数（次数越多，分散1值越易连接）
    - min_dense_area: 最小密集区面积（小于此的1值区域直接转为0，过滤零星点）
    - close_kernel_size: 闭运算核大小（填充密集1值区域内部小空隙）
    """
    # 1. 校验输入格式
    if len(tensor.shape) != 4:
        raise ValueError(f"输入必须是4维BCHW张量 [B,C,H,W]，当前形状：{tensor.shape}")
    batch_size, channels, height, width = tensor.shape
    if channels not in (1, 3, 4):
        raise ValueError(f"仅支持1/3/4通道，当前通道数：{channels}")
    if normalize_range not in ((0, 1), (-1, 1)):
        raise ValueError("归一化范围仅支持 (0,1) 或 (-1,1)")
    
    # 2. 将PyTorch张量转换为NumPy数组
    if isinstance(tensor, torch.Tensor):
        tensor_np = tensor.detach().cpu().numpy()
    else:
        tensor_np = np.array(tensor)
    
    # 3. 反归一化：将归一化的0/1遮罩还原到 [0, 1] 原始范围（确保后续判定准确）
    def denormalize_to_01(x, orig_min, orig_max):
        # 公式：x_01 = (x - orig_min) / (orig_max - orig_min)
        x_01 = (x - orig_min) / (orig_max - orig_min)
        return np.clip(x_01, 0.0, 1.0)  # 裁剪溢出值，确保在[0,1]
    
    orig_min, orig_max = normalize_range
    tensor_01 = denormalize_to_01(tensor_np, orig_min, orig_max)
    
    # 4. 转为纯0/1单通道遮罩（无论输入多少通道，统一处理为单通道）
    # 多通道时取均值（确保3/4通道遮罩正确转为单通道）
    if channels > 1:
        tensor_01 = np.mean(tensor_01, axis=1, keepdims=True)  # [B,1,H,W]
    # 基于阈值生成纯0/1遮罩（避免浮点误差）
    mask_01 = (tensor_01 > mask_threshold).astype(np.uint8)  # 0或1（uint8不影响后续操作）
    
    # 5. 修正形态学核为奇数（确保形态学操作对称）
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    close_kernel_size = close_kernel_size if close_kernel_size % 2 == 1 else close_kernel_size + 1
    dilate_kernel = np.ones((kernel_size, kernel_size), np.uint8)
    close_kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
    
    # 6. 批量处理每个样本
    result_list = []
    for i in range(batch_size):
        # 取单个样本：[1, H, W] → [H, W]（OpenCV需要2D格式）
        single_mask = mask_01[i].squeeze(0)  # [1,H,W] → [H,W]（0或1）
        
        # 7. 膨胀操作：让密集的1值（白点）初步连接（仅操作1值，0值区域不变）
        # 注意：膨胀核作用于1值，会让1值区域扩大，实现分散1值连接
        dilated = cv2.dilate(single_mask, dilate_kernel, iterations=dilate_iter)
        
        # 8. 连通区域分析：过滤零星1值（仅保留面积达标的密集区域）
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            dilated, connectivity=8  # 8连通更贴合密集区连接逻辑
        )
        
        # 生成筛选后的密集区遮罩（达标区域设为1，其余为0）
        dense_mask = np.zeros_like(labels, dtype=np.uint8)
        for label in range(1, num_labels):  # 0是背景（0值），跳过
            area = stats[label, 4]
            if area >= min_dense_area:
                dense_mask[labels == label] = 1  # 仅保留密集区的1值
        
        # 9. 闭运算：填充密集1值区域内部的小空隙（用1值填充，确保连片）
        final_mask = cv2.morphologyEx(dense_mask, cv2.MORPH_CLOSE, close_kernel)
        
        # 10. 恢复BCHW格式：[H,W] → [1, H, W]（单通道）
        final_mask = final_mask[None, ...]  # 增加通道维度：[H,W] → [1,H,W]
        result_list.append(final_mask)
    
    # 11. 拼接批量结果：[batch_size, 1, H, W]，值为0或1（uint8可直接转为bool使用）
    result_tensor = np.stack(result_list, axis=0)
    
    # 12. 转换回PyTorch张量（如果输入是PyTorch张量）
    if isinstance(tensor, torch.Tensor):
        result_tensor = torch.from_numpy(result_tensor).to(tensor.device).float()
    
    return result_tensor  # 输出类型

def mask_edge_feather(
    mask_tensor,  # BCHW格式的0/1遮罩张量 [batch, 1, h, w]
    feather_size=5  # 羽化半径（像素，越大边缘越柔和）
):
    """
    为0/1遮罩边缘添加羽化效果（仅边缘渐变，主体保持0/1）
    
    参数说明：
    - mask_tensor: BCHW格式的0/1遮罩（可以是PyTorch张量或numpy数组，值在0-1之间）
    - feather_size: 羽化半径（像素，建议3-15，根据遮罩尺寸调整）
    
    返回：
    - feathered_mask: 羽化后的BCHW遮罩（float32类型，值在0-1之间，边缘渐变）
    """
    # 将PyTorch张量转换为NumPy数组
    if isinstance(mask_tensor, torch.Tensor):
        mask_np = mask_tensor.detach().cpu().numpy()
    else:
        mask_np = mask_tensor.copy()
    
    # 确保是二值遮罩
    binary_mask = (mask_np > 0.5).astype(np.float32)
    
    # 校验输入格式
    if len(binary_mask.shape) != 4 or binary_mask.shape[1] != 1:
        raise ValueError("输入必须是BCHW格式单通道遮罩 [batch, 1, h, w]")
    
    batch_size, _, h, w = binary_mask.shape
    feather_size = max(1, int(feather_size))  # 确保羽化值为正整数
    
    # 构造羽化核（高斯核，模拟柔边）
    kernel_size = 2 * feather_size + 1  # 核大小为奇数
    gaussian_kernel = cv2.getGaussianKernel(kernel_size, feather_size)
    gaussian_kernel = gaussian_kernel @ gaussian_kernel.T  # 转为2D高斯核
    gaussian_kernel = gaussian_kernel / gaussian_kernel.max()  # 归一化到0-1

    result_list = []
    for i in range(batch_size):
        # 取出单个遮罩 [1, h, w] → [h, w]
        single_mask = binary_mask[i].squeeze(0)
        
        # 对二值遮罩进行高斯模糊，产生羽化效果
        feathered = cv2.filter2D(single_mask, -1, gaussian_kernel)
        
        # 确保值域在[0,1]之间
        feathered = np.clip(feathered, 0.0, 1.0)
        
        # 恢复BCHW格式 [1, h, w]
        feathered = feathered[None, ...]
        result_list.append(feathered)
    
    # 拼接批量结果，转为float32（0-1渐变）
    feathered_mask = np.stack(result_list, axis=0).astype(np.float32)
    
    # 如果输入是PyTorch张量，返回PyTorch张量
    if isinstance(mask_tensor, torch.Tensor):
        feathered_mask = torch.from_numpy(feathered_mask).to(mask_tensor.device).float()
    
    return feathered_mask

if __name__ == '__main__':
    path = r"E:\YuShihang\database\cropped_img1024_center\0558_cropped_.png"
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
    mask_dir = os.path.join(experiment_dir, 'mask')
    os.makedirs(residual_dir, exist_ok=True) 
    os.makedirs(histogram_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    hr_img = transform(image)
    hr_img = hr_img
    diff_imgs, res_imgs = analysis.worker(hr_img, timesteps, sp_delta=125, save_dir=histogram_dir, mode='Linear')
    masks = []
    for i in range(len(res_imgs)):
        res_imgs[i] = (res_imgs[i] - res_imgs[i].min()) / (res_imgs[i].max() - res_imgs[i].min() + 1e-8)
        mean_value = res_imgs[i].mean()
        #plot_res_imgs_distribution(res_imgs[i])
        mask = (res_imgs[i] > mean_value + 0.05).float()
        mask = dense_mask_connect(mask.unsqueeze(0))
        #mask = mask_edge_feather(mask, feather_size=4)
        mask = mask.squeeze(0).float()
        complement_value = 1 - i/timesteps
        mask = torch.where(mask < 1, complement_value, mask)
        masks.append(mask)
    for i, img in tqdm(enumerate(diff_imgs), total=len(diff_imgs)):
        # 确保去除批次维度

        img = img.squeeze(0)
        
        save_image(img, os.path.join(histogram_dir, f'noise_{i}.png'))

    noise_imgs = [] 
    for t in range(timesteps):
        noise_img = resdiffusion(diff_imgs[t], t, masks[t], noise=None, compensate=None)
        noise_imgs.append(noise_img)
    for i, img in tqdm(enumerate(noise_imgs), total=len(noise_imgs)):
        # 确保去除批次维度

        img = img.squeeze(0)
        
        save_image(img, os.path.join(residual_dir, f'noise_{i}.png'))
    for i, img in tqdm(enumerate(masks), total=len(masks)):
        # 确保去除批次维度

        img = img.squeeze(0)
        
        save_image(img, os.path.join(mask_dir, f'noise_{i}.png'))