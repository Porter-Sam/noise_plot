import matplotlib.pyplot as plt
import numpy as np
import os

def plot_diffusion_histograms(diff_imgs, bins=256):
    """
    绘制扩散图像序列的直方图对比
    
    Args:
        diff_imgs: 扩散图像列表
        bins: 直方图bins数量
    """
    n_images = len(diff_imgs)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(min(n_images, 6)):
        img_array = diff_imgs[i].cpu().numpy()
        img_flat = img_array.flatten()
        
        axes[i].hist(img_flat, bins=bins, alpha=0.7, color='blue')
        axes[i].set_title(f'Diffusion Level {i}')
        axes[i].set_xlabel('Pixel Value')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(min(n_images, 6), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_residual_histograms(res_imgs, bins=256):
    """
    绘制残差图像的直方图
    
    Args:
        res_imgs: 残差图像列表
        bins: 直方图bins数量
    """
    n_residuals = len(res_imgs)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for i in range(min(n_residuals, 9)):
        img_array = res_imgs[i].cpu().numpy()
        img_flat = img_array.flatten()
        
        axes[i].hist(img_flat, bins=bins, alpha=0.7, color='green')
        axes[i].set_title(f'Residual Level {i}')
        axes[i].set_xlabel('Residual Value')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)
        
        # 添加均值线
        mean_val = np.mean(img_flat)
        axes[i].axvline(mean_val, color='red', linestyle='--', 
                       label=f'Mean: {mean_val:.4f}')
        axes[i].legend()
    
    # 隐藏多余的子图
    for i in range(min(n_residuals, 9), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    save_dir = 'histograms'
    os.makedirs(save_dir, exist_ok=True)
    # 保存直方图
    plt.savefig(os.path.join(save_dir, 'residual_histograms.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_single_histogram(i, img, title="Image Histogram", bins=256, color='blue', save_dir='histograms'):
    """
    绘制单张图像的直方图
    
    Args:
        img: 图像数据 (tensor或numpy数组)
        title: 图像标题
        bins: 直方图bins数量
        color: 直方图颜色
    """
    # 转换为numpy数组
    if hasattr(img, 'cpu'):
        img_array = img.cpu().numpy()
    else:
        img_array = img
    
    # 展平图像数据
    img_flat = img_array.flatten()
    
    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(img_flat, bins=bins, alpha=0.7, color=color)
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # 添加均值线
    mean_val = np.mean(np.abs(img_flat))
    plt.axvline(mean_val, color='red', linestyle='--', 
               label=f'Mean: {mean_val:.4f}')
    plt.legend()
    
    plt.tight_layout()
    save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)
    # 保存直方图
    plt.savefig(os.path.join(save_dir, f'residual_histograms{i}.png'), dpi=300, bbox_inches='tight')
    #plt.show()

    return mean_val

def plot_compare_histograms(original_img, diff_imgs, res_imgs, level=0):
    """
    对比原始图像、扩散图像和残差图像的直方图
    
    Args:
        original_img: 原始图像
        diff_imgs: 扩散图像列表
        res_imgs: 残差图像列表
        level: 要对比的层级
    """
    if level >= len(diff_imgs) or level >= len(res_imgs):
        print(f"Level {level} out of range")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 原始图像（在指定层级）
    orig_array = diff_imgs[level].cpu().numpy()
    orig_flat = orig_array.flatten()
    axes[0].hist(orig_flat, bins=256, alpha=0.7, color='blue')
    axes[0].set_title(f'Original Image (Level {level})')
    axes[0].set_xlabel('Pixel Value')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)
    
    # 对应层级的下一级扩散图像
    if level + 1 < len(diff_imgs):
        next_array = diff_imgs[level + 1].cpu().numpy()
        next_flat = next_array.flatten()
        axes[1].hist(next_flat, bins=256, alpha=0.7, color='orange')
        axes[1].set_title(f'Diffused Image (Level {level + 1})')
        axes[1].set_xlabel('Pixel Value')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
    
    # 对应的残差图像
    res_array = res_imgs[level].cpu().numpy()
    res_flat = res_array.flatten()
    axes[2].hist(res_flat, bins=256, alpha=0.7, color='green')
    axes[2].set_title(f'Residual Image (Level {level})')
    axes[2].set_xlabel('Residual Value')
    axes[2].set_ylabel('Frequency')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()