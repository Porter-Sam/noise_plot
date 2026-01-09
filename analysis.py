import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from modules import Trans
from plot_histogram import *
os.environ["OMP_NUM_THREADS"] = "1"

from datetime import datetime

def create_experiment_dir(base_dir="experiments"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_dir, timestamp)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir



def cropper(raw_img):
    if raw_img.shape[1] > raw_img.shape[2]:
        crop_size = raw_img.shape[2]
        h_offset= (raw_img.shape[1] - raw_img.shape[2]) // 2
        w_offset= 0
    elif raw_img.shape[1] < raw_img.shape[2]:
        crop_size = raw_img.shape[1]
        h_offset= 0
        w_offset= (raw_img.shape[2] - raw_img.shape[1]) // 2
    else:
        crop_size = raw_img.shape[1]
        h_offset= 0
        w_offset= 0
        

    cropped_img = raw_img[:, h_offset:h_offset+crop_size, w_offset:w_offset+crop_size]

    return cropped_img


def resdiffusion(hr_img, timesteps, sp_delta, mode='bilinear'):
    diff_imgs = []
    diff_imgs.append(hr_img)
    current_img = hr_img
    hr_img = hr_img.unsqueeze(0) #填充维度满足（N,C,W,H）
    for i in range(timesteps):
        coord_shape = (current_img.shape[0], current_img.shape[1]-sp_delta, current_img.shape[2]-sp_delta)
        current_img = current_img.unsqueeze(0) #填充维度满足（N,C,W,H）
        diff_img = F.interpolate(hr_img, size=(coord_shape[1], coord_shape[2]), mode=mode) #下采样
        diff_img = diff_img.squeeze(0) #去掉填充的维度
        diff_imgs.append(diff_img)
        current_img = diff_img

    return diff_imgs

def res_calculate(diff_imgs, mode='MIN'):
    res_imgs = []
    for i in range(len(diff_imgs)-1):
        img_lr = diff_imgs[i+1].unsqueeze(0)
        img_hr = diff_imgs[i].unsqueeze(0)
        lr = diff_imgs[-1].unsqueeze(0)
        if mode == 'MAX':
            lr_img = F.interpolate(img_lr, size=(diff_imgs[0].shape[1], diff_imgs[0].shape[2]), mode='bicubic')
            hr_img = F.interpolate(img_hr, size=(diff_imgs[0].shape[1], diff_imgs[0].shape[2]), mode='bicubic')
            lr_img = lr_img.squeeze(0)
            hr_img = hr_img.squeeze(0)
            res_img = hr_img - lr_img
        elif mode == 'MIN':
            lr_img = F.interpolate(lr, size=(diff_imgs[i].shape[1], diff_imgs[i].shape[2]), mode='bicubic')
            lr_img = lr_img.squeeze(0)
            res_img = diff_imgs[i] - lr_img
        else:
            lr_img = F.interpolate(img_lr, size=(diff_imgs[i].shape[1], diff_imgs[i].shape[2]), mode='bicubic')
            lr_img = lr_img.squeeze(0)
            res_img = diff_imgs[i] - lr_img
        res_imgs.append(res_img)
    
    return res_imgs

def worker(hr_img, timesteps=8, sp_delta=125, save_dir='histograms', mode='MAX'):
    cropped_img = cropper(hr_img)
    diff_imgs = resdiffusion(cropped_img, timesteps, sp_delta)
    res_imgs = res_calculate(diff_imgs, mode=mode)
    #plot_diffusion_histograms(diff_imgs)


    '''
    diff_imgs[-1] = diff_imgs[-1].unsqueeze(0)
    lr_img = F.interpolate(diff_imgs[-1], size=(hr_img.shape[1], hr_img.shape[2]), mode='bicubic')
    lr_img = lr_img.squeeze(0)
    full_res = hr_img - lr_img
    full_mean = plot_single_histogram(i, full_res, save_dir=save_dir)
    propotion = [m / full_mean for m in mean]
    '''
    return diff_imgs, res_imgs
 
def plot_mean_values(mean_values, save_dir, save_name):
    """
    绘制mean值随层级变化的图表
    
    Args:
        mean_values: mean值列表
    """
    plt.figure(figsize=(12, 6))
    
    # 创建x轴数据（层级）
    levels = range(len(mean_values))
    
    # 绘制折线图
    plt.plot(levels, mean_values, marker='o', linewidth=2, markersize=4)
    
    # 添加图表信息
    plt.title('Mean Values Across Residual Levels')
    plt.xlabel('Residual Level')
    plt.ylabel('Mean Value')
    plt.grid(True, alpha=0.3)
    
    # 保存图表
    save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, save_name), dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    path = "E:/YuShihang/database/cropped_img1024_center/0362_cropped_.png"
    image = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    experiment_dir = create_experiment_dir()
    save_dir = experiment_dir
    histogram_dir = os.path.join(experiment_dir, 'histograms')
    residual_dir = os.path.join(experiment_dir, 'residuals')
    os.makedirs(residual_dir, exist_ok=True) 

    hr_img = transform(image)
    hr_img = hr_img
    _, res_imgs, mean = worker(hr_img, save_dir=histogram_dir)
    df = pd.DataFrame({'mean_value': mean})
    df.to_csv(os.path.join(save_dir, 'mean.csv'), index=False)
    for i, img in tqdm(enumerate(res_imgs), total=len(res_imgs)):
        img = Trans.single_tensor2img(img)
        img = Image.fromarray(img)
        img.save(os.path.join(residual_dir, f'residual_{i}.png'))

    plot_mean_values(mean, save_dir=save_dir, save_name='mean.png')
    #plot_mean_values(propotion, save_dir=save_dir, save_name='propotion.png')
