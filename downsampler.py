from PIL import Image
from modules import Trans
import os
from os import path as osp
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm
from noise_mix import noise_mix

def tensor_to_image(tensor):
    """
    将tensor转换为图像数组，确保正确的数据范围和类型
    
    Args:
        tensor: 形状为(C, H, W)的torch.Tensor，值域可能超出[0,1]
    
    Returns:
        numpy.ndarray: 形状为(H, W, C)的uint8数组
    """
    # 首先确保tensor在CPU上
    tensor = tensor.cpu()
    
    # 将值限制在[0,1]范围内
    tensor = tensor.clamp(0, 1)
    
    # 转换为[0,255]范围并转换为uint8类型
    img_array = (tensor * 255).permute(1, 2, 0).numpy().astype('uint8')
    
    return img_array

def resdiffusion(hr_img, timesteps, sp_delta, save_dir, mode='bilinear', step='linear'):
    diff_imgs = []
    diff_imgs.append(hr_img)
    current_img = hr_img
    hr_img = hr_img.unsqueeze(0) #填充维度满足（N,C,W,H）
    alpha = [0.2, 0.2, 0.2]
    sigma = [1, 1, 1]
    for i in tqdm(range(timesteps)):
        coord_shape = (current_img.shape[0], current_img.shape[1]-sp_delta, current_img.shape[2]-sp_delta)
        current_img = current_img.unsqueeze(0) #填充维度满足（N,C,W,H）
        if step == 'linear':
            diff_img = F.interpolate(current_img, scale_factor=1/2, mode=mode) #下采样
        else:
            diff_img = F.interpolate(hr_img, scale_factor=1/2, mode=mode)
        diff_img = diff_img.squeeze(0) #去掉填充的维度
        diff_img_e = noise_mix(diff_img, alpha=alpha[i], mu=0, sigma=sigma[i], schedule='VE')
        diff_imgs.append(diff_img_e)
        current_img = diff_img

        save_img = tensor_to_image(diff_img_e)
        save_img = Image.fromarray(save_img)
        save_img.save(os.path.join(save_dir, f'diff_img_{i}_{step}_{mode}.png'))

    return diff_imgs

if __name__ == '__main__':
    path = "M:/github/database/DIV2K_valid_HR_256/0808_cropped.png"
    img_name, extension = osp.splitext(osp.basename(path))
    image = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    hr_img = transform(image)
    hr_img = hr_img
    save_dir = 'diff_imgs_2'
    step = 'linear'
    mode = 'bicubic'
    img_dir = os.path.join(save_dir, f'{img_name}_{step}_{mode}')
    os.makedirs(img_dir, exist_ok=True)
    
    diff_imgs = resdiffusion(hr_img, timesteps=3, sp_delta=64, save_dir=img_dir, mode=mode, step=step)