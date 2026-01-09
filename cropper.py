import os
from PIL import Image
from multiprocessing import Pool
import glob
import matplotlib.pyplot as plt
from os import path as osp
import numpy as np
from numpy import asarray
from tqdm import tqdm

def cropper(args):
    i, path, save_dir, crop_size, thresh_size = args
    img_name, extension = osp.splitext(osp.basename(path))
    img = Image.open(path).convert('RGB')
    img = asarray(img)
    h, w, c = img.shape
    if h > w:
        img = np.rot90(img)  # 逆时针旋转90度
        # 或者使用 img = np.rot90(img, k=1)
        h, w, c = img.shape
    if h >= crop_size[1] and w >= crop_size[0]:
        # 从图像中心开始裁切
        center_x = w // 2
        center_y = h // 2
        
        # 计算裁切区域的左上角坐标
        x_start = max(0, center_x - crop_size[0] // 2)
        y_start = max(0, center_y - crop_size[1] // 2)
        
        # 确保裁切区域不超出图像边界
        if x_start + crop_size[0] <= w and y_start + crop_size[1] <= h:
            cropped_img = img[y_start:y_start+crop_size[1], x_start:x_start+crop_size[0], :]
            cropped_img_pil = Image.fromarray(cropped_img.astype('uint8'), 'RGB')
            cropped_img_pil.save(f'{save_dir}/{img_name}_cropped{extension}')

def build_cropped_dataset(paths, save_dir, crop_size, thresh_size):
    if isinstance(crop_size, int):
        crop_size = [crop_size, crop_size]
    
    def get_cropper_args():
        for i, path in enumerate(paths):
            yield i, path, save_dir, crop_size, thresh_size

    with Pool(processes=os.cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(cropper, list(get_cropper_args())), total=len(paths), desc='Cropping images'))



if __name__ == '__main__':
    cropped_img_list = []
    cropped_img_list += sorted(glob.glob('M:/github/database/DIV2K_valid_HR/*.png'))
    save_dir = 'M:/github/database/DIV2K_valid_HR_256'
    os.makedirs(save_dir, exist_ok=True)
    build_cropped_dataset(cropped_img_list, save_dir, crop_size=[256, 256], thresh_size=[256, 256])
    
            
            

