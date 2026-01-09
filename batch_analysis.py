# batch_analysis.py
import os
import torch
from PIL import Image
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
from analysis import worker, create_experiment_dir, Trans
import matplotlib.pyplot as plt

def batch_analyze_images(image_paths, timesteps=96, sp_delta=8, mode='MAX'):
    """
    批量分析多张图片
    
    Args:
        image_paths: 图片路径列表
        timesteps: 时间步数
        sp_delta: 空间增量
    """
    
    # 创建实验目录
    experiment_dir = create_experiment_dir("batch_experiments")
    main_histogram_dir = os.path.join(experiment_dir, 'histograms')
    main_residual_dir = os.path.join(experiment_dir, 'residuals')
    os.makedirs(main_histogram_dir, exist_ok=True)
    os.makedirs(main_residual_dir, exist_ok=True)
    
    # 总体mean值存储
    all_mean_values = []
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # 批量处理图片
    for idx, img_path in enumerate(tqdm(image_paths, desc="Processing images")):
        try:
            # 创建每个图片的子目录
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            img_histogram_dir = os.path.join(main_histogram_dir, img_name)
            img_residual_dir = os.path.join(main_residual_dir, img_name)
            os.makedirs(img_histogram_dir, exist_ok=True)
            os.makedirs(img_residual_dir, exist_ok=True)
            
            # 加载图片
            image = Image.open(img_path).convert('RGB')
            hr_img = transform(image)
            
            # 处理图片
            _, res_imgs, mean = worker(hr_img, timesteps, sp_delta, save_dir=img_histogram_dir, mode=mode)
            
            # 保存残差图像
            for i, img in enumerate(res_imgs):
                img_array = Trans.single_tensor2img(img)
                img_pil = Image.fromarray(img_array)
                img_pil.save(os.path.join(img_residual_dir, f'residual_{i}.png'))
            
            # 保存mean值
            df = pd.DataFrame({'mean_value': mean})
            df.to_csv(os.path.join(experiment_dir, f'{img_name}_mean.csv'), index=False)
            
            # 收集总体数据
            all_mean_values.append({
                'image_name': img_name,
                'mean_values': mean
            })
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    # 绘制所有图片的mean值对比图
    plt.figure(figsize=(12, 6))
    for data in all_mean_values:
        plt.plot(range(len(data['mean_values'])), data['mean_values'], 
                marker='o', linewidth=2, markersize=2, label=data['image_name'])
    
    plt.title('Mean Values Across Residual Levels (All Images)')
    plt.xlabel('Residual Level')
    plt.ylabel('Mean Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(experiment_dir, 'all_mean_values_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存总体统计信息
    summary_data = []
    for data in all_mean_values:
        summary_data.append({
            'image_name': data['image_name'],
            'mean_of_means': sum(data['mean_values']) / len(data['mean_values'])
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(experiment_dir, 'summary.csv'), index=False)
    
    print(f"Batch analysis completed. Results saved to: {experiment_dir}")
    return experiment_dir

def get_image_paths_from_directory(directory):
    """
    从目录中获取所有图片路径
    
    Args:
        directory: 图片目录路径
        
    Returns:
        图片路径列表
    """
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    image_paths = []
    
    for filename in os.listdir(directory):
        if filename.lower().endswith(valid_extensions):
            image_paths.append(os.path.join(directory, filename))
    
    return image_paths

if __name__ == '__main__':
    # 方式1: 直接指定图片路径列表
    image_paths = [
        "E:/YuShihang/database/cropped_img1024_center/0156_cropped_.png",
        "E:/YuShihang/database/cropped_img1024_center/0402_cropped_.png",
        "E:/YuShihang/database/cropped_img1024_center/0431_cropped_.png",
        "E:/YuShihang/database/cropped_img1024_center/0063_cropped_.png",
        "E:/YuShihang/database/cropped_img1024_center/0065_cropped_.png",
        "E:/YuShihang/database/cropped_img1024_center/0057_cropped_.png",
        "E:/YuShihang/database/cropped_img1024_center/0565_cropped_.png",
        "E:/YuShihang/database/cropped_img1024_center/0725_cropped_.png"
        
        # 添加更多图片路径
        # "path/to/another/image.png",
    ]
    
    # 方式2: 从目录自动获取所有图片
    # image_dir = "E:/YuShihang/dataset/cropped_img"
    # image_paths = get_image_paths_from_directory(image_dir)
    
    # 执行批量分析
    batch_analyze_images(image_paths, timesteps=6, sp_delta=64, mode='Linear')