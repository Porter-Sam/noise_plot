import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
#from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from edsr import make_edsr_baseline
import torch
from torch.nn import functional as F





class Feature4DMatcher:
    """处理4D特征(b, c, h, w)的高低分辨率特征匹配器"""
    
    def __init__(self):
        """初始化4D特征匹配器"""
        self.high_features = None  # 高分辨率特征 (b, c_high, h_high, w_high)
        self.low_features = None   # 低分辨率特征 (b, c_low, h_low, w_low)
        self.high_channels = None  # 高分辨率通道数
        self.low_channels = None   # 低分辨率通道数
        self.batch_size = None     # 批次大小
        self.similarity_matrix = None  # 相似度矩阵 (c_high, c_low)
        
        # 特征名称
        self.high_names = None
        self.low_names = None
        self.high_feat_gen = make_edsr_baseline(n_resblocks=16, n_feats=64, res_scale=1,
                                               scale=2, no_upsampling=True, rgb_range=1)
        self.low_feat_gen = make_edsr_baseline(n_resblocks=16, n_feats=64, res_scale=1,
                                               scale=2, no_upsampling=True, rgb_range=1)
        

    def gen_feats(self, hr_img, lr_img):
        high_res_features = self.high_feat_gen(hr_img).detach().numpy()
        low_res_features = self.low_feat_gen(lr_img).detach().numpy()
        return high_res_features, low_res_features
        
    def set_features(self, high_res_features, low_res_features,
                    high_channel_names=None, low_channel_names=None):
        """
        设置高低分辨率4D特征
        
        参数:
            high_res_features: 高分辨率特征，形状为 (b, c_high, h_high, w_high)
            low_res_features: 低分辨率特征，形状为 (b, c_low, h_low, w_low)
            high_channel_names: 高分辨率通道名称列表
            low_channel_names: 低分辨率通道名称列表
        """
        # 验证输入形状
        if len(high_res_features.shape) != 4 or len(low_res_features.shape) != 4:
            raise ValueError("特征必须是4D数组，形状为 (b, c, h, w)")
            
        if high_res_features.shape[0] != low_res_features.shape[0]:
            raise ValueError("高低分辨率特征的批次大小必须一致")
            
        # 存储特征
        self.high_features = high_res_features
        self.low_features = low_res_features
        
        # 提取特征维度信息
        self.batch_size = high_res_features.shape[0]
        self.high_channels = high_res_features.shape[1]
        self.low_channels = low_res_features.shape[1]
        
        # 设置通道名称
        self.high_names = high_channel_names if high_channel_names else \
                         [f'hr_channel{i+1}' for i in range(self.high_channels)]
        self.low_names = low_channel_names if low_channel_names else \
                        [f'lr_channel{i+1}' for i in range(self.low_channels)]
        
        # 打印特征信息
        print(f"高分辨率特征: 批次={self.batch_size}, 通道={self.high_channels}, "
              f"尺寸={high_res_features.shape[2:]}")
        print(f"低分辨率特征: 批次={self.batch_size}, 通道={self.low_channels}, "
              f"尺寸={low_res_features.shape[2:]}")
              
        if self.high_channels <= self.low_channels:
            warnings.warn("高分辨率特征通道数应大于低分辨率特征通道数")
            
        return self
    
    def preprocess_features(self, normalize_per_channel=True):
        """
        预处理4D特征
        
        参数:
            normalize_per_channel: 是否对每个通道进行标准化
        """
        if normalize_per_channel:
            # 对每个通道进行标准化 (b, c, h, w) -> 按通道标准化
            for b in range(self.batch_size):
                for c in range(self.high_channels):
                    channel = self.high_features[b, c]
                    mean = np.mean(channel)
                    std = np.std(channel) + 1e-8  # 防止除零
                    self.high_features[b, c] = (channel - mean) / std
                
                for c in range(self.low_channels):
                    channel = self.low_features[b, c]
                    mean = np.mean(channel)
                    std = np.std(channel) + 1e-8
                    self.low_features[b, c] = (channel - mean) / std
            
            print("特征已按通道标准化")
        
        return self
    
    def _flatten_spatial_dimensions(self, features):
        """
        将空间维度展平 (b, c, h, w) -> (b, c, h*w)
        
        参数:
            features: 4D特征数组
        
        返回:
            展平后的3D特征数组 (b, c, h*w)
        """
        b, c, h, w = features.shape
        return features.reshape(b, c, h * w)
    
    def _compute_channel_similarity(self, high_chan, low_chan, method='cosine'):
        """
        计算两个通道（包含所有批次）的相似度
        
        参数:
            high_chan: 高分辨率通道特征，形状为 (b, h_high*w_high)
            low_chan: 低分辨率通道特征，形状为 (b, h_low*w_low)
            method: 相似度计算方法
        
        返回:
            平均相似度分数
        """
        batch_similarities = []
        
        # 对每个批次计算相似度
        for b in range(self.batch_size):
            # 获取当前批次的特征向量
            high_vec = high_chan[b]
            low_vec = low_chan[b]
            
            # 确保向量长度一致（如不一致则截断较长的）
            min_len = min(len(high_vec), len(low_vec))
            high_vec = high_vec[:min_len]
            low_vec = low_vec[:min_len]
            
            # 计算相似度
            if method == 'cosine':
                sim = 1 - cosine(high_vec, low_vec)
            elif method == 'pearson':
                sim, _ = pearsonr(high_vec, low_vec)
            else:
                raise ValueError(f"不支持的相似度方法: {method}")
                
            batch_similarities.append(sim)
        
        # 返回批次平均相似度
        return np.mean(batch_similarities)
    
    def calculate_similarity(self, method='cosine', mode=None):
        """
        计算高分辨率与低分辨率特征通道之间的相似度
        
        参数:
            method: 相似度计算方法，可选 'cosine' 或 'pearson'
        
        返回:
            相似度矩阵 (c_high, c_low)
        """
        # 展平空间维度
        high_flat = self._flatten_spatial_dimensions(self.high_features)  # (b, c_high, h_high*w_high)
        low_flat = self._flatten_spatial_dimensions(self.low_features)    # (b, c_low, h_low*w_low)
        
        # 初始化相似度矩阵
        self.similarity_matrix = np.zeros((self.high_channels, self.low_channels))
        
        # 计算每个通道对的相似度
        print(f"正在计算{method}相似度矩阵...")
        for i in range(self.high_channels):
            # 获取高分辨率第i个通道的所有批次
            high_chan = high_flat[:, i, :]  # (b, h_high*w_high)
            
            for j in range(self.low_channels):
                # 获取低分辨率第j个通道的所有批次
                low_chan = low_flat[:, j, :]  # (b, h_low*w_low)
                
                # 计算相似度
                self.similarity_matrix[i, j] = self._compute_channel_similarity(
                    high_chan, low_chan, method
                )
        
        # 转换为DataFrame便于查看
        self.similarity_matrix = pd.DataFrame(
            self.similarity_matrix,
            index=self.high_names,
            columns=self.low_names
        )
        count_above_08 = (self.similarity_matrix.values > 0.4).sum()
        total_count = self.similarity_matrix.size
        percentage = (count_above_08 / total_count) * 100

        if mode == 'mean':
            return self.similarity_matrix.values.mean()
        elif mode == 'percent':
            print(f"{percentage:.2f}% 的通道的相似度分数高于0.8")
            return count_above_08, percentage
        elif mode == 'weighted':
            return self.similarity_matrix.mean(axis=0).mean()
        else:
            return self.similarity_matrix
    
    def find_best_matches(self, top_n=3):
        """
        为每个低分辨率通道找到最匹配的高分辨率通道
        
        参数:
            top_n: 每个低分辨率通道返回的最佳匹配数量
        
        返回:
            字典，键为低分辨率通道名，值为匹配结果DataFrame
        """
        if self.similarity_matrix is None:
            raise ValueError("请先计算相似度矩阵")
            
        matches = {}
        for low_chan in self.low_names:
            # 按相似度降序排序
            sorted_matches = self.similarity_matrix[low_chan].sort_values(ascending=False)
            # 取前top_n个
            matches[low_chan] = sorted_matches.head(top_n).reset_index()
            matches[low_chan].columns = ['高分辨率通道', '相似度分数']
        
        return matches
    
    def find_top_high_channels(self, top_n=5):
        """
        找到与低分辨率通道整体最相似的高分辨率通道
        
        参数:
            top_n: 返回的高分辨率通道数量
        
        返回:
            包含高分辨率通道及其平均相似度的Series
        """
        if self.similarity_matrix is None:
            raise ValueError("请先计算相似度矩阵")
            
        # 计算每个高分辨率通道与所有低分辨率通道的平均相似度
        high_channel_scores = self.similarity_matrix.mean(axis=1)
        # 按平均相似度降序排序
        top_high_channels = high_channel_scores.sort_values(ascending=False).head(top_n)
        
        return top_high_channels
    
    def visualize_similarity(self, figsize=(12, 10), cmap="YlOrRd"):
        """可视化高低分辨率通道相似度矩阵"""
        if self.similarity_matrix is None:
            raise ValueError("请先计算相似度矩阵")
            
        plt.figure(figsize=figsize)
        
        # 绘制热力图
        sns.heatmap(
            self.similarity_matrix,
            annot=False,
            cmap=cmap,
            vmin=-1 if 'pearson' in str(self.similarity_matrix) else 0,
            vmax=1,
            cbar=True,
            cbar_kws={"label": "similarity score"}
        )
        
        plt.title("Similarity matrix of HR and LR feature channels", fontsize=16)
        plt.tight_layout()
        return plt
    
    def visualize_channel_matches(self, channel_idx, top_n=3, figsize=(15, 5)):
        """
        可视化特定低分辨率通道与其最佳匹配的高分辨率通道的特征图
        
        参数:
            channel_idx: 低分辨率通道索引
            top_n: 显示的最佳匹配数量
            figsize: 图像大小
        """
        if self.similarity_matrix is None:
            raise ValueError("请先计算相似度矩阵")
            
        if channel_idx < 0 or channel_idx >= self.low_channels:
            raise ValueError(f"低分辨率通道索引超出范围，有效范围: 0-{self.low_channels-1}")
            
        # 获取最佳匹配
        low_chan_name = self.low_names[channel_idx]
        best_matches = self.similarity_matrix[low_chan_name].sort_values(ascending=False).head(top_n)
        high_indices = [self.high_names.index(name) for name in best_matches.index]
        
        # 创建图像
        fig, axes = plt.subplots(1, top_n + 1, figsize=figsize)
        
        # 显示低分辨率通道
        axes[0].imshow(self.low_features[0, channel_idx], cmap='viridis')
        axes[0].set_title(f'低分辨率通道 {channel_idx+1}\n{low_chan_name}')
        axes[0].axis('off')
        
        # 显示匹配的高分辨率通道
        for i, high_idx in enumerate(high_indices):
            axes[i+1].imshow(self.high_features[0, high_idx], cmap='viridis')
            axes[i+1].set_title(f'匹配 {i+1}: 高通道 {high_idx+1}\n相似度: {best_matches.iloc[i]:.4f}')
            axes[i+1].axis('off')
        
        plt.tight_layout()
        return plt
    
    def forward(self, hr_feat, lr_feat, heat_map=None):
        self.set_features(hr_feat, lr_feat)
        self.preprocess_features(normalize_per_channel=True)
        if heat_map is not None:
            sim_matrix = self.calculate_similarity()
            plt1 = matcher.visualize_similarity()
            return plt1
        else:
            sim_matrix = self.calculate_similarity(mode='percent')
            return sim_matrix


# 示例用法
if __name__ == "__main__":
    # 读取图像
    
    hr_img1 = plt.imread('/datasdb/ysh/dataset/DIV2K_valid_HR_1080p/0803_cropped.png')
    hr_img2 = plt.imread('/datasdb/ysh/dataset/DIV2K_valid_HR_1080p/0805_cropped.png')
    

    # 确保图像在正确范围内
    if hr_img1.max() > 1:
        hr_img1 = hr_img1.astype(np.float32) / 255.0
    if hr_img2.max() > 1:
        hr_img2 = hr_img2.astype(np.float32) / 255.0

    # 添加批次维度并转换为 (B, C, H, W)
    hr_img1 = np.transpose(hr_img1, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    hr_img1 = np.expand_dims(hr_img1, axis=0)   # (1, C, H, W)
    hr_img1 = torch.from_numpy(hr_img1).float()
    hr_img2 = np.transpose(hr_img2, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    hr_img2 = np.expand_dims(hr_img2, axis=0)   # (1, C, H, W)
    hr_img2 = torch.from_numpy(hr_img2).float()
    hr_img = torch.cat([hr_img1, hr_img2], dim=0)  # (2, C, H, W)

    hr_img_ = F.interpolate(hr_img, scale_factor=0.5, mode='bicubic', align_corners=False)
    lr_img = F.interpolate(hr_img_, scale_factor=0.5, mode='bicubic', align_corners=False)
    lr_img_ = F.interpolate(lr_img, scale_factor=2, mode='bicubic', align_corners=False)
   



    # 创建4D特征匹配器
    matcher = Feature4DMatcher()
    high_res_features, low_res_features = matcher.gen_feats(hr_img_, lr_img_)
    out = matcher.forward(high_res_features, low_res_features)

    '''
    # 生成特征 (b, c, h, w)
    high_res_features, low_res_features = matcher.gen_feats(hr_img_, lr_img_)
    
    # 设置特征
    matcher.set_features(high_res_features, low_res_features)
    
    # 预处理特征
    matcher.preprocess_features(normalize_per_channel=True)
    
    # 计算余弦相似度（也可尝试 'pearson'）
    matcher.calculate_similarity(method='cosine')
    
    # 为每个低分辨率通道找到最佳匹配的3个高分辨率通道
    best_matches = matcher.find_best_matches(top_n=3)
    
    # 打印匹配结果
    print("\n每个低分辨率通道的最佳高分辨率通道匹配:")
    for low_chan, matches in best_matches.items():
        print(f"\n{low_chan}:")
        print(matches.to_string(index=False))
    
    # 找到与低分辨率通道整体最相似的前5个高分辨率通道
    top_high = matcher.find_top_high_channels(top_n=5)
    print("\n与低分辨率通道整体最相似的高分辨率通道:")
    print(top_high)
    
    # 可视化相似度矩阵
    plt1 = matcher.visualize_similarity()
    plt1.savefig('4d_feature_similarity_matrix.png')
    print("\n特征通道相似度矩阵热力图已保存为 4d_feature_similarity_matrix.png")
    
    # 可视化第一个低分辨率通道的最佳匹配
    plt2 = matcher.visualize_channel_matches(channel_idx=0, top_n=3)
    plt2.savefig('channel_matches_visualization.png')
    print("通道匹配可视化已保存为 channel_matches_visualization.png")
    
    # 显示图形
    plt.show()
    '''