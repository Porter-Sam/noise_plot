import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

#设置字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STHeiti', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False 

# 生成信号
def generate_signal(N=1024, f1=0.1, f2=0.22, fs=1, noise_std=1):
    """
    生成包含两个正弦分量和高斯白噪声的信号
    
    参数:
        N: 信号长度
        f1: 第一个正弦分量的频率
        f2: 第二个正弦分量的频率
        noise_std: 噪声的标准差
    返回:
        x: 生成的信号
    """
    f1 = f1 / fs
    f2 = f2 / fs
    n = np.arange(N)
    # 生成两个正弦分量
    sine1 = 10 * np.sin(2 * np.pi * f1 * n + np.pi / 3)
    sine2 = 5 * np.sin(2 * np.pi * f2 * n + np.pi / 4)
    # 生成高斯白噪声
    np.random.seed(42)  # 固定随机种子，保证可重复性
    noise = np.random.normal(0, noise_std, N)
    # 合成信号
    return sine1 + sine2 + noise

# 功率谱估计
def estimate_power_spectrum(x, fs=1, method='welch', nperseg=256):
    """
    估计信号的功率谱密度
    
    参数:
        x: 输入信号
        fs: 采样频率
        method: 估计方法，可选 'periodogram' 或 'welch'
        nperseg: Welch 方法的分段长度
    返回:
        f: 频率轴
        Pxx: 功率谱密度
    """
    if method == 'periodogram':
        # 周期图法
        f, Pxx = periodogram(x, fs=fs)
    elif method == 'welch':
        # Welch 法
        f, Pxx = welch(x, fs=fs, nperseg=nperseg)
    else:
        raise ValueError(f"不支持的方法: {method}")
    return f, Pxx

# 估计频率
def estimate_frequencies(f, Pxx, num_peaks=2):
    """
    从功率谱中估计频率
    
    参数:
        f: 频率轴
        Pxx: 功率谱密度
        num_peaks: 要估计的峰值数量
    返回:
        estimated_freqs: 估计的频率
    """
    # 找到功率谱中的峰值
    peak_indices = signal.find_peaks(Pxx, height=np.max(Pxx)*0.1)[0]
    # 按功率排序
    peak_indices = sorted(peak_indices, key=lambda i: Pxx[i], reverse=True)
    # 取前 num_peaks 个峰值
    estimated_freqs = [f[i] for i in peak_indices[:num_peaks]]
    return estimated_freqs

# 主函数
def main():
    # 生成信号
    N = 1024
    f1_true, f2_true = 0.1, 0.22
    fs = 2.0  # 采样频率
    x = generate_signal(N, f1_true, f2_true, fs)
    
    # 功率谱估计
    f, Pxx = estimate_power_spectrum(x, fs, method='welch', nperseg=1024)
    
    # 估计频率
    estimated_freqs = estimate_frequencies(f, Pxx)
    
    # 打印结果
    print(f"真实频率: f1 = {f1_true}, f2 = {f2_true}")
    print(f"估计频率: f1 = {estimated_freqs[0]:.4f}, f2 = {estimated_freqs[1]:.4f}")
    
    # 绘制功率谱
    plt.figure(figsize=(10, 6))
    plt.plot(f, 10 * np.log10(Pxx))  # 转换为 dB
    plt.plot(estimated_freqs, 10 * np.log10([Pxx[np.where(f == freq)[0][0]] for freq in estimated_freqs]), 'ro')
    plt.xlabel('频率 (Hz)')
    plt.ylabel('功率谱密度 (dB/Hz)')
    plt.title('信号的功率谱估计')
    plt.grid(True)
    plt.xlim(0, fs / 2)  # 显示 0 到奈奎斯特频率
    plt.legend(['功率谱', '估计的频率'])
    plt.show()
    plt.savefig('power_spectrum.png', dpi=300, bbox_inches='tight') 

    

def periodogram(x, fs=1.0):
    N = len(x)
    # 计算FFT
    X = np.fft.fft(x)
    # 计算功率谱密度
    Pxx = np.abs(X)**2 / (N * fs)
    # 频率轴
    f = np.fft.fftfreq(N, 1/fs)
    # 只返回正频率部分
    return f[:N//2], Pxx[:N//2]



def welch(x, fs=1.0, nperseg=256):
    N = len(x)
    
    # 默认使用50%重叠
    noverlap = nperseg // 2
    nstep = nperseg - noverlap
    
    # 计算段数
    nsegments = (N - noverlap) // nstep
    
    # 使用汉宁窗
    window = np.hanning(nperseg)
    U = np.sum(window**2)  # 窗口能量归一化因子
    
    # 存储各段的周期图
    Pxx_segments = []
    
    for k in range(nsegments):
        # 提取数据段
        start = k * nstep
        end = start + nperseg
        if end <= N:
            segment = x[start:end]
            
            # 加窗
            windowed_segment = segment * window
            
            # 计算FFT和周期图
            fft_seg = np.fft.fft(windowed_segment)
            Pxx_seg = np.abs(fft_seg)**2 / U
            Pxx_segments.append(Pxx_seg)
    
    # 平均所有段的结果
    Pxx_avg = np.mean(Pxx_segments, axis=0)
    
    # 频率轴
    f = np.fft.fftfreq(nperseg, 1/fs)
    
    return f[:nperseg//2], Pxx_avg[:nperseg//2]


if __name__ == "__main__":
    main()