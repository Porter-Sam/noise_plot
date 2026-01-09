import numpy as np
import matplotlib.pyplot as plt

def plot_function(func, x_range, num_points=1000, title="Function Plot", 
                  xlabel="x", ylabel="y", figsize=(10, 6), grid=True):
    """
    通用函数绘制器
    
    参数:
        func: 要绘制的函数（可调用对象）
        x_range: x轴范围，格式为 (xmin, xmax)
        num_points: 采样点数量
        title: 图表标题
        xlabel: x轴标签
        ylabel: y轴标签
        figsize: 图表大小
        grid: 是否显示网格
    
    返回:
        matplotlib.pyplot对象
    """
    # 生成x值
    x = np.linspace(x_range[0], x_range[1], num_points)
    
    # 计算y值
    y = func(x)
    
    # 创建图表
    plt.figure(figsize=figsize)
    plt.plot(x, y, linewidth=2)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    if grid:
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt

# 示例1: 绘制 y = 1 - e^(-10^x)
def custom_function(x):
    return 1 - np.exp(-np.power(10, x))

# 绘制指定函数
plt1 = plot_function(
    func=custom_function,
    x_range=(-2, 2),  # x的范围
    title=r"$y = 1 - e^{-10^x}$",
    xlabel="x",
    ylabel="y"
)
plt1.show()

# 示例2: 绘制其他常见函数
# 正弦函数
plt2 = plot_function(
    func=lambda x: np.sin(x),
    x_range=(0, 4*np.pi),
    title="Sine Function",
    xlabel="x",
    ylabel="sin(x)"
)
plt2.show()

# 示例3: 多项式函数
plt3 = plot_function(
    func=lambda x: x**3 - 2*x**2 + x - 1,
    x_range=(-3, 3),
    title="Polynomial Function: $y = x^3 - 2x^2 + x - 1$",
    xlabel="x",
    ylabel="y"
)
plt3.show()

# 使用方法：
# 1. 定义你的函数（可以是普通函数或lambda表达式）
# 2. 调用 plot_function 函数并传入相应参数
# 3. 调用返回的plt对象的show()方法显示图像


if __name__ == "__main__":
    plt1 = plot_function(
    func=lambda x: 1 - np.exp(-np.power(10, x)),  # 您要绘制的函数
    x_range=(-2, 2),                              # x的取值范围
    title=r"$y = 1 - e^{-10^x}$",                 # 图标题
    xlabel="x",                                   # x轴标签
    ylabel="y"                                    # y轴标签
    )
    plt.show()