import torch
import math

class CellEncoding(torch.nn.Module):
    def __init__(self, num_frequencies=4):
        super().__init__()
        self.num_frequencies = num_frequencies

    def forward(self, current_res, H, W, device):
        # 1. 计算 Cell 标量
        cell = 1.0 / current_res
        cell = -torch.log2(cell)
        
        # 2. 使用Transformer正弦位置编码方式
        encoding = self.get_sinusoid_encoding_table(1, self.num_frequencies*2, cell.item(), device)
        
        return cell, encoding.squeeze(0)

    def get_sinusoid_encoding_table(self, batch_size, d_model, pos, device):
        """使用Transformer正弦位置编码公式生成编码"""
        pe = torch.zeros(batch_size, d_model, device=device)
        
        position = torch.full((batch_size, 1), pos, device=device)
        
        # 计算div_term用于正弦/余弦编码
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device, dtype=torch.float) *
            (-math.log(10000.0) / d_model)
        )
        
        # 对偶数位置应用sin，奇数位置应用cos
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置(0,2,4,...)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term[:, :-1])  # 奇数位置(1,3,5,...)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置(1,3,5,...)
            
        return pe

if __name__ == '__main__':
    cell_encoding = CellEncoding()
    current_res = torch.linspace(64, 512, 9)
    for i in range(len(current_res)):
        cell, encoding = cell_encoding(current_res[i], 256, 256, 'cpu')
        print(f"current_res: {current_res[i]}, cell: {cell}, encoding: {encoding}")