import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pointnet2_cls import PointNet2Encoder  # 请确保路径正确或用相对导入替代

class FoldingLayer(nn.Module):
    """
    FoldingNet中的折叠操作
    """
    def __init__(self, in_channel: int, out_channels: list):
        super(FoldingLayer, self).__init__()

        layers = []
        for oc in out_channels[:-1]:
            conv = nn.Conv1d(in_channel, oc, 1)
            bn = nn.BatchNorm1d(oc)
            active = nn.ReLU(inplace=True)
            layers.extend([conv, bn, active])
            in_channel = oc
        out_layer = nn.Conv1d(in_channel, out_channels[-1], 1)
        layers.append(out_layer)
        
        self.layers = nn.Sequential(*layers)

    def forward(self, grids, codewords):
        """
        参数:
            grids: 重塑的2D网格或中间重建的点云 [B, 2, N] 或 [B, 3, N]
            codewords: 编码器输出的特征 [B, latent_dim, N]
        """
        # 连接
        x = torch.cat([grids, codewords], dim=1)
        # 共享MLP
        x = self.layers(x)
        
        return x

class FoldingNetDecoder(nn.Module):
    def __init__(self, grid_size=45, latent_dim=512):
        super(FoldingNetDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        self.num_points = grid_size * grid_size  # M = 2025

        # 创建2D网格，使用更小的范围 [-0.3, 0.3] 而不是 [-1, 1]
        xx = np.linspace(-0.3, 0.3, grid_size, dtype=np.float32)
        yy = np.linspace(-0.3, 0.3, grid_size, dtype=np.float32)
        self.grid = np.meshgrid(xx, yy)  # (2, 45, 45)
        
        # 重塑
        self.grid = torch.Tensor(self.grid).view(2, -1)  # (2, 45 * 45)
        
        # 使用FoldingLayer实现折叠操作
        self.fold1 = FoldingLayer(latent_dim + 2, [512, 512, 3])
        self.fold2 = FoldingLayer(latent_dim + 3, [512, 512, 3])

    def forward(self, z):
        """
        输入:
            z: 编码器输出的特征 [B, latent_dim]
        输出:
            重建的点云 [B, M, 3]
        """
        B = z.size(0)
        
        # 获取网格并准备重复
        grid = self.grid.to(z.device)                      # [2, M]
        grid = grid.unsqueeze(0).repeat(B, 1, 1)           # [B, 2, M]
        
        # 重复特征向量以匹配点数
        z = z.unsqueeze(2).repeat(1, 1, self.num_points)   # [B, latent_dim, M]
        
        # 第一次折叠操作
        fold1_result = self.fold1(grid, z)                 # [B, 3, M]
        
        # 第二次折叠操作
        fold2_result = self.fold2(fold1_result, z)         # [B, 3, M]
        
        # 转置为 [B, M, 3] 以与原始格式保持一致
        recon = fold2_result.permute(0, 2, 1)              # [B, M, 3]
        
        return recon



