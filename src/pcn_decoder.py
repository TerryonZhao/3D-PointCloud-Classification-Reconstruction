import torch
import torch.nn as nn

class PCNDecoder(nn.Module):
    def __init__(self, latent_dim=512, num_coarse=512, num_fine=2048, grid_size=2):
        super(PCNDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.grid_size = grid_size
        self.num_grid_points = grid_size ** 2  # 每个 coarse 点展开的细化点数
        
        # 确保点数匹配
        assert self.num_fine == self.num_coarse * (self.grid_size ** 2)

        # ===== Coarse Decoder =====
        self.coarse_mlp = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_coarse * 3)
        )

        # ===== 2D Folding Grid (针对y方向拉伸问题调整) =====
        self.x_grid_scale = 0.05  # x方向保持不变
        self.y_grid_scale = 0.03  # y方向减小，以抵消拉伸效应
        
        # 使用不同的x和y方向网格比例
        a = torch.linspace(-self.x_grid_scale, self.x_grid_scale, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-self.y_grid_scale, self.y_grid_scale, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2)  # (1, 2, S) - 调整后的形状

        # ===== Fine Folding MLP (与标准实现对齐) =====
        self.folding_mlp = nn.Sequential(
            nn.Conv1d(latent_dim + 2 + 3, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1)
        )

    def forward(self, x):  # x: [B, latent_dim]
        B = x.size(0)
        device = x.device
        
        # ===== Coarse output =====
        # 生成粗糙点云
        coarse = self.coarse_mlp(x).reshape(B, self.num_coarse, 3)  # [B, num_coarse, 3]
        
        # ===== Fine output =====
        folding_seed = self.folding_seed.to(device)  # (1, 2, grid_size^2)
        
        # 创建细化点云 (与标准实现保持一致)
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)  # (B, num_coarse, S, 3)
        point_feat = point_feat.reshape(B, self.num_fine, 3).transpose(2, 1)  # (B, 3, num_fine)
        
        # 准备网格种子 (与标准实现保持一致)
        seed = folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1)  # (B, 2, num_coarse, S)
        seed = seed.reshape(B, 2, self.num_fine)  # (B, 2, num_fine)
        
        # 准备全局特征 (与标准实现保持一致)
        feat_global = x.unsqueeze(2).expand(-1, -1, self.num_fine)  # (B, latent_dim, num_fine)
        
        # 拼接特征 (与标准实现保持一致)
        feat = torch.cat([feat_global, seed, point_feat], dim=1)  # (B, latent_dim+2+3, num_fine)
        
        # 通过folding网络
        fine_offset = self.folding_mlp(feat)  # (B, 3, num_fine)
        
        # 应用残差连接 (与标准实现保持一致)
        fine_points = fine_offset + point_feat  # (B, 3, num_fine)
        
        # 确保输出格式正确 (与标准实现保持一致)
        fine_points = fine_points.transpose(1, 2).contiguous()  # (B, num_fine, 3)
        
        return coarse.contiguous(), fine_points
