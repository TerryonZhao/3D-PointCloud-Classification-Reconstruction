import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import sample_and_group


# Utility MLP block
def mlp(channels):
    return nn.Sequential(*[
        nn.Sequential(nn.Conv2d(channels[i], channels[i + 1], 1),
                      nn.BatchNorm2d(channels[i + 1]),
                      nn.ReLU())
        for i in range(len(channels) - 1)
    ])


class PointNet2Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        # SA layer 1: 2048 → 512
        self.npoint1 = 512
        self.radius1 = 0.2
        self.k1 = 32
        self.mlp1 = mlp([3, 64, 64, 128])

        # SA layer 2: 512 → 128
        self.npoint2 = 128
        self.radius2 = 0.4
        self.k2 = 64
        self.mlp2 = mlp([128, 128, 128, 256])

        # SA layer 3: 128 → 1 (global)
        self.mlp3 = mlp([256, 256, 512, 512])

    def forward(self, xyz):
        """
        Input:
            xyz: [B, N, 3]
        Output:
            global feature: [B, 1024]
        """
        B, N, _ = xyz.shape

        # Layer 1
        new_xyz1, grouped_xyz1 = sample_and_group(self.npoint1, self.radius1, self.k1, xyz,
                                                  use_xyz=True)  # [B, 512, 32, 3]
        grouped_xyz1 = grouped_xyz1.permute(0, 3, 2, 1)  # [B, 3, 32, 512]
        feat1 = self.mlp1(grouped_xyz1)  # [B, 128, 32, 512]
        feat1 = F.max_pool2d(feat1, kernel_size=[feat1.size(2), 1])  # [B, 128, 1, 512]
        feat1 = feat1.squeeze(2).permute(0, 2, 1)  # [B, 512, 128]

        # Layer 2
        new_xyz2, grouped_feat2 = sample_and_group(self.npoint2, self.radius2, self.k2, new_xyz1, 
                                                  features=feat1, use_xyz=False)
        grouped_feat2 = grouped_feat2.permute(0, 3, 2, 1)  # [B, 128, 64, 128]
        feat2 = self.mlp2(grouped_feat2)
        feat2 = F.max_pool2d(feat2, kernel_size=[feat2.size(2), 1])
        feat2 = feat2.squeeze(2).permute(0, 2, 1)  # [B, 128, 256]

        # Layer 3 (global)
        grouped_feat3 = feat2.permute(0, 2, 1).unsqueeze(-1)  # [B, 256, 128, 1]
        feat3 = self.mlp3(grouped_feat3)
        feat3 = F.max_pool2d(feat3, kernel_size=[feat3.size(2), 1])
        global_feat = feat3.squeeze(-1).squeeze(-1)  # [B, 512]

        return global_feat