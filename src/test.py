import torch
from foldingnet_ae import FoldingNetDecoder  # 请确认导入路径无误

# 模拟一个 batch 的 latent vector
B = 32
latent_dim = 1024
z = torch.randn(B, latent_dim)  # 模拟 PointNet++ 编码器输出

# 初始化 decoder
decoder = FoldingNetDecoder(grid_size=32, latent_dim=latent_dim)

# 前向传播 + shape 打印（在 decoder 中已定义 forward）
with torch.no_grad():
    recon = decoder(z)
