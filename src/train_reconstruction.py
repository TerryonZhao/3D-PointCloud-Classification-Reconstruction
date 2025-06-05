import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import os
import numpy as np
import yaml

# 修复导入路径
from src.pointnet2_cls import PointNet2Encoder
from src.foldingnet_ae import FoldingNetDecoder
from src.config_utils import load_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 点云数据增强函数
def augment_point_cloud(point_cloud, jitter_sigma=0.01, rot_range=np.pi/18, scale_low=0.8, scale_high=1.2):
    """
    对点云进行数据增强，包括随机旋转、缩放和抖动
    
    参数:
        point_cloud: [B, N, 3] 点云数据
        jitter_sigma: 抖动的标准差
        rot_range: 旋转角度范围 (弧度)
        scale_low, scale_high: 缩放范围
    
    返回:
        增强后的点云 [B, N, 3]
    """
    batch_size = point_cloud.shape[0]
    result = point_cloud.clone()
    
    for i in range(batch_size):
        # 1. 随机旋转
        theta = np.random.uniform(-rot_range, rot_range)
        rotation_matrix = torch.tensor([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=point_cloud.device)
        
        result[i] = torch.matmul(result[i], rotation_matrix)
        
        # 2. 随机缩放
        scale = np.random.uniform(scale_low, scale_high)
        result[i] *= scale
        
        # 3. 添加噪声
        jitter = torch.randn_like(result[i]) * jitter_sigma
        result[i] += jitter
    
    return result

# Chamfer距离损失函数
def chamfer_distance(x, y):
    """
    计算两组点云之间的Chamfer距离
    x: [B, N, 3] - 输入点云
    y: [B, M, 3] - 重建点云
    """
    # 为每个点找到最近的点
    x = x.unsqueeze(2)  # [B, N, 1, 3]
    y = y.unsqueeze(1)  # [B, 1, M, 3]
    
    # 计算每个点之间的欧几里得距离的平方
    dist = torch.sum((x - y) ** 2, dim=-1)  # [B, N, M]
    
    # 从x到y的最小距离
    min_dist_xy = torch.min(dist, dim=2)[0]  # [B, N]
    # 从y到x的最小距离
    min_dist_yx = torch.min(dist, dim=1)[0]  # [B, M]
    
    # 计算Chamfer距离
    chamfer_dist = torch.mean(min_dist_xy, dim=1) + torch.mean(min_dist_yx, dim=1)  # [B]
    
    return torch.mean(chamfer_dist)

# 加载 .h5 数据
def load_h5_dataset(path):
    with h5py.File(path, 'r') as f:
        data = torch.tensor(f['data'][:], dtype=torch.float32)  # 确保数据是 float32 类型
        labels = torch.tensor(f['labels'][:], dtype=torch.long)  # 确保标签是长整型
    return data, labels

class PointCloudAutoEncoder(nn.Module):
    def __init__(self):
        super(PointCloudAutoEncoder, self).__init__()
        self.encoder = PointNet2Encoder() # output_dim=512
        self.decoder = FoldingNetDecoder(grid_size=45, latent_dim=512)  # output_dim=3

    def forward(self, x):  # x: [B, N, 3]
        z = self.encoder(x)  # [B, 512]
        recon = self.decoder(z)  # [B, 2025, 3]
        return recon

def train_one_epoch(model, train_loader, optimizer, epoch, num_epochs):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    loop = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{num_epochs}]", leave=False)
    
    for points, _ in loop:  # 自编码器不需要标签
        points = points.to(device)
        
        # 数据增强
        points = augment_point_cloud(points)
        
        # 前向传播
        optimizer.zero_grad()
        reconstructed = model(points)
        
        # 计算损失
        loss = chamfer_distance(points, reconstructed)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def evaluate(model, test_loader):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for points, _ in test_loader:
            points = points.to(device)
            reconstructed = model(points)
            loss = chamfer_distance(points, reconstructed)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    return avg_loss

def main():
    # 加载数据
    train_data, train_labels = load_h5_dataset('datasets/modelnet10_train_2048.h5')
    test_data, test_labels = load_h5_dataset('datasets/modelnet10_test_2048.h5')
    
    # 将batch size从32降低到8，以便更细致地学习特征
    train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=8, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=8)
    
    # 创建模型和优化器
    model = PointCloudAutoEncoder().to(device)
    # 降低学习率从0.001到0.0001，避免过快收敛到局部最优
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    # 调整学习率调度器，增加稳定性
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
    
    # 创建结果保存目录
    os.makedirs("results/logs", exist_ok=True)
    
    # 增加训练参数
    num_epochs = 200
    best_loss = float('inf')
    
    # 创建日志文件记录训练和测试损失
    loss_log_file = open("results/logs/reconstruction_losses.txt", "w")
    loss_log_file.write("Epoch,Train_Loss,Test_Loss,Learning_Rate\n")  # 写入标题行
    
    # 确保检查点目录存在
    os.makedirs("checkpoints", exist_ok=True)
    
    # 训练循环
    for epoch in range(num_epochs):
        # 训练一个epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, epoch, num_epochs)
        
        # 学习率调度
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # 评估
        test_loss = evaluate(model, test_loader)
        
        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, LR: {current_lr:.6f}")
        
        # 记录损失到txt文件
        loss_log_file.write(f"{epoch+1},{train_loss:.6f},{test_loss:.6f},{current_lr:.6f}\n")
        loss_log_file.flush()  # 立即写入文件
        
        # 保存最佳模型
        if test_loss < best_loss:
            best_loss = test_loss
            # 确保 models 目录存在
            os.makedirs("models", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, "models/autoencoder_best.pth")
            print("          ✅ Best model saved.")
        
        # 每50个epoch保存一次检查点，方便后续分析和恢复训练
        if (epoch + 1) % 50 == 0:
            os.makedirs("models", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss,
            }, f"models/autoencoder_epoch_{epoch+1}.pth")
            print(f"          ✓ Checkpoint at epoch {epoch+1} saved.")
            
    # 关闭日志文件
    loss_log_file.close()
    print(f"✅ 训练和测试损失已保存到 results/logs/reconstruction_losses.txt")

if __name__ == "__main__":
    main()
