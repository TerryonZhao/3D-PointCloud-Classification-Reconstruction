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

from src.pointnet2_cls import PointNet2Encoder  
from src.pcn_decoder import PCNDecoder  
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

# 创建部分点云函数（模拟扫描不完全）
def create_partial_point_cloud(point_cloud, partial_ratio=0.5, noise_ratio=0.02):
    """
    创建部分点云（模拟扫描不完全）
    
    参数:
        point_cloud: 原始点云 [B, N, 3]
        partial_ratio: 保留的点的比例
        noise_ratio: 添加的噪声比例
    
    返回:
        partial_point_cloud: 部分点云 [B, N*partial_ratio, 3]
    """
    B, N, _ = point_cloud.shape
    device = point_cloud.device
    
    # 计算要保留的点数
    keep_points = int(N * partial_ratio)
    
    # 为每个样本创建随机掩码
    partial_clouds = []
    for i in range(B):
        # 随机选择点
        pcd = point_cloud[i]
        indices = torch.randperm(N, device=device)[:keep_points]
        partial_cloud = pcd[indices]
        
        # 添加一些噪声
        if noise_ratio > 0:
            noise = torch.randn_like(partial_cloud) * noise_ratio
            partial_cloud = partial_cloud + noise
        
        partial_clouds.append(partial_cloud)
    
    # 堆叠为一个批次
    partial_batch = torch.stack(partial_clouds)
    
    return partial_batch

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

class PointCloudCompletion(nn.Module):
    """
    结合PointNet++编码器和PCN解码器的点云补全网络
    """
    def __init__(self, latent_dim=512, num_coarse=512, num_fine=2048, grid_size=2):
        super(PointCloudCompletion, self).__init__()
        self.encoder = PointNet2Encoder()  # 输出 512 维特征
        self.decoder = PCNDecoder(
            latent_dim=latent_dim, 
            num_coarse=num_coarse, 
            num_fine=num_fine,
            grid_size=grid_size
        )

    def forward(self, x):  # x: [B, N, 3]
        z = self.encoder(x)  # [B, 512]
        coarse_points, fine_points = self.decoder(z)  # [B, num_coarse, 3], [B, num_fine, 3]
        return coarse_points, fine_points

def train_one_epoch(model, train_loader, optimizer, epoch, num_epochs, partial_ratio=0.5):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_coarse_loss = 0.0
    total_fine_loss = 0.0
    loop = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{num_epochs}]", leave=False)
    
    for points, _ in loop:  # 不需要标签
        points = points.to(device)
        batch_size = points.shape[0]
        
        # 数据增强
        points = augment_point_cloud(points)
        
        # 创建部分点云作为输入
        partial_points = create_partial_point_cloud(points, partial_ratio=partial_ratio)
        
        # 前向传播
        optimizer.zero_grad()
        coarse_points, fine_points = model(partial_points)
        
        # 计算损失（对粗略和精细点云都应用Chamfer距离）
        coarse_loss = chamfer_distance(points, coarse_points)
        fine_loss = chamfer_distance(points, fine_points)
        
        # 总损失（对精细点云的损失给予更高的权重）
        loss = 0.5 * coarse_loss + 1.0 * fine_loss
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_coarse_loss += coarse_loss.item()
        total_fine_loss += fine_loss.item()
        
        loop.set_postfix(loss=loss.item(), coarse=coarse_loss.item(), fine=fine_loss.item())
    
    # 计算平均损失
    avg_loss = total_loss / len(train_loader)
    avg_coarse_loss = total_coarse_loss / len(train_loader)
    avg_fine_loss = total_fine_loss / len(train_loader)
    
    return avg_loss, avg_coarse_loss, avg_fine_loss

def evaluate(model, test_loader, partial_ratio=0.5):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    total_coarse_loss = 0.0
    total_fine_loss = 0.0
    
    with torch.no_grad():
        for points, _ in test_loader:
            points = points.to(device)
            
            # 创建部分点云作为输入
            partial_points = create_partial_point_cloud(points, partial_ratio=partial_ratio)
            
            # 前向传播
            coarse_points, fine_points = model(partial_points)
            
            # 计算损失
            coarse_loss = chamfer_distance(points, coarse_points)
            fine_loss = chamfer_distance(points, fine_points)
            loss = 0.5 * coarse_loss + 1.0 * fine_loss
            
            total_loss += loss.item()
            total_coarse_loss += coarse_loss.item()
            total_fine_loss += fine_loss.item()
    
    # 计算平均损失
    avg_loss = total_loss / len(test_loader)
    avg_coarse_loss = total_coarse_loss / len(test_loader)
    avg_fine_loss = total_fine_loss / len(test_loader)
    
    return avg_loss, avg_coarse_loss, avg_fine_loss

def main():
    # 加载数据
    print("加载数据...")
    train_data, train_labels = load_h5_dataset('datasets/modelnet10_train_2048.h5')
    test_data, test_labels = load_h5_dataset('datasets/modelnet10_test_2048.h5')
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=32)
    
    # 创建模型和优化器
    print("创建模型和优化器...")
    # 设置合理的参数：确保 num_fine = num_coarse * grid_size^2
    # 因为ground truth是4096点，所以我们设置num_coarse=1048，grid_size=2 (1024*2^2=4096)
    model = PointCloudCompletion(
        latent_dim=512, 
        num_coarse=512, 
        num_fine=2048,  # 等于 num_coarse * grid_size^2
        grid_size=2     # 每个粗糙点展开为4个细节点
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
    
    # 创建结果保存目录
    os.makedirs("models", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    
    # 训练参数
    num_epochs = 100
    best_loss = float('inf')
    partial_ratio = 0.5  # 不完整点云的比例
    
    # 创建日志文件
    loss_log_file = open("results/logs/completion_losses.txt", "w")
    loss_log_file.write("epoch,train_loss,train_coarse_loss,train_fine_loss,test_loss,test_coarse_loss,test_fine_loss,learning_rate\n")
    
    # 训练循环
    print(f"开始训练，总共 {num_epochs} 个 epochs...")
    for epoch in range(num_epochs):
        # 训练一个epoch
        train_loss, train_coarse_loss, train_fine_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            num_epochs=num_epochs,
            partial_ratio=partial_ratio
        )
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 评估
        test_loss, test_coarse_loss, test_fine_loss = evaluate(
            model=model, 
            test_loader=test_loader,
            partial_ratio=partial_ratio
        )
        
        # 更新学习率
        scheduler.step()
        
        # 打印进度
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.6f} (Coarse: {train_coarse_loss:.6f}, Fine: {train_fine_loss:.6f}) - "
              f"Test Loss: {test_loss:.6f} (Coarse: {test_coarse_loss:.6f}, Fine: {test_fine_loss:.6f}), "
              f"LR: {current_lr:.6f}")
        
        # 记录到损失日志
        loss_log_file.write(f"{epoch+1},{train_loss:.6f},{train_coarse_loss:.6f},{train_fine_loss:.6f},"
                           f"{test_loss:.6f},{test_coarse_loss:.6f},{test_fine_loss:.6f},{current_lr:.6f}\n")
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
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
            }, "models/PCN_best.pth")
            print("          ✅ 最佳模型已保存")
        
        # 每50个epoch保存一次检查点
        if (epoch + 1) % 50 == 0:
            os.makedirs("models", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': test_loss,
            }, f"models/completion_epoch_{epoch+1}.pth")
            print(f"          ✓ Epoch {epoch+1} 的检查点已保存")
    
    # 关闭日志文件
    loss_log_file.close()
    print(f"✅ 训练完成，损失记录已保存到 results/logs/completion_losses.txt")

if __name__ == "__main__":
    main()