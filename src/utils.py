# utils.py

import torch

def square_distance(src, dst):
    # src: [B, N, C], dst: [B, M, C]
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # [B, N, M]
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Inputs:
        points: [B, N, C]
        idx: [B, S] or [B, S, nsample]
    Returns:
        new_points: indexed points, shape depends on idx
    """
    device = points.device
    B = points.shape[0]

    # Flatten idx for gather
    view_shape = list(idx.shape)
    view_shape.append(points.shape[-1])  # → [B, S, C] or [B, S, nsample, C]

    if idx.dim() == 2:
        # 处理 idx 是二维的情况 [B, S]
        batch_indices = torch.arange(B, dtype=torch.long, device=device).view(B, 1)
        batch_indices = batch_indices.expand(-1, idx.shape[1])
        new_points = points[batch_indices.flatten(), idx.flatten()].view(*view_shape)
    else:
        # 处理 idx 是三维的情况 [B, S, nsample]
        batch_indices = torch.arange(B, dtype=torch.long, device=device).view(B, 1, 1)
        batch_indices = batch_indices.expand(-1, idx.shape[1], idx.shape[2])
        new_points = points[batch_indices, idx]

    return new_points.view(*view_shape)


def farthest_point_sample(xyz, npoint):
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(xyz.device)
    distance = torch.ones(B, N).to(xyz.device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(xyz.device)
    batch_indices = torch.arange(B, dtype=torch.long).to(xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    B, N, C = xyz.shape
    S = new_xyz.shape[1]
    group_idx = torch.arange(N, device=xyz.device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)  # [B, S, N]
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, features=None, use_xyz=True):
    B, N, C = xyz.shape
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)  # [B, npoint, nsample]
    grouped_xyz = index_points(xyz, idx) - new_xyz.view(B, npoint, 1, C)  # [B, npoint, nsample, 3]
    
    if features is not None:
        grouped_features = index_points(features, idx)  # [B, npoint, nsample, D]
        if use_xyz:
            # 组合点坐标和特征
            grouped_features = torch.cat([grouped_xyz, grouped_features], dim=-1)
            return new_xyz, grouped_features
        else:
            return new_xyz, grouped_features
    else:
        if use_xyz:
            return new_xyz, grouped_xyz
        else:
            return new_xyz, None

# Chamfer距离损失函数
def chamfer_distance(x, y, reduce=True):
    """
    计算两组点云之间的Chamfer距离
    x: [B, N, 3] - 输入点云
    y: [B, M, 3] - 重建点云
    reduce: bool - 是否返回平均值，如果为False则返回每个样本的Chamfer距离
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
    
    if reduce:
        return torch.mean(chamfer_dist)
    else:
        return chamfer_dist

def save_checkpoint(model, optimizer, scheduler=None, epoch=0, val_loss=None, filename="checkpoint.pth"):
    """
    保存模型检查点
    
    参数:
        model: 要保存的模型
        optimizer: 优化器
        scheduler: 学习率调度器（可选）
        epoch: 当前的epoch数
        val_loss: 验证损失（可选）
        filename: 保存的文件名
    """
    # 创建要保存的字典
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    # 如果有学习率调度器，也保存它的状态
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # 如果有验证损失，也保存它
    if val_loss is not None:
        checkpoint['val_loss'] = val_loss
    
    # 保存到文件
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    加载模型检查点
    
    参数:
        checkpoint_path: 检查点文件路径
        model: 要加载权重的模型
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）
    
    返回:
        val_loss: 检查点中保存的验证损失（如果有）
    """
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 如果提供了优化器，加载它的状态
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 如果提供了学习率调度器，且检查点中有它的状态，也加载
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # 返回验证损失（如果有）
    return checkpoint.get('val_loss', float('inf'))