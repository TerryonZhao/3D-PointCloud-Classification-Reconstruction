
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import yaml

# 导入相关函数
from src.train_completion import chamfer_distance, PointCloudCompletion, create_partial_point_cloud
from src.evaluate_completion import get_modelnet10_classes, load_h5_dataset
from src.train_reconstruction import PointCloudAutoEncoder

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

def visualize_cd_loss_per_class(samples_data, ae_reconstructions, save_dir='results/visualization'):
    """
    计算并可视化每个类别在不同方法下的Chamfer Distance (CD)损失，并生成柱状图
    
    参数:
        samples_data: 包含每个类别样本数据的列表，每项为 (class_name, original, partial, coarse, fine)
        ae_reconstructions: 自动编码器重建结果的列表
        save_dir: 保存路径
    """
    # 收集类别名称
    class_names = [data[0] for data in samples_data]
    
    # 初始化保存不同方法的CD损失
    coarse_losses = []
    fine_losses = []
    ae_losses = []
    
    # 分别计算每个类别的损失
    for i, (class_name, original, partial, coarse, fine) in enumerate(samples_data):
        # 将NumPy数组转换为PyTorch张量并添加批次维度
        original_tensor = torch.tensor(original, dtype=torch.float32).unsqueeze(0).to(device)
        coarse_tensor = torch.tensor(coarse, dtype=torch.float32).unsqueeze(0).to(device)
        fine_tensor = torch.tensor(fine, dtype=torch.float32).unsqueeze(0).to(device)
        ae_tensor = torch.tensor(ae_reconstructions[i], dtype=torch.float32).unsqueeze(0).to(device)
        
        # 计算损失
        coarse_cd = chamfer_distance(original_tensor, coarse_tensor)
        fine_cd = chamfer_distance(original_tensor, fine_tensor)
        ae_cd = chamfer_distance(original_tensor, ae_tensor)
        
        # 添加到列表
        coarse_losses.append(coarse_cd.item())
        fine_losses.append(fine_cd.item())
        ae_losses.append(ae_cd.item())
    
    # 设置绘图参数
    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()
    
    # 设置每个类别的x位置
    x = np.arange(len(class_names))
    width = 0.15  # 柱的宽度
    
    # 绘制柱状图
    rects1 = ax.bar(x - width, coarse_losses, width, label='Coarse (PCN)', color='red', alpha=0.7)
    rects2 = ax.bar(x, fine_losses, width, label='Fine (PCN)', color='purple', alpha=0.7)
    rects3 = ax.bar(x + width, ae_losses, width, label='FoldingNet', color='orange', alpha=0.7)
    
    # 添加标签和标题
    ax.set_xlabel('Class', fontsize=14)
    ax.set_ylabel('Chamfer Distance (CD)', fontsize=14)
    ax.set_title('Chamfer Distance Loss in Different Classes', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend(fontsize=12)
    
    # 自动调整布局
    plt.tight_layout()
    
    # 保存图像
    cd_loss_path = os.path.join(save_dir, 'cd_loss_per_class.png')
    plt.savefig(cd_loss_path, dpi=300, bbox_inches='tight')
    print(f"CD损失对比图已保存至 {cd_loss_path}")
    
    # 计算平均损失
    avg_losses = {
        'coarse_avg': np.mean(coarse_losses),
        'fine_avg': np.mean(fine_losses),
        'ae_avg': np.mean(ae_losses)
    }
    
    print(f"平均CD损失比较:")
    print(f"  粗略重建(Coarse PCN): {avg_losses['coarse_avg']:.6f}")
    print(f"  精细重建(Fine PCN): {avg_losses['fine_avg']:.6f}")
    print(f"  FoldingNet: {avg_losses['ae_avg']:.6f}")
    
    return avg_losses

def main():
    # 确保保存目录存在
    save_dir = 'completion_visualization'
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载测试数据
    print("正在加载测试数据...")
    test_data, test_labels = load_h5_dataset('datasets/modelnet10_test_2048.h5')
    print(f"成功加载 {len(test_data)} 个测试样本")
    
    # 加载PCN模型
    print("正在加载PCN模型...")
    model_path = 'models/PCN_best.pth'
    model = PointCloudCompletion(
        latent_dim=512,
        num_coarse=512,
        num_fine=2048,
        grid_size=2
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("PCN模型加载成功")
    
    # 加载AutoEncoder模型
    print("正在加载AutoEncoder模型...")
    ae_model = PointCloudAutoEncoder().to(device)
    ae_checkpoint = torch.load('models/autoencoder_best.pth', map_location=device)
    ae_model.load_state_dict(ae_checkpoint['model_state_dict'])
    ae_model.eval()
    print("AutoEncoder模型加载成功")
    
    # 获取ModelNet10类别名称
    classes = get_modelnet10_classes()
    
    # 为每个类别选择样本索引（从每个类别中选择第一个样本）
    selected_indices = []
    for class_idx in range(len(classes)):
        indices = torch.where(test_labels == class_idx)[0]
        if len(indices) > 0:
            selected_indices.append(indices[0].item())
    
    # 生成重建点云数据
    fixed_ratio = 0.5  # 使用固定的遮挡率
    all_class_samples = []
    ae_reconstructions = []
    
    print(f"正在处理所有类别的点云补全，遮挡率：{fixed_ratio*100:.0f}%...")
    with torch.no_grad():
        for idx in selected_indices:
            # 获取样本
            points = test_data[idx].unsqueeze(0).to(device)
            label = test_labels[idx].item()
            class_name = classes[label]
            
            print(f"  正在处理类别 '{class_name}'...")
            
            # 创建部分点云
            partial_points = create_partial_point_cloud(points, partial_ratio=fixed_ratio)
            
            # PCN模型补全
            coarse_points, fine_points = model(partial_points)
            
            # AutoEncoder模型重建（使用Partial点云）
            ae_recon_points = ae_model(partial_points)
            
            # 转换为NumPy
            original_np = points.squeeze().cpu().numpy()
            partial_np = partial_points.squeeze().cpu().numpy()
            coarse_np = coarse_points.squeeze().cpu().numpy()
            fine_np = fine_points.squeeze().cpu().numpy()
            ae_recon_np = ae_recon_points.squeeze().cpu().numpy()
            
            # 收集样本数据
            all_class_samples.append((class_name, original_np, partial_np, coarse_np, fine_np))
            ae_reconstructions.append(ae_recon_np)
    
    # 计算并可视化CD损失
    print("正在计算和可视化Chamfer Distance损失...")
    visualize_cd_loss_per_class(all_class_samples, ae_reconstructions, save_dir)
    
if __name__ == '__main__':
    main()
