import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import sys

# 导入相关函数
from src.train_completion import chamfer_distance, PointCloudCompletion, create_partial_point_cloud
from src.evaluate_completion import get_modelnet10_classes, load_h5_dataset
from src.train_reconstruction import PointCloudAutoEncoder

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

def plot_pcn_fdn_comparison(samples_data, ae_reconstructions, save_dir='results/visualization'):
    """
    可视化Fine PCN相比FDN的性能提升
    参数:
        samples_data: 包含每个类别样本数据的列表，每项为 (class_name, original, partial, coarse, fine)
        ae_reconstructions: 自动编码器重建结果的列表
        save_dir: 保存路径
    """
    # 收集类别名称和索引
    class_names = [data[0] for data in samples_data]
    class_indices = list(range(len(class_names)))
    
    # 初始化保存不同方法的CD损失
    fine_losses = []
    ae_losses = []
    improvements = []
    
    # 分别计算每个类别的损失
    for i, (class_name, original, partial, coarse, fine) in enumerate(samples_data):
        # 将NumPy数组转换为PyTorch张量并添加批次维度
        original_tensor = torch.tensor(original, dtype=torch.float32).unsqueeze(0).to(device)
        fine_tensor = torch.tensor(fine, dtype=torch.float32).unsqueeze(0).to(device)
        ae_tensor = torch.tensor(ae_reconstructions[i], dtype=torch.float32).unsqueeze(0).to(device)
        
        # 计算损失
        fine_cd = chamfer_distance(original_tensor, fine_tensor)
        ae_cd = chamfer_distance(original_tensor, ae_tensor)
        
        # 计算改进量 (FDN - Fine PCN)，正值表示Fine PCN更好
        improvement = ae_cd - fine_cd
        
        # 添加到列表
        fine_losses.append(fine_cd.item())
        ae_losses.append(ae_cd.item())
        improvements.append(improvement.item())
    
    # 将类别按照FDN的损失值排序
    sorted_indices = sorted(class_indices, key=lambda i: ae_losses[i])
    
    # 重新整理数据
    sorted_class_names = [class_names[i] for i in sorted_indices]
    sorted_fine_losses = [fine_losses[i] for i in sorted_indices]
    sorted_ae_losses = [ae_losses[i] for i in sorted_indices]
    sorted_improvements = [improvements[i] for i in sorted_indices]
    
    # 创建图表
    plt.figure(figsize=(12, 6))
    
    # 绘制FDN的CD损失曲线（红线）
    plt.plot(range(len(sorted_class_names)), sorted_ae_losses, color='red', linewidth=1.5, label='FDN')
    
    # 填充改进区域
    for i in range(len(sorted_class_names)):
        if sorted_improvements[i] > 0:  # 如果PCN比FDN好
            plt.bar(i, sorted_improvements[i], bottom=sorted_fine_losses[i], 
                   color='blue', alpha=0.7, width=0.8)
    
    # 添加标签和标题
    plt.xlabel('Object Instances', fontsize=12)
    plt.ylabel('Chamfer Distance (CD)', fontsize=12)
    plt.title('Performance Improvement of Fine PCN over FDN', fontsize=14)
    
    # 创建图例
    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0], [0], color='red', linewidth=1.5, label='FDN'),
        Patch(facecolor='blue', alpha=0.7, label='Improvement')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # 优化布局
    plt.tight_layout()
    
    # 保存图像
    comparison_path = os.path.join(save_dir, 'fine_pcn_vs_fdn.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"Fine PCN与FDN对比图已保存至 {comparison_path}")
    
    # 计算整体性能提升
    total_improvement = np.sum(improvements)
    avg_improvement = np.mean(improvements)
    improvement_ratio = np.sum([1 for imp in improvements if imp > 0]) / len(improvements)
    
    print(f"性能分析:")
    print(f"  总改进量: {total_improvement:.6f}")
    print(f"  平均改进量: {avg_improvement:.6f}")
    print(f"  改进比例: {improvement_ratio*100:.1f}% ({np.sum([1 for imp in improvements if imp > 0])}/{len(improvements)})")

def main():
    # 确保保存目录存在
    save_dir = 'results/visualization'
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
    
    # 为每个类别选择多个样本（每个类别选择更多样本以增加数据点数量）
    num_samples_per_class = 10  # 从每个类别中选择10个样本
    selected_indices = []
    
    for class_idx in range(len(classes)):
        indices = torch.where(test_labels == class_idx)[0]
        if len(indices) > 0:
            # 从每个类别中最多选择num_samples_per_class个样本
            for i in range(min(num_samples_per_class, len(indices))):
                selected_indices.append(indices[i].item())
    
    # 生成重建点云数据
    fixed_ratio = 0.5  # 使用固定的遮挡率
    all_class_samples = []
    ae_reconstructions = []
    
    print(f"正在处理选定的点云补全样本，遮挡率：{fixed_ratio*100:.0f}%...")
    with torch.no_grad():
        for idx in selected_indices:
            # 获取样本
            points = test_data[idx].unsqueeze(0).to(device)
            label = test_labels[idx].item()
            class_name = f"{classes[label]}_{idx}"  # 添加索引以区分同一类别的不同样本
            
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
    
    # 绘制对比图
    print("正在绘制Fine PCN与FDN的对比图...")
    plot_pcn_fdn_comparison(all_class_samples, ae_reconstructions, save_dir)
    
if __name__ == '__main__':
    main()
