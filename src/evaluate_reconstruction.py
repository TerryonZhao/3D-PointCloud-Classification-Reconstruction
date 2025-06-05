import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import yaml

from src.train_reconstruction import PointCloudAutoEncoder, load_h5_dataset

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 可视化点云函数
def visualize_pointcloud(original, reconstructed, title=None, save_path=None):
    """
    可视化原始点云和重建点云
    
    参数:
        original: [N, 3] - 原始点云
        reconstructed: [M, 3] - 重建点云
        title: 标题
        save_path: 保存路径
    """
    fig = plt.figure(figsize=(12, 6))
    
    # 原始点云
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(original[:, 0], original[:, 1], original[:, 2], c='blue', s=5, alpha=0.7)
    ax1.set_title('原始点云')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([-1, 1])
    ax1.set_zlim([-1, 1])
    
    # 重建点云
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(reconstructed[:, 0], reconstructed[:, 1], reconstructed[:, 2], c='red', s=5, alpha=0.7)
    ax2.set_title('重建点云')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-1, 1])
    ax2.set_zlim([-1, 1])
    
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# 获取ModelNet10的类别名称
def get_modelnet10_classes():
    return [
        'bathtub', 'bed', 'chair', 'desk', 'dresser',
        'monitor', 'night_stand', 'sofa', 'table', 'toilet'
    ]

def visualize_reconstruction(model_path, test_data, test_labels, num_samples=5, save_dir='results/reconstruction'):
    """
    简单地可视化点云重建结果
    
    参数:
        model_path: 模型权重路径
        test_data: 测试数据
        test_labels: 测试标签
        num_samples: 要可视化的样本数量
        save_dir: 结果保存目录
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    print(f"结果将保存到: {save_dir}")
    
    # 加载模型
    print("正在加载模型...")
    model = PointCloudAutoEncoder().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("模型加载成功")
    
    # 获取类别名称
    classes = get_modelnet10_classes()
    
    # 随机选择一些样本进行可视化
    indices = np.random.choice(len(test_data), num_samples, replace=False)
    
    # 可视化重建结果
    print(f"开始可视化 {num_samples} 个样本...")
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # 获取样本
            print(f"处理样本 {i+1}/{num_samples}...")
            points = test_data[idx].unsqueeze(0).to(device)
            label = test_labels[idx].item()
            class_name = classes[label]
            
            # 重建
            reconstructed = model(points)
            
            # 转换为NumPy
            points_np = points.squeeze().cpu().numpy()
            recon_np = reconstructed.squeeze().cpu().numpy()
            
            # 可视化
            title = f'{class_name}'
            save_path = os.path.join(save_dir, f'{class_name}_sample_{i}.png')
            
            # 可视化并保存
            print(f"正在可视化和保存样本 {i+1}...")
            visualize_pointcloud(points_np, recon_np, title, save_path)
            
            print(f"已保存样本 {i+1}/{num_samples}: {save_path}")
    
    print(f"可视化完成！结果已保存至 '{save_dir}' 目录")

def main():
    # 加载测试数据
    print("正在加载测试数据...")
    test_data, test_labels = load_h5_dataset('datasets/modelnet10_test_2048.h5')
    print(f"成功加载 {len(test_data)} 个测试样本")
    
    # 模型路径
    model_path = 'models/autoencoder_best.pth'
    
    # 可视化重建结果
    visualize_reconstruction(
        model_path=model_path,
        test_data=test_data,
        test_labels=test_labels,
        num_samples=10,  # 可视化10个样本
        save_dir='results/reconstruction'
    )

if __name__ == '__main__':
    main()