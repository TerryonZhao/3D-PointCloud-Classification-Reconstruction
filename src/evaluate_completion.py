import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import yaml

from src.train_completion import PointCloudCompletion, load_h5_dataset, create_partial_point_cloud

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 可视化点云函数
def visualize_pointcloud(original, partial, coarse, fine, title=None, save_path=None):
    """
    可视化原始点云、部分点云、粗略重建点云和精细重建点云
    
    参数:
        original: [N, 3] - 原始完整点云
        partial: [M, 3] - 部分点云（输入）
        coarse: [K, 3] - 粗略重建点云
        fine: [L, 3] - 精细重建点云
        title: 标题
        save_path: 保存路径
    """
    fig = plt.figure(figsize=(20, 30))
    
    # 原始点云
    ax1 = fig.add_subplot(141, projection='3d')
    ax1.scatter(original[:, 0], original[:, 1], original[:, 2], c='blue', s=10, alpha=0.6)
    ax1.set_title('Complete Point Cloud')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([-1, 1])
    ax1.set_zlim([-1, 1])
    
    # 部分点云
    ax2 = fig.add_subplot(142, projection='3d')
    ax2.scatter(partial[:, 0], partial[:, 1], partial[:, 2], c='green', s=10, alpha=0.6)
    ax2.set_title('Partial Point Cloud')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-1, 1])
    ax2.set_zlim([-1, 1])
    
    # 粗略重建点云
    ax3 = fig.add_subplot(143, projection='3d')
    ax3.scatter(coarse[:, 0], coarse[:, 1], coarse[:, 2], c='red', s=10, alpha=0.6)
    ax3.set_title('Coarse Reconstruction')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_xlim([-1, 1])
    ax3.set_ylim([-1, 1])
    ax3.set_zlim([-1, 1])
    
    # 精细重建点云
    ax4 = fig.add_subplot(144, projection='3d')
    ax4.scatter(fine[:, 0], fine[:, 1], fine[:, 2], c='purple', s=10, alpha=0.6)
    ax4.set_title('Fine Reconstruction')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.set_xlim([-1, 1])
    ax4.set_ylim([-1, 1])
    ax4.set_zlim([-1, 1])
    
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        
def visualize_comparison_across_ratios(original, partial_list, fine_list, ratios, class_name, save_path=None):
    """
    可视化特定类别在不同遮挡率下的输入与输出对比
    
    参数:
        original: [N, 3] - 原始完整点云
        partial_list: 不同遮挡率下的部分点云列表
        fine_list: 不同遮挡率下的重建点云列表
        ratios: 可见率列表
        class_name: 类别名称
        save_path: 保存路径
    """
    # 使用更大的图像尺寸以获得更好的可视化效果
    fig = plt.figure(figsize=(18, 8))
    
    # 设置标题
    fig.suptitle(f'{class_name.capitalize()} - Point Cloud Completion at Different Visibility Ratios', fontsize=18)
    
    # 创建三行：第一行为原始点云（参考），第二行为输入点云，第三行为输出点云
    num_ratios = len(ratios)
    
    # 第一行 - 原始完整点云（参考）
    ax_orig = fig.add_subplot(3, 1, 1, projection='3d')
    ax_orig.scatter(original[:, 0], original[:, 1], original[:, 2], c='blue', s=10, alpha=0.6)
    ax_orig.set_title(f'Original Complete Point Cloud ({class_name.capitalize()})', fontsize=16)
    ax_orig.set_xlim([-0.7, 0.7])
    ax_orig.set_ylim([-0.7, 0.7])
    ax_orig.set_zlim([-0.7, 0.7])
    ax_orig.axis('off')
    ax_orig.view_init(elev=30, azim=45)  # 设置一个好的视角
    
    # 第二行 - 输入点云（不同可见率）
    for i, (ratio, partial) in enumerate(zip(ratios, partial_list)):
        ax = fig.add_subplot(3, num_ratios, num_ratios+i+1, projection='3d')
        ax.scatter(partial[:, 0], partial[:, 1], partial[:, 2], c='green', s=10, alpha=0.6)
        
        if i == 0:
            ax.set_ylabel('Input\n(Partial)', fontsize=14)
            
        ax.set_title(f'Partial {int(ratio*100)}%', fontsize=14)
        ax.set_xlim([-0.7, 0.7])
        ax.set_ylim([-0.7, 0.7])
        ax.set_zlim([-0.7, 0.7])
        ax.axis('off')
        ax.view_init(elev=30, azim=45)  # 与原始点云保持相同视角
    
    # 第三行 - 输出点云（重建）
    for i, (ratio, fine) in enumerate(zip(ratios, fine_list)):
        ax = fig.add_subplot(3, num_ratios, 2*num_ratios+i+1, projection='3d')
        ax.scatter(fine[:, 0], fine[:, 1], fine[:, 2], c='purple', s=10, alpha=0.6)

        if i == 0:
            ax.set_ylabel('Output\n(Completed)', fontsize=14)
            
        ax.set_xlim([-0.7, 0.7])
        ax.set_ylim([-0.7, 0.7])
        ax.set_zlim([-0.7, 0.7])
        ax.axis('off')
        ax.view_init(elev=30, azim=45)  # 与前面保持相同视角
    
    # 调整子图之间的间距
    plt.subplots_adjust(hspace=0.15, wspace=0.05)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为顶部标题腾出空间
    
    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches='tight')  # 更高的DPI以获得更清晰的图像
        plt.close()
    else:
        plt.show()

def visualize_all_classes_at_fixed_ratio(samples_data, fixed_ratio=0.5, save_path=None, ae_reconstructions=None):
    """
    在固定遮挡率下，一次性展示所有类别的点云补全效果，格式为10行（类别）× 5列（complete, partial, coarse, fine, autoencoder）
    
    参数:
        samples_data: 包含每个类别样本数据的列表，每项为 (class_name, original, partial, coarse, fine)
        fixed_ratio: 固定遮挡率
        save_path: 保存路径
        ae_reconstructions: 自动编码器重建结果的列表
    """
    num_classes = len(samples_data)
    num_cols = 5 if ae_reconstructions else 4  # 如果有自动编码器结果，则为5列，否则为4列
    num_rows = num_classes  # 10个类别
    
    # 创建一个更紧凑的图像，但保持显示大小足够大
    fig = plt.figure(figsize=(25, 30) if ae_reconstructions else (20, 30))  # 添加自动编码器列后稍微增加宽度
    
    # 只设置一个主标题
    fig.suptitle(f'All Classes Point Cloud Completion at Partial Ratio {fixed_ratio:.1f}', fontsize=30, y=0.98)
    
    # 减小子图之间的间距，使点云更紧凑
    plt.subplots_adjust(hspace=0.1, wspace=0.01)
    
    for i, (class_name, original, partial, coarse, fine) in enumerate(samples_data):
        # 每个类别一行，包含4个点云：原始、部分、粗略重建、精细重建
        
        # Original point cloud
        ax1 = fig.add_subplot(num_rows, num_cols, i * num_cols + 1, projection='3d')
        ax1.scatter(original[:, 0], original[:, 1], original[:, 2], c='blue', s=10, alpha=0.7)  # 增加点大小和透明度
        
        # 设置类别标签和列标题
        ax1.set_ylabel(class_name, fontsize=14, rotation=90, labelpad=20)
        if i == 0:
            # 只在第一行的每列设置一次标题
            ax1.set_title('Complete', fontsize=24, pad=10)
            
        ax1.set_xlim([-0.7, 0.7])  # 更加缩小显示范围，使点云显得更大
        ax1.set_ylim([-0.7, 0.7])
        ax1.set_zlim([-0.7, 0.7])
        ax1.axis('off')
        # 改变视角以更好地显示点云形状
        ax1.view_init(elev=30, azim=45)  # 调整视角以更好地展示点云
        
        # Partial point cloud
        ax2 = fig.add_subplot(num_rows, num_cols, i * num_cols + 2, projection='3d')
        ax2.scatter(partial[:, 0], partial[:, 1], partial[:, 2], c='green', s=10, alpha=0.6)  # 增加点大小和透明度
        if i == 0:
            ax2.set_title('Partial', fontsize=24, pad=10)
        ax2.set_xlim([-0.7, 0.7])
        ax2.set_ylim([-0.7, 0.7])
        ax2.set_zlim([-0.7, 0.7])
        ax2.axis('off')
        ax2.view_init(elev=30, azim=45)  # 保持与第一列相同的视角
        
        # Coarse reconstruction
        ax3 = fig.add_subplot(num_rows, num_cols, i * num_cols + 3, projection='3d')
        ax3.scatter(coarse[:, 0], coarse[:, 1], coarse[:, 2], c='red', s=10, alpha=0.6)  # 增加点大小
        if i == 0:
            ax3.set_title('Coarse (PCN)', fontsize=24, pad=10)
        ax3.set_xlim([-0.7, 0.7])
        ax3.set_ylim([-0.7, 0.7])
        ax3.set_zlim([-0.7, 0.7])
        ax3.axis('off')
        ax3.view_init(elev=30, azim=45)
        
        # Fine reconstruction
        ax4 = fig.add_subplot(num_rows, num_cols, i * num_cols + 4, projection='3d')
        ax4.scatter(fine[:, 0], fine[:, 1], fine[:, 2], c='purple', s=10, alpha=0.7)  # 增加点大小
        if i == 0:
            ax4.set_title('Fine (PCN)', fontsize=24, pad=10)
        ax4.set_xlim([-0.7, 0.7])
        ax4.set_ylim([-0.7, 0.7])
        ax4.set_zlim([-0.7, 0.7])
        ax4.axis('off')
        ax4.view_init(elev=30, azim=45)
        
        # AutoEncoder reconstruction (如果提供)
        if ae_reconstructions:
            ae_recon = ae_reconstructions[i]
            ax5 = fig.add_subplot(num_rows, num_cols, i * num_cols + 5, projection='3d')
            ax5.scatter(ae_recon[:, 0], ae_recon[:, 1], ae_recon[:, 2], c='orange', s=8, alpha=0.4)
            if i == 0:
                ax5.set_title('FoldingNet', fontsize=24, pad=10)
            ax5.set_xlim([-0.7, 0.7])
            ax5.set_ylim([-0.7, 0.7])
            ax5.set_zlim([-0.7, 0.7])
            ax5.axis('off')
            ax5.view_init(elev=30, azim=45)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为顶部标题腾出空间
    
    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches='tight')  # 提高DPI获得更清晰的图像
        plt.close()
    else:
        plt.show()

# 获取ModelNet10的类别名称
def get_modelnet10_classes():
    return [
        'bathtub', 'bed', 'chair', 'desk', 'dresser',
        'monitor', 'night_stand', 'sofa', 'table', 'toilet'
    ]

# 为每个类别选择一个样本
def get_sample_for_each_class(test_data, test_labels, num_classes=10):
    """
    为每个类别选择一个样本
    """
    selected_indices = []
    
    for class_idx in range(num_classes):
        # 找到所有属于该类别的样本索引
        indices = torch.where(test_labels == class_idx)[0]
        
        # 如果该类别有样本，随机选择一个
        if len(indices) > 0:
            selected_index = indices[np.random.randint(0, len(indices))]
            selected_indices.append(selected_index.item())
    
    return selected_indices

# 修改后的可视化函数，只生成一张大图，显示所有类别在固定遮挡率下的补全效果
def visualize_completion(model_path, test_data, test_labels, save_dir='completion_visualization'):
    """
    可视化点云补全结果：
    1. 生成一张10行(类别)x4列(complete, partial, coarse, fine)的大图
    2. 为chair和table类别生成不同遮挡率下(0, 0.2, 0.4, 0.6, 0.8)的输入与输出对比图
    
    参数:
        model_path: PCN模型权重路径
        test_data: 测试数据
        test_labels: 测试标签
        save_dir: 结果保存目录
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved to: {save_dir}")
    
    # 加载PCN模型
    print("Loading PCN model...")
    model = PointCloudCompletion(
        latent_dim=512,
        num_coarse=512,  # 与训练时保持一致
        num_fine=2048,   # 确保等于 num_coarse * grid_size^2
        grid_size=2      # 与训练时保持一致，每个粗糙点展开为4个细节点
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("PCN model loaded successfully")
    
    # 加载AutoEncoder模型
    print("Loading AutoEncoder model...")
    from train_reconstruction import PointCloudAutoEncoder
    ae_model = PointCloudAutoEncoder().to(device)
    ae_checkpoint = torch.load('checkpoints/autoencoder_best.pth', map_location=device)
    ae_model.load_state_dict(ae_checkpoint['model_state_dict'])
    ae_model.eval()
    print("AutoEncoder model loaded successfully")
    
    # 获取类别名称
    classes = get_modelnet10_classes()
    
    # 为每个类别选择一个样本
    selected_indices = get_sample_for_each_class(test_data, test_labels)
    
    # 可视化所有类别在固定遮挡率下的补全效果（固定遮挡率为0.5）
    fixed_ratio = 0.5
    all_class_samples = []
    ae_reconstructions = []  # 存储自动编码器重建结果
    
    print(f"Processing all classes at partial ratio {fixed_ratio*100:.0f}%...")
    
    # 记录chair和table类别的样本索引，用于后续不同遮挡率的可视化
    chair_idx = None
    table_idx = None
    chair_class_idx = classes.index('chair')
    table_class_idx = classes.index('table')
    
    with torch.no_grad():
        for i, idx in enumerate(selected_indices):
            # 获取样本
            points = test_data[idx].unsqueeze(0).to(device)
            label = test_labels[idx].item()
            class_name = classes[label]
            
            print(f"  Processing class '{class_name}'...")
            
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
            
            # 收集这个类别的样本数据，包括粗略重建结果
            all_class_samples.append((class_name, original_np, partial_np, coarse_np, fine_np))
            ae_reconstructions.append(ae_recon_np)
            
            # 记录chair和table类别的样本索引
            if label == chair_class_idx:
                chair_idx = idx
            elif label == table_class_idx:
                table_idx = idx
    
    # 将所有类别在固定遮挡率下的补全效果合并在一起
    all_classes_save_path = os.path.join(save_dir, f'all_classes_partial_{int(fixed_ratio*100)}.png')
    print(f"Generating and saving overview of all classes completion at partial ratio {fixed_ratio*100:.0f}%...")
    visualize_all_classes_at_fixed_ratio(all_class_samples, fixed_ratio, all_classes_save_path, ae_reconstructions)
    print(f"Overview image saved to {all_classes_save_path}")
    
    # 为chair和table类别创建不同遮挡率下的可视化比较
    visibility_ratios = [0.0, 0.2, 0.4, 0.6, 0.8]  # 5个不同的可见率 (遮挡率=1-可见率)
    
    # 处理chair类别
    if chair_idx is not None:
        process_class_across_ratios(model, test_data[chair_idx], 'chair', visibility_ratios, save_dir)
    else:
        print("Warning: No chair sample found in the selected samples.")
    
    # 处理table类别
    if table_idx is not None:
        process_class_across_ratios(model, test_data[table_idx], 'table', visibility_ratios, save_dir)
    else:
        print("Warning: No table sample found in the selected samples.")

def process_class_across_ratios(model, points, class_name, visibility_ratios, save_dir):
    """
    为特定类别在不同遮挡率下创建输入与输出对比可视化
    
    参数:
        model: 点云补全模型
        points: 原始点云数据 (单个样本)
        class_name: 类别名称
        visibility_ratios: 可见率列表 (遮挡率 = 1 - 可见率)
        save_dir: 保存目录
    """
    print(f"Processing '{class_name}' class at different visibility ratios...")
    
    # 将点云数据转移到设备上并添加批次维度
    original_tensor = points.unsqueeze(0).to(device)
    original_np = points.numpy()
    
    partial_list = []
    fine_list = []
    
    # 对每个可见率进行处理
    with torch.no_grad():
        for ratio in visibility_ratios:
            # 为100%可见率设置特殊处理 (使用原始完整点云)
            if ratio == 0.0:
                # 对于0%可见率，创建极小的部分点云 (几乎全部遮挡)
                partial_points = create_partial_point_cloud(original_tensor, partial_ratio=1.0-0.01)
            else:
                # 创建部分点云
                partial_points = create_partial_point_cloud(original_tensor, partial_ratio=1.0-ratio)
            
            # 补全
            _, fine_points = model(partial_points)
            
            # 转换为NumPy
            partial_np = partial_points.squeeze().cpu().numpy()
            fine_np = fine_points.squeeze().cpu().numpy()
            
            # 添加到列表
            partial_list.append(partial_np)
            fine_list.append(fine_np)
    
    # 保存可视化结果
    save_path = os.path.join(save_dir, f'{class_name}_visibility_comparison.png')
    print(f"  Generating and saving {class_name} comparison across visibility ratios...")
    visualize_comparison_across_ratios(original_np, partial_list, fine_list, visibility_ratios, class_name, save_path)
    print(f"  Comparison image saved to {save_path}")

def main():
    # 加载测试数据
    print("正在加载测试数据...")
    test_data, test_labels = load_h5_dataset('datasets/modelnet10_test_2048.h5')
    print(f"成功加载 {len(test_data)} 个测试样本")
    
    # 模型路径
    model_path = 'models/PCN_best.pth'
    
    # 可视化补全结果
    visualize_completion(
        model_path=model_path,
        test_data=test_data,
        test_labels=test_labels,
        save_dir='results/visualization'
    )

if __name__ == '__main__':
    main()