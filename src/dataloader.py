import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import urllib.request
import zipfile
import h5py
import shutil
from pathlib import Path
import trimesh
import random

class ModelNet10Dataset(Dataset):
    """ModelNet10 数据集."""
    
    def __init__(self, root_dir=None, train=True, download=False, num_points=2048, transform=None):
        """
        参数:
            root_dir (string): 数据集根目录
            train (bool): 如果为 True，加载训练集，否则加载测试集
            download (bool): 如果为 True，从互联网下载数据集并放在 root_dir
            num_points (int): 每个点云中的点数量
            transform (callable, optional): 可选的数据转换
        """
        self.root_dir = root_dir or os.path.expanduser("~/data")
        self.train = train
        self.num_points = num_points
        self.transform = transform

        self.classes = ['bathtub', 'bed', 'chair', 'desk', 'dresser',
                        'monitor', 'night_stand', 'sofa', 'table', 'toilet']

        self.processed_data_file = os.path.join(
            self.root_dir, f"modelnet10_{'train' if train else 'test'}_{num_points}.h5")
        
        if download:
            self._download_and_process_data()
        
        # 加载处理后的数据
        if not os.path.exists(self.processed_data_file):
            raise RuntimeError("数据集未找到。请设置 download=True 下载数据集")
        
        with h5py.File(self.processed_data_file, 'r') as f:
            self.data = np.array(f['data'])
            self.labels = np.array(f['labels'])

        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        point_cloud = self.data[idx].astype(np.float32)
        label = self.labels[idx]
        
        if self.transform:
            point_cloud = self.transform(point_cloud)
        
        return torch.from_numpy(point_cloud), label
    
    def _download_and_process_data(self):
        """下载并处理 ModelNet10 数据集."""
        os.makedirs(self.root_dir, exist_ok=True)
        
        # 下载文件
        url = "https://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
        zip_path = os.path.join(self.root_dir, "ModelNet10.zip")
        
        if not os.path.exists(zip_path):
            print("下载 ModelNet10 数据集...")
            urllib.request.urlretrieve(url, zip_path)
        
        # 解压文件
        extract_folder = os.path.join(self.root_dir, "ModelNet10")
        if not os.path.exists(extract_folder):
            print("解压 ModelNet10 数据集...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.root_dir)
        
        # 处理数据并转换为点云
        print("处理3D模型并转换为点云...")
        split = "train" if self.train else "test"
        all_points = []
        all_labels = []
        
        for i, class_name in enumerate(self.classes):
            folder_path = os.path.join(extract_folder, class_name, split)
            if not os.path.exists(folder_path):
                continue
                
            off_files = [f for f in os.listdir(folder_path) if f.endswith('.off')]
            for off_file in off_files:
                try:
                    off_path = os.path.join(folder_path, off_file)
                    # 读取 .off 文件并采样点
                    mesh = trimesh.load(off_path)
                    # 均匀采样点云
                    sampled_points = mesh.sample(self.num_points)
                    
                    # 归一化到单位球体
                    center = np.mean(sampled_points, axis=0)
                    sampled_points = sampled_points - center
                    max_dist = np.max(np.sqrt(np.sum(sampled_points ** 2, axis=1)))
                    sampled_points = sampled_points / max_dist
                    
                    all_points.append(sampled_points)
                    all_labels.append(i)
                except Exception as e:
                    print(f"处理 {off_file} 时出错: {str(e)}")
        
        # 保存为H5格式
        if all_points:
            data_array = np.array(all_points)
            labels_array = np.array(all_labels)
            
            with h5py.File(self.processed_data_file, 'w') as f:
                f.create_dataset('data', data=data_array)
                f.create_dataset('labels', data=labels_array)
            
            print(f"处理完成！保存到 {self.processed_data_file}")
            print(f"数据形状: {data_array.shape}, 标签形状: {labels_array.shape}")


# 用于测试的代码
def test_modelnet10_dataset():
    """测试 ModelNet10Dataset 类."""
    # 创建训练集数据
    train_dataset = ModelNet10Dataset(root_dir="datasets", train=True, download=True, num_points=2048)
    print(f"训练数据集大小: {len(train_dataset)}")
    
    # 创建测试集数据
    test_dataset = ModelNet10Dataset(root_dir="datasets", train=False, download=True, num_points=2048)
    print(f"测试数据集大小: {len(test_dataset)}")
    
    # 测试 __getitem__()
    point_cloud, label = train_dataset[0]
    print(f"点云形状: {point_cloud.shape}")
    print(f"标签: {label}")
    print(f"类别: {train_dataset.classes[label]}")
    
    # 创建 DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    for batch_points, batch_labels in train_dataloader:
        print(f"批处理训练点云形状: {batch_points.shape}")
        print(f"批处理训练标签形状: {batch_labels.shape}")
        break
        
    for batch_points, batch_labels in test_dataloader:
        print(f"批处理测试点云形状: {batch_points.shape}")
        print(f"批处理测试标签形状: {batch_labels.shape}")
        break

