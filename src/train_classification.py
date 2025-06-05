import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import os
import sys
import yaml

# 确保导入路径正确
from src.pointnet2_cls import PointNet2Encoder
from src.config_utils import load_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 1. 加载 .h5 数据
def load_h5_dataset(path):
    with h5py.File(path, 'r') as f:
        data = torch.tensor(f['data'][:], dtype=torch.float32)  # 确保数据是 float32 类型
        labels = torch.tensor(f['labels'][:], dtype=torch.long)  # 确保标签是长整型
    return data, labels

# ✅ 2. 定义 PointNet++ 分类模型
class PointNetClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.encoder = PointNet2Encoder()
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        latent = self.encoder(x)       # [B, 1024]
        logits = self.classifier(latent)  # [B, 10]
        return logits

def main():
    # 加载数据
    train_data, train_labels = load_h5_dataset('datasets/modelnet10_train_1024.h5')
    test_data, test_labels = load_h5_dataset('datasets/modelnet10_test_1024.h5')
    
    train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=32)
    
    # 创建模型、损失函数和优化器
    model = PointNetClassifier(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # ✅ 3. 训练循环
    best_acc = 0.0
    for epoch in range(50):  # 只训练 10 个 epoch
        model.train()
        total_loss, correct, total = 0, 0, 0
        loop = tqdm(train_loader, desc=f"[Epoch {epoch+1}/50]", leave=False)

        for points, labels in loop:
            points, labels = points.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(points)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item(), acc=correct / total)

        train_acc = correct / total
        print(f"[Epoch {epoch+1}] Train Loss: {total_loss:.4f}, Accuracy: {train_acc:.4f}")

        # ✅ 验证
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for points, labels in test_loader:
                points, labels = points.to(device), labels.to(device)
                outputs = model(points)
                pred = outputs.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        print(f"          Test Accuracy: {acc:.4f}")

        # ✅ 保存最优模型
        if acc > best_acc:
            best_acc = acc
            # 确保 models 目录存在
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/classification_best.pth")
            print("          ✅ Best model saved.")

if __name__ == "__main__":
    main()
