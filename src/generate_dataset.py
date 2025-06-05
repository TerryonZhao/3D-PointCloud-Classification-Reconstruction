import os
from dataloader import ModelNet10Dataset

def main():
    """生成2048点的ModelNet10数据集"""
    print("正在生成2048点的ModelNet10训练数据集...")
    train_dataset = ModelNet10Dataset(root_dir="data", train=True, download=True, num_points=2048)
    print(f"训练数据集生成完成，大小: {len(train_dataset)}")
    
    print("正在生成2048点的ModelNet10测试数据集...")
    test_dataset = ModelNet10Dataset(root_dir="data", train=False, download=True, num_points=2048)
    print(f"测试数据集生成完成，大小: {len(test_dataset)}")
    
    # 检查生成的数据集
    print("检查生成的数据集文件...")
    train_file = os.path.join("data", "modelnet10_train_2048.h5")
    test_file = os.path.join("data", "modelnet10_test_2048.h5")
    
    if os.path.exists(train_file) and os.path.exists(test_file):
        print(f"✅ 数据集生成成功!")
        print(f"训练数据集: {train_file}")
        print(f"测试数据集: {test_file}")
    else:
        print("❌ 数据集生成失败，请检查错误信息。")

if __name__ == "__main__":
    main()
