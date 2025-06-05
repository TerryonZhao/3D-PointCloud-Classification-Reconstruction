# 3D点云形状分类与重建 - Python包
"""
此模块包含用于3D点云分类和重建的模型和工具
"""

# 添加项目根目录到Python路径
import os
import sys

# 获取当前文件的绝对路径
current_path = os.path.abspath(__file__)

# 获取src目录的路径
src_dir = os.path.dirname(current_path)

# 获取项目根目录的路径
project_root = os.path.dirname(src_dir)

# 添加项目根目录到Python路径
if project_root not in sys.path:
    sys.path.append(project_root)
root_dir = os.path.dirname(src_dir)

# 将项目根目录添加到Python路径
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
