"""
配置工具模块 - 用于加载项目配置
"""

import os
import yaml

def load_config(config_file='configs/config.yaml'):
    """
    加载YAML配置文件
    
    参数:
        config_file: 配置文件路径 (相对于项目根目录)
    
    返回:
        配置字典
    """
    # 获取项目根目录的绝对路径
    # 假设此文件位于src/目录下
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(root_dir, config_file)
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        return {}
