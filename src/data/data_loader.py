"""
数据加载和预处理模块
"""
import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, List
from sklearn.model_selection import train_test_split
from aeon.datasets import load_from_ts_file
from aeon.datasets import load_basic_motions, load_italy_power_demand


def load_dataset(dataset_name: str, split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
    """
    加载内置数据集

    参数:
        dataset_name: 数据集名称，可选 "basic_motions" 或 "italy_power_demand"
        split: 数据集分割，可选 "train" 或 "test"

    返回:
        X: 时间序列数据
        y: 标签
    """
    if dataset_name == "basic_motions":
        X, y = load_basic_motions(split=split)
    elif dataset_name == "italy_power_demand":
        X, y = load_italy_power_demand(split=split)
    else:
        raise ValueError(f"未知数据集: {dataset_name}")

    return X, y


def load_custom_dataset(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载自定义数据集

    参数:
        file_path: 数据文件路径，支持.ts格式和.npz格式

    返回:
        X: 时间序列数据
        y: 标签
    """
    if file_path.endswith('.ts'):
        X, y = load_from_ts_file(file_path)
    elif file_path.endswith('.npz'):
        # 加载.npz格式的数据集
        print(f"加载.npz格式的数据集: {file_path}")
        data = np.load(file_path)
        X = data['X']
        y = data['y']
    else:
        raise ValueError(f"不支持的文件格式: {file_path}，支持的格式有 .ts 和 .npz")

    return X, y


def split_dataset(X: np.ndarray, y: np.ndarray,
                 test_size: float = 0.2,
                 random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    分割数据集为训练集和测试集

    参数:
        X: 时间序列数据
        y: 标签
        test_size: 测试集比例
        random_state: 随机种子

    返回:
        X_train: 训练数据
        X_test: 测试数据
        y_train: 训练标签
        y_test: 测试标签
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


def save_dataset(X: np.ndarray, y: np.ndarray, file_path: str) -> None:
    """
    保存数据集

    参数:
        X: 时间序列数据
        y: 标签
        file_path: 保存路径
    """
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 保存数据
    np.savez(file_path, X=X, y=y)

    print(f"数据集已保存到 {file_path}")


def load_saved_dataset(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载保存的数据集

    参数:
        file_path: 数据文件路径

    返回:
        X: 时间序列数据
        y: 标签
    """
    data = np.load(file_path)
    X = data['X']
    y = data['y']

    return X, y
