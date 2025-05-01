"""
转置数据，使其符合MultiROCKET的要求
"""
import os
import numpy as np
from src.data.data_loader import load_custom_dataset, save_dataset


def transpose_data(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    转置数据，使其符合MultiROCKET的要求
    
    参数:
        X: 特征数据，形状为 (n_samples, n_timepoints, n_features)
        y: 标签数据
    
    返回:
        转置后的特征数据和标签数据
    """
    # 检查数据形状
    print(f"原始数据形状: X={X.shape}, y={y.shape}")
    
    # 转置数据，将特征和时间点维度交换
    # 从 (n_samples, n_timepoints, n_features) 变为 (n_samples, n_features, n_timepoints)
    X_transposed = np.transpose(X, (0, 2, 1))
    
    print(f"转置后数据形状: X={X_transposed.shape}, y={y.shape}")
    
    return X_transposed, y


def process_dataset(file_path: str, output_path: str = None) -> None:
    """
    处理数据集，转置数据
    
    参数:
        file_path: 数据文件路径
        output_path: 输出文件路径，如果为None则覆盖原文件
    """
    print(f"处理数据集: {file_path}")
    
    # 加载数据
    X, y = load_custom_dataset(file_path)
    
    # 转置数据
    X_transposed, y = transpose_data(X, y)
    
    # 保存转置后的数据
    if output_path is None:
        output_path = file_path
    
    save_dataset(X_transposed, y, output_path)
    print(f"已保存转置后的数据到: {output_path}")


def main():
    """主函数"""
    # 处理训练集
    process_dataset("output/prepared_data/train_dataset.npz")
    
    # 处理验证集
    process_dataset("output/prepared_data/val_dataset.npz")
    
    # 处理测试集
    process_dataset("output/prepared_data/test_dataset.npz")


if __name__ == "__main__":
    main()
