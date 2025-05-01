"""
将三分类数据转换为二分类数据

此脚本将原始的三分类数据（成功/失败/无效）转换为二分类数据（成功/失败），
通过移除标签为2（无效）的样本。
"""
import os
import numpy as np
from src.data.data_loader import load_custom_dataset, save_dataset


def convert_to_binary(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    将三分类数据转换为二分类数据
    
    参数:
        X: 特征数据，形状为 (n_samples, n_features, n_timepoints)
        y: 标签数据，形状为 (n_samples,)
    
    返回:
        转换后的特征数据和标签数据
    """
    # 找出标签为0（失败）或1（成功）的样本索引
    binary_indices = (y == 0) | (y == 1)
    
    # 统计各类别数量
    n_fail = np.sum(y == 0)
    n_success = np.sum(y == 1)
    n_invalid = np.sum(y == 2)
    
    print(f"原始数据统计:")
    print(f"  失败样本数: {n_fail}")
    print(f"  成功样本数: {n_success}")
    print(f"  无效样本数: {n_invalid}")
    print(f"  总样本数: {len(y)}")
    
    # 筛选出二分类样本
    X_binary = X[binary_indices]
    y_binary = y[binary_indices]
    
    print(f"转换后数据统计:")
    print(f"  失败样本数: {np.sum(y_binary == 0)}")
    print(f"  成功样本数: {np.sum(y_binary == 1)}")
    print(f"  总样本数: {len(y_binary)}")
    
    return X_binary, y_binary


def process_dataset(file_path: str, output_path: str = None) -> None:
    """
    处理数据集，将三分类转换为二分类
    
    参数:
        file_path: 数据文件路径
        output_path: 输出文件路径，如果为None则覆盖原文件
    """
    print(f"处理数据集: {file_path}")
    
    # 加载数据
    X, y = load_custom_dataset(file_path)
    print(f"数据形状: X={X.shape}, y={y.shape}")
    
    # 转换为二分类
    X_binary, y_binary = convert_to_binary(X, y)
    
    # 保存转换后的数据
    if output_path is None:
        # 创建新的文件名，在原文件名基础上添加"_binary"
        base_path = os.path.splitext(file_path)[0]
        output_path = f"{base_path}_binary.npz"
    
    save_dataset(X_binary, y_binary, output_path)
    print(f"已保存二分类数据到: {output_path}")


def main():
    """主函数"""
    # 创建输出目录
    os.makedirs("output/prepared_data/binary", exist_ok=True)
    
    # 处理训练集
    process_dataset(
        "output/prepared_data/train_dataset.npz",
        "output/prepared_data/binary/train_dataset.npz"
    )
    
    # 处理验证集
    process_dataset(
        "output/prepared_data/val_dataset.npz",
        "output/prepared_data/binary/val_dataset.npz"
    )
    
    # 处理测试集
    process_dataset(
        "output/prepared_data/test_dataset.npz",
        "output/prepared_data/binary/test_dataset.npz"
    )


if __name__ == "__main__":
    main()
