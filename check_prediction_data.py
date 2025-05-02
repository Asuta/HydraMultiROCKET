"""
检查和修复预测数据中的缺失值
"""
import os
import numpy as np
import pandas as pd
import argparse
from src.data.data_loader import load_custom_dataset, save_dataset


def check_missing_values(X: np.ndarray, y: np.ndarray) -> bool:
    """
    检查数据中是否有缺失值

    参数:
        X: 特征数据
        y: 标签数据

    返回:
        是否有缺失值
    """
    has_missing_X = np.isnan(X).any()
    has_missing_y = np.isnan(y).any()

    if has_missing_X:
        print(f"特征数据中有缺失值，缺失值数量: {np.isnan(X).sum()}")

    if has_missing_y:
        print(f"标签数据中有缺失值，缺失值数量: {np.isnan(y).sum()}")

    return has_missing_X or has_missing_y


def fix_missing_values(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    修复数据中的缺失值

    参数:
        X: 特征数据
        y: 标签数据

    返回:
        修复后的特征数据和标签数据
    """
    # 检查标签中的缺失值
    if np.isnan(y).any():
        # 找出有效标签的索引
        valid_indices = ~np.isnan(y)
        X = X[valid_indices]
        y = y[valid_indices]
        print(f"已移除标签中有缺失值的样本，剩余样本数: {len(y)}")

    # 检查特征中的缺失值
    if np.isnan(X).any():
        # 对于特征中的缺失值，使用前向填充
        for i in range(len(X)):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    if np.isnan(X[i, j, k]):
                        # 如果是第一个时间点，使用该特征的均值填充
                        if j == 0:
                            # 计算该特征在所有样本中的均值（排除NaN）
                            feature_mean = np.nanmean(X[:, :, k])
                            X[i, j, k] = feature_mean
                        else:
                            # 使用前一个时间点的值填充
                            X[i, j, k] = X[i, j-1, k]

        print(f"已修复特征数据中的缺失值")

    return X, y


def process_dataset(file_path: str, output_path: str = None) -> None:
    """
    处理数据集，检查并修复缺失值

    参数:
        file_path: 数据文件路径
        output_path: 输出文件路径，如果为None则覆盖原文件
    """
    print(f"处理数据集: {file_path}")

    # 加载数据
    X, y = load_custom_dataset(file_path)
    print(f"数据形状: X={X.shape}, y={y.shape}")

    # 检查缺失值
    has_missing = check_missing_values(X, y)

    if has_missing:
        # 修复缺失值
        X, y = fix_missing_values(X, y)

        # 保存修复后的数据
        if output_path is None:
            output_path = file_path

        save_dataset(X, y, output_path)
        print(f"已保存修复后的数据到: {output_path}")
    else:
        print("数据中没有缺失值，无需修复")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='检查和修复数据中的缺失值')
    parser.add_argument('--data_dir', type=str, default='output/data/prediction_data', help='数据目录')
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 处理训练集
    process_dataset(os.path.join(args.data_dir, "train_dataset.npz"))

    # 处理验证集
    process_dataset(os.path.join(args.data_dir, "val_dataset.npz"))

    # 处理测试集
    process_dataset(os.path.join(args.data_dir, "test_dataset.npz"))


if __name__ == "__main__":
    main()
