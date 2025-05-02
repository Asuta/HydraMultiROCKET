"""
查看.npz文件内容的工具脚本
"""
import numpy as np
import argparse
import matplotlib.pyplot as plt
import sys
import os
from typing import Tuple, Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入中文字体修复模块
from src.visualization.fix_chinese_font import fix_chinese_display

# 修复中文显示问题
fix_chinese_display()

def load_npz_file(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载.npz文件并返回其中的数组

    参数:
        file_path: .npz文件路径

    返回:
        X: 特征数组
        y: 标签数组
    """
    data = np.load(file_path)
    X = data['X']
    y = data['y']
    return X, y

def print_npz_info(file_path: str) -> None:
    """
    打印.npz文件的基本信息

    参数:
        file_path: .npz文件路径
    """
    print(f"\n查看文件: {file_path}")
    X, y = load_npz_file(file_path)

    print(f"数据形状:")
    print(f"  X 形状: {X.shape}")
    print(f"  y 形状: {y.shape}")

    print(f"\n数据类型:")
    print(f"  X 类型: {X.dtype}")
    print(f"  y 类型: {y.dtype}")

    print(f"\n标签分布:")
    unique_labels, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"  标签 {label}: {count} 个样本 ({count/len(y)*100:.2f}%)")

    print(f"\nX 数据统计:")
    print(f"  最小值: {X.min()}")
    print(f"  最大值: {X.max()}")
    print(f"  均值: {X.mean()}")
    print(f"  标准差: {X.std()}")

    # 打印X的前几个样本的部分数据
    print(f"\nX 数据示例 (前2个样本的前5个时间点):")
    for i in range(min(2, X.shape[0])):
        print(f"  样本 {i+1}:")
        for j in range(min(5, X.shape[1])):
            feature_values = X[i, j, :5]  # 只显示前5个时间点
            print(f"    特征 {j+1}: {feature_values}")

def visualize_sample(file_path: str, sample_idx: int = 0, feature_indices: Optional[list] = None, output_dir: str = '.') -> None:
    """
    可视化一个样本的时间序列数据

    参数:
        file_path: .npz文件路径
        sample_idx: 要可视化的样本索引
        feature_indices: 要可视化的特征索引列表，如果为None则可视化所有特征
        output_dir: 图像输出目录
    """
    X, y = load_npz_file(file_path)

    if sample_idx >= X.shape[0]:
        print(f"错误: 样本索引 {sample_idx} 超出范围 (0-{X.shape[0]-1})")
        return

    # 获取样本
    sample = X[sample_idx]
    label = y[sample_idx]

    # 确定要可视化的特征
    if feature_indices is None:
        feature_indices = list(range(sample.shape[0]))

    # 创建图形
    plt.figure(figsize=(12, 8))

    # 绘制每个特征的时间序列
    for i, feature_idx in enumerate(feature_indices):
        if feature_idx >= sample.shape[0]:
            print(f"警告: 特征索引 {feature_idx} 超出范围 (0-{sample.shape[0]-1})")
            continue

        plt.subplot(len(feature_indices), 1, i+1)
        plt.plot(sample[feature_idx])
        plt.title(f"特征 {feature_idx+1}")

        if i == 0:
            plt.suptitle(f"样本 {sample_idx+1} (标签: {label})")

    plt.tight_layout()

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 保存图像
    output_path = os.path.join(output_dir, f"sample_{sample_idx}_visualization.png")
    plt.savefig(output_path)
    print(f"已保存可视化图像到: {output_path}")
    plt.close()

def visualize_feature_distribution(file_path: str, feature_idx: int = 0, output_dir: str = '.') -> None:
    """
    可视化一个特征在所有样本中的分布

    参数:
        file_path: .npz文件路径
        feature_idx: 要可视化的特征索引
        output_dir: 图像输出目录
    """
    X, y = load_npz_file(file_path)

    if feature_idx >= X.shape[1]:
        print(f"错误: 特征索引 {feature_idx} 超出范围 (0-{X.shape[1]-1})")
        return

    # 提取特定特征的所有样本数据
    feature_data = X[:, feature_idx, :]

    # 计算每个时间点的统计信息
    mean_values = np.mean(feature_data, axis=0)
    std_values = np.std(feature_data, axis=0)

    # 创建图形
    plt.figure(figsize=(12, 6))

    # 绘制均值线
    plt.plot(mean_values, 'b-', label='均值')

    # 绘制标准差区间
    plt.fill_between(
        range(len(mean_values)),
        mean_values - std_values,
        mean_values + std_values,
        alpha=0.2,
        color='b',
        label='±1 标准差'
    )

    plt.title(f"特征 {feature_idx+1} 在所有样本中的分布")
    plt.xlabel('时间点')
    plt.ylabel('值')
    plt.legend()

    plt.tight_layout()

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 保存图像
    output_path = os.path.join(output_dir, f"feature_{feature_idx}_distribution.png")
    plt.savefig(output_path)
    print(f"已保存特征分布图像到: {output_path}")
    plt.close()

def visualize_class_distribution(file_path: str, feature_idx: int = 0, output_dir: str = '.') -> None:
    """
    可视化不同类别的特征分布

    参数:
        file_path: .npz文件路径
        feature_idx: 要可视化的特征索引
        output_dir: 图像输出目录
    """
    X, y = load_npz_file(file_path)

    if feature_idx >= X.shape[1]:
        print(f"错误: 特征索引 {feature_idx} 超出范围 (0-{X.shape[1]-1})")
        return

    # 获取唯一的类别
    unique_classes = np.unique(y)

    # 创建图形
    plt.figure(figsize=(12, 6))

    # 为每个类别绘制均值线
    for class_label in unique_classes:
        # 获取该类别的样本索引
        class_indices = np.where(y == class_label)[0]

        # 提取该类别的特征数据
        class_data = X[class_indices, feature_idx, :]

        # 计算均值
        class_mean = np.mean(class_data, axis=0)

        # 绘制均值线
        plt.plot(class_mean, label=f'类别 {int(class_label)}')

    plt.title(f"特征 {feature_idx+1} 在不同类别中的分布")
    plt.xlabel('时间点')
    plt.ylabel('均值')
    plt.legend()

    plt.tight_layout()

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 保存图像
    output_path = os.path.join(output_dir, f"feature_{feature_idx}_class_distribution.png")
    plt.savefig(output_path)
    print(f"已保存类别分布图像到: {output_path}")
    plt.close()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='查看.npz文件内容')
    parser.add_argument('file_path', type=str, help='.npz文件路径')
    parser.add_argument('--visualize', '-v', action='store_true', help='是否可视化样本')
    parser.add_argument('--sample', '-s', type=int, default=0, help='要可视化的样本索引')
    parser.add_argument('--features', '-f', type=str, help='要可视化的特征索引，用逗号分隔')
    parser.add_argument('--feature-dist', '-fd', type=int, default=None, help='要可视化分布的特征索引')
    parser.add_argument('--class-dist', '-cd', type=int, default=None, help='要可视化类别分布的特征索引')
    parser.add_argument('--output-dir', '-o', type=str, default='.', help='图像输出目录')
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 打印文件信息
    print_npz_info(args.file_path)

    # 可视化样本
    if args.visualize:
        feature_indices = None
        if args.features:
            feature_indices = [int(idx) for idx in args.features.split(',')]

        visualize_sample(args.file_path, args.sample, feature_indices, args.output_dir)

    # 可视化特征分布
    if args.feature_dist is not None:
        visualize_feature_distribution(args.file_path, args.feature_dist, args.output_dir)

    # 可视化类别分布
    if args.class_dist is not None:
        visualize_class_distribution(args.file_path, args.class_dist, args.output_dir)

if __name__ == "__main__":
    main()
