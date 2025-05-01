"""
比较三分类模型和二分类模型的性能
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.models.hydra_multirocket import HydraMultiRocketModel
from src.data.data_loader import load_custom_dataset
from src.visualization.fix_chinese_font import fix_chinese_display

# 修复中文显示问题
fix_chinese_display()


def evaluate_model(model_path, test_data_path, binary=False):
    """
    评估模型性能

    参数:
        model_path: 模型路径
        test_data_path: 测试数据路径
        binary: 是否为二分类模型

    返回:
        评估结果字典
    """
    print(f"加载模型: {model_path}")
    model = HydraMultiRocketModel.load(model_path)

    print(f"加载测试数据: {test_data_path}")
    X, y = load_custom_dataset(test_data_path)

    if not binary:
        # 对于三分类模型，我们只评估类别0和1的性能
        binary_indices = (y == 0) | (y == 1)
        X = X[binary_indices]
        y = y[binary_indices]

    print(f"测试数据形状: X={X.shape}, y={y.shape}")

    # 进行预测
    y_pred = model.predict(X)

    # 计算评估指标
    accuracy = accuracy_score(y, y_pred)

    # 检查是否为二分类问题
    unique_labels = np.unique(np.concatenate([y, y_pred]))
    if len(unique_labels) <= 2:
        precision = precision_score(y, y_pred, average='binary', pos_label=1)
        recall = recall_score(y, y_pred, average='binary', pos_label=1)
        f1 = f1_score(y, y_pred, average='binary', pos_label=1)
    else:
        # 多分类问题使用macro平均
        precision = precision_score(y, y_pred, average='macro')
        recall = recall_score(y, y_pred, average='macro')
        f1 = f1_score(y, y_pred, average='macro')

    # 计算混淆矩阵
    cm = confusion_matrix(y, y_pred)

    # 返回评估结果
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "y_true": y,
        "y_pred": y_pred
    }


def plot_comparison(results_tri, results_bin):
    """
    绘制模型比较图

    参数:
        results_tri: 三分类模型评估结果
        results_bin: 二分类模型评估结果
    """
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 绘制评估指标比较
    metrics = ["accuracy", "precision", "recall", "f1"]
    tri_values = [results_tri[m] for m in metrics]
    bin_values = [results_bin[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    axes[0].bar(x - width/2, tri_values, width, label='三分类模型')
    axes[0].bar(x + width/2, bin_values, width, label='二分类模型')

    axes[0].set_ylabel('得分')
    axes[0].set_title('模型性能比较')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 为每个柱子添加数值标签
    for i, v in enumerate(tri_values):
        axes[0].text(i - width/2, v + 0.01, f'{v:.2f}', ha='center')

    for i, v in enumerate(bin_values):
        axes[0].text(i + width/2, v + 0.01, f'{v:.2f}', ha='center')

    # 绘制类别分布比较
    tri_class_counts = np.bincount(results_tri["y_true"].astype(int))
    bin_class_counts = np.bincount(results_bin["y_true"].astype(int))

    tri_pred_counts = np.bincount(results_tri["y_pred"].astype(int))
    bin_pred_counts = np.bincount(results_bin["y_pred"].astype(int))

    # 确保两个数组长度相同
    max_len = max(len(tri_pred_counts), len(bin_pred_counts))
    if len(tri_pred_counts) < max_len:
        tri_pred_counts = np.pad(tri_pred_counts, (0, max_len - len(tri_pred_counts)))
    if len(bin_pred_counts) < max_len:
        bin_pred_counts = np.pad(bin_pred_counts, (0, max_len - len(bin_pred_counts)))

    # 计算预测的类别比例
    tri_pred_ratio = tri_pred_counts / tri_pred_counts.sum()
    bin_pred_ratio = bin_pred_counts / bin_pred_counts.sum()

    # 绘制预测类别比例
    labels = ['类别 0', '类别 1']
    x = np.arange(len(labels))

    axes[1].bar(x - width/2, tri_pred_ratio[:2], width, label='三分类模型')
    axes[1].bar(x + width/2, bin_pred_ratio[:2], width, label='二分类模型')

    axes[1].set_ylabel('预测比例')
    axes[1].set_title('模型预测类别分布')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 为每个柱子添加数值标签
    for i, v in enumerate(tri_pred_ratio[:2]):
        axes[1].text(i - width/2, v + 0.01, f'{v:.2f}', ha='center')

    for i, v in enumerate(bin_pred_ratio[:2]):
        axes[1].text(i + width/2, v + 0.01, f'{v:.2f}', ha='center')

    plt.tight_layout()

    # 保存图表
    os.makedirs('output/comparison', exist_ok=True)
    plt.savefig('output/comparison/model_comparison.png')
    print("比较图表已保存到 output/comparison/model_comparison.png")

    # 显示图表
    plt.show()


def main():
    """主函数"""
    # 三分类模型路径
    tri_model_path = "output/models/multirocket_hydra_20250501_185639/model.pkl"
    tri_test_data_path = "output/prepared_data/test_dataset.npz"

    # 二分类模型路径
    bin_model_path = "output/models/binary/multirocket_hydra_20250501_191701/model.pkl"
    bin_test_data_path = "output/prepared_data/binary/test_dataset.npz"

    # 评估三分类模型
    print("评估三分类模型...")
    results_tri = evaluate_model(tri_model_path, tri_test_data_path)

    # 评估二分类模型
    print("评估二分类模型...")
    results_bin = evaluate_model(bin_model_path, bin_test_data_path, binary=True)

    # 打印评估结果
    print("\n===== 三分类模型评估结果 =====")
    print(f"准确率: {results_tri['accuracy']:.4f}")
    print(f"精确率: {results_tri['precision']:.4f}")
    print(f"召回率: {results_tri['recall']:.4f}")
    print(f"F1分数: {results_tri['f1']:.4f}")
    print("混淆矩阵:")
    print(results_tri['confusion_matrix'])

    print("\n===== 二分类模型评估结果 =====")
    print(f"准确率: {results_bin['accuracy']:.4f}")
    print(f"精确率: {results_bin['precision']:.4f}")
    print(f"召回率: {results_bin['recall']:.4f}")
    print(f"F1分数: {results_bin['f1']:.4f}")
    print("混淆矩阵:")
    print(results_bin['confusion_matrix'])

    # 绘制比较图
    plot_comparison(results_tri, results_bin)


if __name__ == "__main__":
    main()
