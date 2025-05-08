"""
推理脚本
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.models.hydra_multirocket import HydraMultiRocketModel
from src.data.data_loader import load_dataset, load_custom_dataset
from src.utils.metrics import calculate_metrics, print_metrics, print_classification_report, print_confusion_matrix
from src.visualization.visualize import plot_confusion_matrix
from src.visualization.fix_chinese_font import fix_chinese_display

# 修复中文显示问题
fix_chinese_display()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用Hydra+MultiROCKET模型进行推理')
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径')
    parser.add_argument('--dataset', type=str, default='italy_power_demand', help='数据集名称')
    parser.add_argument('--custom_dataset_path', type=str, help='自定义数据集路径')
    parser.add_argument('--output_dir', type=str, default='output/predictions/results', help='输出目录')
    parser.add_argument('--output_proba', action='store_true', help='是否输出预测概率')
    parser.add_argument('--save_results', action='store_true', help='是否保存预测结果到文件')
    parser.add_argument('--show_all_proba', action='store_true', help='是否显示所有样本的预测概率')
    parser.add_argument('--save_csv', action='store_true', help='是否将预测结果保存为CSV文件')
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    print(f"加载模型: {args.model_path}")
    model = HydraMultiRocketModel.load(args.model_path)

    # 加载数据
    if args.custom_dataset_path:
        print(f"从自定义路径加载数据: {args.custom_dataset_path}")
        X, y = load_custom_dataset(args.custom_dataset_path)
    else:
        print(f"加载内置数据集: {args.dataset}")
        X, y = load_dataset(args.dataset, split="test")

    print(f"数据形状: {X.shape}")

    # 进行预测
    print("进行预测")
    y_pred = model.predict(X)

    # 如果需要，计算预测概率
    if args.output_proba:
        print("计算预测概率")
        y_proba = model.predict_proba(X)
        print(f"预测概率形状: {y_proba.shape}")

        # 打印样本的预测概率
        if args.show_all_proba:
            print("\n所有样本的预测概率:")
            for i in range(len(y_proba)):
                print(f"样本 {i}: 真实标签 = {y[i]}, 预测标签 = {y_pred[i]}, 预测概率 = {y_proba[i]}")
        else:
            # 只打印前5个样本
            n_samples = min(5, len(y_proba))
            print(f"\n前{n_samples}个样本的预测概率:")
            for i in range(n_samples):
                print(f"样本 {i}: 真实标签 = {y[i]}, 预测标签 = {y_pred[i]}, 预测概率 = {y_proba[i]}")

        # 如果需要保存结果
        if args.save_results:
            results_path = os.path.join(args.output_dir, "prediction_results.npz")
            np.savez(
                results_path,
                y_true=y,
                y_pred=y_pred,
                y_proba=y_proba
            )
            print(f"预测结果已保存到: {results_path}")

        # 如果需要保存为CSV文件
        if args.save_csv:
            # 创建一个包含所有预测信息的DataFrame
            results_df = pd.DataFrame()
            results_df['sample_id'] = np.arange(len(y))
            results_df['true_label'] = y
            results_df['predicted_label'] = y_pred

            # 添加每个类别的预测概率
            for i in range(y_proba.shape[1]):
                results_df[f'probability_class_{i}'] = y_proba[:, i]

            # 保存为CSV文件
            csv_path = os.path.join(args.output_dir, "prediction_results.csv")
            results_df.to_csv(csv_path, index=False)
            print(f"预测结果已保存为CSV文件: {csv_path}")

    # 计算并打印指标
    metrics = calculate_metrics(y, y_pred)
    print_metrics(metrics)
    print_classification_report(y, y_pred)
    print_confusion_matrix(y, y_pred)

    # 可视化混淆矩阵
    cm_save_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(y, y_pred, title="预测混淆矩阵", save_path=cm_save_path)

    print("预测完成")


if __name__ == "__main__":
    main()
