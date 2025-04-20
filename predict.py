"""
推理脚本
"""
import os
import argparse
import numpy as np

from src.models.hydra_multirocket import HydraMultiRocketModel
from src.data.data_loader import load_dataset, load_custom_dataset
from src.utils.metrics import calculate_metrics, print_metrics, print_classification_report, print_confusion_matrix
from src.visualization.visualize import plot_confusion_matrix


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用Hydra+MultiROCKET模型进行推理')
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径')
    parser.add_argument('--dataset', type=str, default='italy_power_demand', help='数据集名称')
    parser.add_argument('--custom_dataset_path', type=str, help='自定义数据集路径')
    parser.add_argument('--output_dir', type=str, default='output/predictions', help='输出目录')
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
