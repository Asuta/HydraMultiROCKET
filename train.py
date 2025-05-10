"""
训练脚本
"""
import os
import argparse
import numpy as np
from datetime import datetime
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

from src.config.config import load_config, get_default_config, save_config
from src.data.data_loader import load_dataset, load_custom_dataset, split_dataset
from src.models.hydra_multirocket import HydraMultiRocketModel
from src.utils.metrics import calculate_metrics, print_metrics, print_classification_report, print_confusion_matrix
from src.utils.logger import setup_logger
from src.visualization.visualize import plot_time_series, plot_confusion_matrix
from src.visualization.fix_chinese_font import fix_chinese_display

# 修复中文显示问题
fix_chinese_display()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练Hydra+MultiROCKET模型')
    parser.add_argument('--config', type=str, default='config/custom.yaml', help='配置文件路径')
    parser.add_argument('--dataset', type=str, help='数据集名称，覆盖配置文件中的设置')
    parser.add_argument('--model_type', type=str, help='模型类型，覆盖配置文件中的设置')
    parser.add_argument('--output_dir', type=str, help='输出目录，覆盖配置文件中的设置')
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 加载配置
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"配置文件不存在: {args.config}，使用默认配置")
        config = get_default_config()

    # 命令行参数覆盖配置文件
    if args.dataset:
        config.data.dataset_name = args.dataset
    if args.model_type:
        config.model.model_type = args.model_type
    if args.output_dir:
        config.output_dir = args.output_dir

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.output_dir, f"{config.model.model_type}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # 保存配置
    config_save_path = os.path.join(output_dir, "config.yaml")
    save_config(config, config_save_path)

    # 设置日志
    log_file = os.path.join(output_dir, "train.log")
    logger = setup_logger("train", log_file)

    logger.info(f"配置: {OmegaConf.to_yaml(config)}")

    # 加载数据
    logger.info(f"加载数据集: {config.data.dataset_name}")
    if config.data.custom_dataset_path:
        logger.info(f"从自定义路径加载: {config.data.custom_dataset_path}")

        # 检查是否提供了验证集和测试集路径
        if hasattr(config.data, 'val_dataset_path') and hasattr(config.data, 'test_dataset_path'):
            logger.info(f"加载训练集: {config.data.custom_dataset_path}")
            X_train, y_train = load_custom_dataset(config.data.custom_dataset_path)

            logger.info(f"加载验证集: {config.data.val_dataset_path}")
            X_val, y_val = load_custom_dataset(config.data.val_dataset_path)

            logger.info(f"加载测试集: {config.data.test_dataset_path}")
            X_test, y_test = load_custom_dataset(config.data.test_dataset_path)
        else:
            # 如果没有提供验证集和测试集路径，则从训练集中划分
            X, y = load_custom_dataset(config.data.custom_dataset_path)
            X_train, X_test, y_train, y_test = split_dataset(
                X, y,
                test_size=config.data.test_size,
                random_state=config.data.random_state
            )
            X_val, y_val = X_test, y_test  # 使用测试集作为验证集
    else:
        logger.info(f"加载内置数据集: {config.data.dataset_name}")
        X_train, y_train = load_dataset(config.data.dataset_name, split="train")
        X_test, y_test = load_dataset(config.data.dataset_name, split="test")
        X_val, y_val = X_test, y_test  # 使用测试集作为验证集

    logger.info(f"训练数据形状: {X_train.shape}, 验证数据形状: {X_val.shape}, 测试数据形状: {X_test.shape}")

    # 可视化部分数据
    plot_save_path = os.path.join(output_dir, "time_series_samples.png")
    plot_time_series(X_train, y_train, title="训练数据样本", save_path=plot_save_path)

    # 创建模型
    logger.info(f"创建模型: {config.model.model_type}")
    model = HydraMultiRocketModel(
        model_type=config.model.model_type,
        n_kernels=config.model.n_kernels,
        max_dilations_per_kernel=config.model.max_dilations_per_kernel,
        n_features_per_kernel=config.model.n_features_per_kernel,
        random_state=config.model.random_state
    )

    # 训练模型
    logger.info("开始训练模型")
    model.fit(X_train, y_train)

    # 评估模型
    logger.info("评估模型")
    y_pred = model.predict(X_test)

    # 计算并打印指标
    metrics = calculate_metrics(y_test, y_pred)
    print_metrics(metrics)
    print_classification_report(y_test, y_pred)
    print_confusion_matrix(y_test, y_pred)

    # 记录指标
    for name, value in metrics.items():
        logger.info(f"{name}: {value:.4f}")

    # 可视化混淆矩阵
    cm_save_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_test, y_pred, title="测试集混淆矩阵", save_path=cm_save_path)

    # 保存模型
    model_save_path = os.path.join(output_dir, "model.pkl")
    model.save(model_save_path)
    logger.info(f"模型已保存到: {model_save_path}")

    logger.info("训练完成")


if __name__ == "__main__":
    main()
