"""
测试概率预测功能（使用2024年数据）
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.models.hydra_multirocket import HydraMultiRocketModel
from src.data.data_loader import load_custom_dataset

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def main():
    """主函数"""
    # 加载模型
    model_path = "output/models_new/multirocket_hydra_20250429_222207/model.pkl"
    print(f"加载模型: {model_path}")
    model = HydraMultiRocketModel.load(model_path)

    # 加载测试数据
    data_path = "output/prepared_data_2024/test_dataset.npz"
    print(f"加载测试数据: {data_path}")
    X, y = load_custom_dataset(data_path)

    # 检查是否存在无效类别（标签为2）
    if 2 in y:
        print("检测到无效类别（标签为2），将其过滤掉")
        valid_mask = (y != 2)
        X = X[valid_mask]
        y = y[valid_mask]
        print(f"过滤后的数据形状: {X.shape}")

    print(f"数据形状: {X.shape}, 标签形状: {y.shape}")

    # 预测类别
    print("预测类别...")
    y_pred = model.predict(X)

    # 预测概率
    print("预测概率...")
    try:
        y_proba = model.predict_proba(X)
        print(f"概率形状: {y_proba.shape}")

        # 打印前10个样本的预测概率
        print("\n前10个样本的预测概率:")
        for i in range(min(10, len(y))):
            print(f"样本 {i+1}: 标签 = {y[i]}, 预测 = {y_pred[i]}, 概率 = {y_proba[i]}")

        # 绘制概率分布图
        plt.figure(figsize=(10, 6))

        # 获取正类（标签1）的概率
        proba_class_1 = y_proba[:, 1] if y_proba.shape[1] > 1 else 1 - y_proba[:, 0]

        # 根据真实标签分组
        proba_class_1_true_0 = proba_class_1[y == 0]
        proba_class_1_true_1 = proba_class_1[y == 1]

        # 绘制直方图
        plt.hist(proba_class_1_true_0, bins=20, alpha=0.5, label='真实标签 = 0 (失败)')
        plt.hist(proba_class_1_true_1, bins=20, alpha=0.5, label='真实标签 = 1 (成功)')

        plt.title('预测为成功的概率分布 (2024年数据)')
        plt.xlabel('预测为成功的概率')
        plt.ylabel('样本数量')
        plt.legend()
        plt.grid(True)

        # 保存图表
        plt.savefig('output/probability_distribution_2024.png')
        print("概率分布图已保存到 output/probability_distribution_2024.png")

        # 计算高概率预测的准确率
        threshold = 0.7  # 设置概率阈值
        high_prob_mask = np.max(y_proba, axis=1) >= threshold
        if np.sum(high_prob_mask) > 0:
            high_prob_X = X[high_prob_mask]
            high_prob_y = y[high_prob_mask]
            high_prob_pred = y_pred[high_prob_mask]
            high_prob_accuracy = np.mean(high_prob_y == high_prob_pred)
            print(f"\n高概率预测 (>= {threshold}) 的样本数: {np.sum(high_prob_mask)}")
            print(f"高概率预测的准确率: {high_prob_accuracy:.4f}")
        else:
            print(f"\n没有概率 >= {threshold} 的预测")

    except NotImplementedError as e:
        print(f"错误: {e}")
        print("该模型不支持概率预测")

if __name__ == "__main__":
    main()
