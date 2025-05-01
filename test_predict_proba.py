"""
测试概率预测功能
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.models.hydra_multirocket import HydraMultiRocketModel
from src.data.data_loader import load_custom_dataset


def main():
    """主函数"""
    # 加载模型
    model_path = "output/models/multirocket_hydra_20250501_163949/model.pkl"
    print(f"加载模型: {model_path}")
    model = HydraMultiRocketModel.load(model_path)
    
    # 加载测试数据
    data_path = "output/prepared_data/test_dataset.npz"
    print(f"加载测试数据: {data_path}")
    X, y = load_custom_dataset(data_path)
    
    # 检查是否存在无效类别（标签为2）
    if 2 in y:
        print("检测到无效类别（标签为2），将其过滤掉")
        valid_mask = (y != 2)
        X = X[valid_mask]
        y = y[valid_mask]
        print(f"过滤后的数据形状: {X.shape}")
    
    # 进行预测
    print("进行预测...")
    y_pred = model.predict(X)
    
    # 进行概率预测
    print("进行概率预测...")
    try:
        y_proba = model.predict_proba(X)
        print(f"概率预测形状: {y_proba.shape}")
        
        # 打印前5个样本的概率预测结果
        print("\n前5个样本的概率预测结果:")
        for i in range(min(5, len(y))):
            print(f"样本 {i+1}:")
            print(f"  真实标签: {y[i]}")
            print(f"  预测标签: {y_pred[i]}")
            print(f"  预测概率: {y_proba[i]}")
        
        # 绘制概率分布图
        plt.figure(figsize=(10, 6))
        
        # 计算每个类别的平均概率
        class_labels = np.unique(y)
        avg_proba = np.zeros((len(class_labels), y_proba.shape[1]))
        
        for i, label in enumerate(class_labels):
            mask = (y == label)
            avg_proba[i] = np.mean(y_proba[mask], axis=0)
        
        # 绘制条形图
        x = np.arange(len(class_labels))
        width = 0.2
        
        for i in range(y_proba.shape[1]):
            plt.bar(x + i*width, avg_proba[:, i], width, label=f'类别 {i}的概率')
        
        plt.xlabel('真实类别')
        plt.ylabel('平均预测概率')
        plt.title('各类别的平均预测概率分布')
        plt.xticks(x + width, [f'类别 {int(label)}' for label in class_labels])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图表
        os.makedirs('output/predictions', exist_ok=True)
        plt.savefig('output/predictions/probability_distribution.png')
        print("概率分布图已保存到 output/predictions/probability_distribution.png")
        
        # 显示图表
        plt.show()
        
    except NotImplementedError as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()
