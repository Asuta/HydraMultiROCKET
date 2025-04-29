"""
测试概率预测功能
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.models.hydra_multirocket import HydraMultiRocketModel
from src.data.data_loader import load_custom_dataset

def main():
    """主函数"""
    # 加载模型
    model_path = "output/multirocket_hydra_20250429_220549/model.pkl"
    print(f"加载模型: {model_path}")
    model = HydraMultiRocketModel.load(model_path)
    
    # 加载测试数据
    data_path = "output/prepared_data/test_dataset.npz"
    print(f"加载测试数据: {data_path}")
    X, y = load_custom_dataset(data_path)
    
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
        
        plt.title('预测为成功的概率分布')
        plt.xlabel('预测为成功的概率')
        plt.ylabel('样本数量')
        plt.legend()
        plt.grid(True)
        
        # 保存图表
        plt.savefig('output/probability_distribution.png')
        print("概率分布图已保存到 output/probability_distribution.png")
        
    except NotImplementedError as e:
        print(f"错误: {e}")
        print("该模型不支持概率预测")

if __name__ == "__main__":
    main()
