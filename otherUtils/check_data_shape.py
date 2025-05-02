"""
检查数据形状
"""
import numpy as np
from src.data.data_loader import load_custom_dataset

def main():
    """主函数"""
    # 加载训练数据
    train_path = "output/prepared_data/train_dataset.npz"
    print(f"加载训练数据: {train_path}")
    X_train, y_train = load_custom_dataset(train_path)
    print(f"训练数据形状: X={X_train.shape}, y={y_train.shape}")
    
    # 加载测试数据
    test_path = "output/prepared_data/test_dataset.npz"
    print(f"加载测试数据: {test_path}")
    X_test, y_test = load_custom_dataset(test_path)
    print(f"测试数据形状: X={X_test.shape}, y={y_test.shape}")
    
    # 加载模型数据
    model_path = "output/multirocket_hydra_20250502_094059/model.pkl"
    print(f"模型路径: {model_path}")

if __name__ == "__main__":
    main()
