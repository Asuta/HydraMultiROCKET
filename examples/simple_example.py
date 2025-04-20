"""
简单示例：使用Hydra+MultiROCKET进行时间序列分类
"""
import numpy as np
from aeon.datasets import load_italy_power_demand
from aeon.classification.convolution_based import MultiRocketHydraClassifier
from sklearn.metrics import accuracy_score

# 加载数据
print("加载数据...")
X_train, y_train = load_italy_power_demand(split="train")
X_test, y_test = load_italy_power_demand(split="test")

print(f"训练数据形状: {X_train.shape}, 测试数据形状: {X_test.shape}")

# 创建模型
print("创建模型...")
model = MultiRocketHydraClassifier(n_kernels=1000)  # 使用较少的卷积核以加快示例运行速度

# 训练模型
print("训练模型...")
model.fit(X_train, y_train)

# 预测
print("进行预测...")
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")

print("示例完成!")
