"""
自定义模型示例：使用我们封装的HydraMultiRocketModel类
"""
import sys
import os
import numpy as np
from sklearn.metrics import accuracy_score

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import load_dataset
from src.models.hydra_multirocket import HydraMultiRocketModel
from src.utils.metrics import calculate_metrics, print_metrics, print_classification_report

# 加载数据
print("加载数据...")
X_train, y_train = load_dataset("italy_power_demand", split="train")
X_test, y_test = load_dataset("italy_power_demand", split="test")

print(f"训练数据形状: {X_train.shape}, 测试数据形状: {X_test.shape}")

# 创建模型
print("创建模型...")
# 尝试不同的模型类型
models = {
    "hydra": HydraMultiRocketModel(model_type="hydra", n_kernels=8, n_groups=32),
    "multirocket": HydraMultiRocketModel(model_type="multirocket", n_kernels=100),
    "multirocket_hydra": HydraMultiRocketModel(model_type="multirocket_hydra", n_kernels=8, n_groups=32)
}

# 训练和评估每个模型
for name, model in models.items():
    print(f"\n===== 模型: {name} =====")

    # 训练模型
    print("训练模型...")
    model.fit(X_train, y_train)

    # 预测
    print("进行预测...")
    y_pred = model.predict(X_test)

    # 计算并打印指标
    metrics = calculate_metrics(y_test, y_pred)
    print_metrics(metrics)
    print_classification_report(y_test, y_pred)

print("示例完成!")
