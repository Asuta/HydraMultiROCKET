"""
Hydra配置示例：使用Hydra进行配置管理
"""
import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import load_dataset
from src.models.hydra_multirocket import HydraMultiRocketModel
from src.utils.metrics import calculate_metrics, print_metrics


@hydra.main(config_path="../config", config_name="default")
def main(cfg: DictConfig):
    """
    使用Hydra配置的主函数
    
    参数:
        cfg: Hydra配置对象
    """
    print("配置:")
    print(OmegaConf.to_yaml(cfg))
    
    # 加载数据
    print("\n加载数据...")
    X_train, y_train = load_dataset(cfg.data.dataset_name, split="train")
    X_test, y_test = load_dataset(cfg.data.dataset_name, split="test")
    
    print(f"训练数据形状: {X_train.shape}, 测试数据形状: {X_test.shape}")
    
    # 创建模型
    print("\n创建模型...")
    model = HydraMultiRocketModel(
        model_type=cfg.model.model_type,
        n_kernels=cfg.model.n_kernels,
        max_dilations_per_kernel=cfg.model.max_dilations_per_kernel,
        n_features_per_kernel=cfg.model.n_features_per_kernel,
        random_state=cfg.model.random_state
    )
    
    # 训练模型
    print("\n训练模型...")
    model.fit(X_train, y_train)
    
    # 预测
    print("\n进行预测...")
    y_pred = model.predict(X_test)
    
    # 计算并打印指标
    metrics = calculate_metrics(y_test, y_pred)
    print_metrics(metrics)
    
    print("\n示例完成!")


if __name__ == "__main__":
    main()
