"""
配置模块
"""
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from omegaconf import OmegaConf


@dataclass
class DataConfig:
    """数据配置"""
    dataset_name: str = "italy_power_demand"  # 数据集名称
    custom_dataset_path: Optional[str] = None  # 自定义数据集路径（训练集）
    val_dataset_path: Optional[str] = None  # 验证集路径
    test_dataset_path: Optional[str] = None  # 测试集路径
    test_size: float = 0.2  # 测试集比例
    random_state: int = 42  # 随机种子
    save_path: Optional[str] = None  # 数据保存路径


@dataclass
class ModelConfig:
    """模型配置"""
    model_type: str = "multirocket_hydra"  # 模型类型
    n_kernels: int = 8  # 卷积核数量
    n_groups: int = 32  # 每个扩张的组数
    max_dilations_per_kernel: int = 32  # 每个卷积核的最大扩张数
    n_features_per_kernel: int = 4  # 每个卷积核的特征数
    random_state: Optional[int] = None  # 随机种子
    n_jobs: int = 1  # 并行作业数
    use_calibrated_classifier: bool = False  # 是否使用校准后的分类器以获得更好的概率估计
    cv_folds: int = 5  # 校准分类器时使用的交叉验证折数
    save_path: str = "models/hydra_multirocket.pkl"  # 模型保存路径


@dataclass
class TrainingConfig:
    """训练配置"""
    epochs: int = 1  # 训练轮数（对于这种模型通常只需要一轮）
    verbose: bool = True  # 是否显示详细信息
    save_best: bool = True  # 是否保存最佳模型


@dataclass
class Config:
    """总配置"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: str = "output"  # 输出目录


def load_config(config_path: str) -> Config:
    """
    加载配置文件

    参数:
        config_path: 配置文件路径

    返回:
        配置对象
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    cfg = OmegaConf.load(config_path)
    config = OmegaConf.structured(Config)
    config = OmegaConf.merge(config, cfg)

    return config


def save_config(config: Config, config_path: str) -> None:
    """
    保存配置到文件

    参数:
        config: 配置对象
        config_path: 保存路径
    """
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    # 保存配置
    OmegaConf.save(config, config_path)

    print(f"配置已保存到 {config_path}")


def get_default_config() -> Config:
    """
    获取默认配置

    返回:
        默认配置对象
    """
    return Config()
