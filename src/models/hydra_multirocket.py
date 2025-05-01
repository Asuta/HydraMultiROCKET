"""
Hydra+MultiROCKET 模型实现
"""
import os
import pickle
import numpy as np
from typing import Optional, Dict, Any, Union, Tuple

from aeon.classification.convolution_based import (
    HydraClassifier,
    MultiRocketClassifier,
    MultiRocketHydraClassifier
)


class HydraMultiRocketModel:
    """
    Hydra+MultiROCKET 模型封装类
    """

    def __init__(self,
                 model_type: str = "multirocket_hydra",
                 n_kernels: int = 10000,
                 n_groups: int = 64,
                 max_dilations_per_kernel: int = 32,
                 n_features_per_kernel: int = 4,
                 random_state: Optional[int] = None,
                 n_jobs: int = 1,
                 **kwargs):
        """
        初始化模型

        参数:
            model_type: 模型类型，可选 "hydra", "multirocket" 或 "multirocket_hydra"
            n_kernels: 卷积核数量
            n_groups: Hydra变换的每个扩张的组数
            max_dilations_per_kernel: 每个卷积核的最大扩张数 (仅用于MultiRocket)
            n_features_per_kernel: 每个卷积核的特征数 (仅用于MultiRocket)
            random_state: 随机种子
            n_jobs: 并行作业数
            **kwargs: 其他参数
        """
        self.model_type = model_type
        self.n_kernels = n_kernels
        self.n_groups = n_groups
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.n_features_per_kernel = n_features_per_kernel
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        self.model = self._create_model()

    def _create_model(self):
        """
        创建模型实例

        返回:
            模型实例
        """
        if self.model_type == "hydra":
            return HydraClassifier(
                n_kernels=self.n_kernels,
                n_groups=self.n_groups,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                **self.kwargs
            )
        elif self.model_type == "multirocket":
            return MultiRocketClassifier(
                n_kernels=self.n_kernels,
                max_dilations_per_kernel=self.max_dilations_per_kernel,
                n_features_per_kernel=self.n_features_per_kernel,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                **self.kwargs
            )
        elif self.model_type == "multirocket_hydra":
            return MultiRocketHydraClassifier(
                n_kernels=self.n_kernels,
                n_groups=self.n_groups,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                **self.kwargs
            )
        else:
            raise ValueError(f"未知模型类型: {self.model_type}")

    def fit(self, X, y):
        """
        训练模型

        参数:
            X: 训练数据
            y: 训练标签

        返回:
            self
        """
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        预测类别

        参数:
            X: 测试数据

        返回:
            预测的类别
        """
        return self.model.predict(X)

    # 移除概率预测功能

    def save(self, file_path: str):
        """
        保存模型

        参数:
            file_path: 保存路径
        """
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # 保存模型
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

        print(f"模型已保存到 {file_path}")

    @classmethod
    def load(cls, file_path: str):
        """
        加载模型

        参数:
            file_path: 模型文件路径

        返回:
            加载的模型
        """
        with open(file_path, 'rb') as f:
            model = pickle.load(f)

        return model

    def get_params(self):
        """
        获取模型参数

        返回:
            模型参数
        """
        params = {
            'model_type': self.model_type,
            'n_kernels': self.n_kernels,
            'n_groups': self.n_groups,
            'max_dilations_per_kernel': self.max_dilations_per_kernel,
            'n_features_per_kernel': self.n_features_per_kernel,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            **self.kwargs
        }
        return params
