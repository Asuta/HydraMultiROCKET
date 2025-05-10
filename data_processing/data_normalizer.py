"""
数据标准化模块

此模块用于对时间序列数据进行标准化处理
"""
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class TimeSeriesNormalizer:
    """
    时间序列标准化器

    用于对多维时间序列数据进行标准化处理，支持Z-Score和Min-Max标准化
    """

    def __init__(self, method: str = "zscore"):
        """
        初始化标准化器

        参数:
            method: 标准化方法，可选 "zscore" 或 "minmax"
        """
        self.method = method
        self.scalers = {}  # 每个特征维度一个缩放器
        self.n_features = None

    def fit(self, X: np.ndarray) -> 'TimeSeriesNormalizer':
        """
        拟合标准化器

        参数:
            X: 时间序列数据，形状为 (n_samples, n_timepoints, n_features) 或 (n_samples, n_features, n_timepoints)

        返回:
            self
        """
        # 确定特征维度
        if X.ndim == 3:
            if X.shape[1] > X.shape[2]:
                # 形状为 (n_samples, n_timepoints, n_features)
                self.n_features = X.shape[2]
                self.transpose_needed = False
            else:
                # 形状为 (n_samples, n_features, n_timepoints)
                self.n_features = X.shape[1]
                self.transpose_needed = True
        else:
            raise ValueError("输入数据必须是3维的，形状为 (n_samples, n_timepoints, n_features) 或 (n_samples, n_features, n_timepoints)")

        # 为每个特征维度创建一个缩放器
        for i in range(self.n_features):
            if self.method == "zscore":
                self.scalers[i] = StandardScaler()
            elif self.method == "minmax":
                self.scalers[i] = MinMaxScaler()
            else:
                raise ValueError(f"不支持的标准化方法: {self.method}")

            # 提取该特征维度的所有样本和时间点
            if self.transpose_needed:
                # 形状为 (n_samples, n_features, n_timepoints)
                feature_data = X[:, i, :].reshape(-1, 1)
            else:
                # 形状为 (n_samples, n_timepoints, n_features)
                feature_data = X[:, :, i].reshape(-1, 1)

            # 拟合缩放器
            self.scalers[i].fit(feature_data)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        转换数据

        参数:
            X: 时间序列数据，形状为 (n_samples, n_timepoints, n_features) 或 (n_samples, n_features, n_timepoints)

        返回:
            标准化后的数据，与输入形状相同
        """
        if self.n_features is None:
            raise ValueError("必须先调用fit方法")

        # 创建输出数组
        X_transformed = np.zeros_like(X)

        # 对每个特征维度进行转换
        for i in range(self.n_features):
            # 提取该特征维度的所有样本和时间点
            if self.transpose_needed:
                # 形状为 (n_samples, n_features, n_timepoints)
                feature_data = X[:, i, :].reshape(-1, 1)

                # 转换数据
                transformed_data = self.scalers[i].transform(feature_data)

                # 重塑回原始形状
                X_transformed[:, i, :] = transformed_data.reshape(X.shape[0], X.shape[2])
            else:
                # 形状为 (n_samples, n_timepoints, n_features)
                feature_data = X[:, :, i].reshape(-1, 1)

                # 转换数据
                transformed_data = self.scalers[i].transform(feature_data)

                # 重塑回原始形状
                X_transformed[:, :, i] = transformed_data.reshape(X.shape[0], X.shape[1])

        return X_transformed

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        拟合并转换数据

        参数:
            X: 时间序列数据，形状为 (n_samples, n_timepoints, n_features) 或 (n_samples, n_features, n_timepoints)

        返回:
            标准化后的数据，与输入形状相同
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        逆转换数据

        参数:
            X: 标准化后的数据，形状为 (n_samples, n_timepoints, n_features) 或 (n_samples, n_features, n_timepoints)

        返回:
            逆转换后的数据，与输入形状相同
        """
        if self.n_features is None:
            raise ValueError("必须先调用fit方法")

        # 创建输出数组
        X_inverse = np.zeros_like(X)

        # 对每个特征维度进行逆转换
        for i in range(self.n_features):
            # 提取该特征维度的所有样本和时间点
            if self.transpose_needed:
                # 形状为 (n_samples, n_features, n_timepoints)
                feature_data = X[:, i, :].reshape(-1, 1)

                # 逆转换数据
                inverse_data = self.scalers[i].inverse_transform(feature_data)

                # 重塑回原始形状
                X_inverse[:, i, :] = inverse_data.reshape(X.shape[0], X.shape[2])
            else:
                # 形状为 (n_samples, n_timepoints, n_features)
                feature_data = X[:, :, i].reshape(-1, 1)

                # 逆转换数据
                inverse_data = self.scalers[i].inverse_transform(feature_data)

                # 重塑回原始形状
                X_inverse[:, :, i] = inverse_data.reshape(X.shape[0], X.shape[1])

        return X_inverse


def normalize_dataset(
    X_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    method: str = "zscore"
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], TimeSeriesNormalizer]:
    """
    标准化数据集

    参数:
        X_train: 训练集
        X_val: 验证集（可选）
        X_test: 测试集（可选）
        method: 标准化方法，可选 "zscore" 或 "minmax"

    返回:
        X_train_norm: 标准化后的训练集
        X_val_norm: 标准化后的验证集（如果提供）
        X_test_norm: 标准化后的测试集（如果提供）
        normalizer: 标准化器
    """
    # 创建标准化器
    normalizer = TimeSeriesNormalizer(method=method)

    # 在训练集上拟合并转换
    X_train_norm = normalizer.fit_transform(X_train)

    # 转换验证集和测试集（如果提供）
    X_val_norm = normalizer.transform(X_val) if X_val is not None else None
    X_test_norm = normalizer.transform(X_test) if X_test is not None else None

    return X_train_norm, X_val_norm, X_test_norm, normalizer


def normalize_samples_individually(X: np.ndarray, volume_feature_idx: int = 4) -> np.ndarray:
    """
    对每个样本单独进行标准化，价格相关特征和交易量分别标准化

    参数:
        X: 特征数组，形状为 (n_samples, n_features, segment_length)
        volume_feature_idx: 交易量特征的索引，默认为4（对应feature_columns中的'volume'）

    返回:
        X_normalized: 标准化后的特征数组，形状与输入相同
    """
    n_samples, n_features, segment_length = X.shape
    X_normalized = np.zeros_like(X)

    # 对每个样本单独进行标准化
    for i in range(n_samples):
        # 获取当前样本
        sample = X[i]

        # 1. 对价格相关特征（除了交易量）进行统一标准化
        price_features = np.delete(sample, volume_feature_idx, axis=0)

        # 计算价格特征的均值和标准差
        price_mean = np.mean(price_features)
        price_std = np.std(price_features)

        # 如果标准差接近0，设置为1以避免除以0
        if price_std < 1e-10:
            price_std = 1.0

        # 对价格特征进行标准化
        for j in range(n_features):
            if j != volume_feature_idx:
                X_normalized[i, j] = (sample[j] - price_mean) / price_std

        # 2. 单独对交易量特征进行标准化
        volume = sample[volume_feature_idx]
        volume_mean = np.mean(volume)
        volume_std = np.std(volume)

        # 如果标准差接近0，设置为1以避免除以0
        if volume_std < 1e-10:
            volume_std = 1.0

        # 对交易量进行标准化
        X_normalized[i, volume_feature_idx] = (volume - volume_mean) / volume_std

    return X_normalized
