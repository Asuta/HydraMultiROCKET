"""
特征提取器模块

此模块用于从K线数据中提取时间序列片段，并进行标准化处理。
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Union, Optional
from sklearn.preprocessing import StandardScaler


def extract_time_series_segments(df: pd.DataFrame,
                                signal_indices: List[int],
                                segment_length: int = 40,
                                feature_columns: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    从K线数据中提取信号点前的时间序列片段

    参数:
        df: 包含K线数据的DataFrame
        signal_indices: 信号点的索引列表
        segment_length: 每个片段的长度（K线数量）
        feature_columns: 要提取的特征列名列表，如果为None，则使用默认特征

    返回:
        X: 提取的时间序列片段，形状为 (n_samples, segment_length, n_features)
        y: 对应的标签（信号成功或失败），形状为 (n_samples,)
    """
    # 如果未指定特征列，则使用默认特征
    if feature_columns is None:
        feature_columns = ['open', 'high', 'low', 'close', 'volume']

    # 检查必要的列是否存在
    for col in feature_columns + ['success']:
        if col not in df.columns:
            raise ValueError(f"输入DataFrame缺少必要的列: {col}")

    # 初始化结果列表
    X_segments = []
    y_labels = []

    # 对每个信号点提取时间序列片段
    for idx in signal_indices:
        # 确保有足够的历史数据
        if idx < segment_length:
            continue

        # 确保信号已标注
        if pd.isna(df.loc[idx, 'success']):
            continue

        # 提取信号点前的时间序列片段（包含信号点）
        segment = df.iloc[idx-segment_length+1:idx+1][feature_columns].values

        # 如果片段长度不足，则跳过
        if len(segment) < segment_length:
            continue

        # 提取标签
        label = df.loc[idx, 'success']

        # 添加到结果列表
        X_segments.append(segment)
        y_labels.append(label)

    # 转换为numpy数组
    if X_segments:
        X = np.array(X_segments)
        y = np.array(y_labels)
        return X, y
    else:
        # 如果没有有效的片段，返回空数组
        n_features = len(feature_columns)
        return np.empty((0, segment_length, n_features)), np.empty((0,))


def normalize_time_series(X: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    对时间序列数据进行标准化处理

    参数:
        X: 时间序列数据，形状为 (n_samples, segment_length, n_features)
        method: 标准化方法，可选 'zscore'（Z-Score标准化）或 'minmax'（最小-最大缩放）

    返回:
        标准化后的时间序列数据，形状与输入相同
    """
    if X.size == 0:
        return X

    n_samples, segment_length, n_features = X.shape
    X_normalized = np.zeros_like(X)

    # 对每个样本的每个特征维度分别进行标准化
    for i in range(n_samples):
        for j in range(n_features):
            series = X[i, :, j]

            if method == 'zscore':
                # Z-Score标准化: (x - mean) / std
                mean = np.mean(series)
                std = np.std(series)
                if std > 0:
                    X_normalized[i, :, j] = (series - mean) / std
                else:
                    X_normalized[i, :, j] = 0  # 如果标准差为0，则将所有值设为0

            elif method == 'minmax':
                # 最小-最大缩放: (x - min) / (max - min)
                min_val = np.min(series)
                max_val = np.max(series)
                if max_val > min_val:
                    X_normalized[i, :, j] = (series - min_val) / (max_val - min_val)
                else:
                    X_normalized[i, :, j] = 0.5  # 如果最大值等于最小值，则将所有值设为0.5

            else:
                raise ValueError(f"不支持的标准化方法: {method}，支持的方法有 'zscore' 和 'minmax'")

    return X_normalized


def prepare_dataset_from_signals(df: pd.DataFrame,
                               segment_length: int = 40,
                               feature_columns: List[str] = None,
                               normalize: bool = True,
                               normalize_method: str = 'zscore',
                               transpose_for_aeon: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    从带有信号的DataFrame中准备数据集

    参数:
        df: 包含信号和标签的DataFrame
        segment_length: 每个片段的长度（K线数量）
        feature_columns: 要提取的特征列名列表
        normalize: 是否对数据进行标准化
        normalize_method: 标准化方法
        transpose_for_aeon: 是否转置数据以适应aeon库的要求

    返回:
        X: 提取的时间序列片段
            如果transpose_for_aeon=False，形状为 (n_samples, segment_length, n_features)
            如果transpose_for_aeon=True，形状为 (n_samples, n_features, segment_length)
        y: 对应的标签（信号成功或失败），形状为 (n_samples,)
    """
    # 获取所有有效信号的索引
    valid_signals = df.dropna(subset=['success'])
    signal_indices = valid_signals.index.tolist()

    # 提取时间序列片段
    X, y = extract_time_series_segments(
        df=df,
        signal_indices=signal_indices,
        segment_length=segment_length,
        feature_columns=feature_columns
    )

    # 标准化数据
    if normalize and X.size > 0:
        X = normalize_time_series(X, method=normalize_method)

    # 转置数据以适应aeon库的要求
    # aeon库期望的形状是 (n_samples, n_features, segment_length)
    if transpose_for_aeon and X.size > 0:
        X = np.transpose(X, (0, 2, 1))
        print(f"数据已转置，新形状: {X.shape}")

    return X, y
