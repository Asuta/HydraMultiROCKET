"""
特征提取模块

此模块用于从原始数据中提取特征
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def extract_time_series_segment(
    df: pd.DataFrame,
    index: int,
    segment_length: int,
    feature_columns: List[str]
) -> np.ndarray:
    """
    提取时间序列片段

    参数:
        df: 原始数据DataFrame
        index: 信号索引
        segment_length: 片段长度
        feature_columns: 特征列名列表

    返回:
        时间序列片段数组，形状为 (len(feature_columns), segment_length)
    """
    # 计算起始索引
    start_idx = max(0, index - segment_length + 1)
    
    # 如果没有足够的历史数据，用第一个值填充
    if start_idx > 0 and index - start_idx + 1 < segment_length:
        pad_length = segment_length - (index - start_idx + 1)
        
        # 提取可用的历史数据
        available_segment = df.loc[start_idx:index, feature_columns].values.T
        
        # 创建填充数组
        pad_values = np.repeat(available_segment[:, 0:1], pad_length, axis=1)
        
        # 拼接填充数组和可用数据
        segment = np.concatenate([pad_values, available_segment], axis=1)
    else:
        # 提取片段
        segment = df.loc[start_idx:index, feature_columns].values.T
    
    # 确保片段长度正确
    if segment.shape[1] < segment_length:
        # 如果片段长度不足，用最后一个值填充
        pad_length = segment_length - segment.shape[1]
        pad_values = np.repeat(segment[:, -1:], pad_length, axis=1)
        segment = np.concatenate([segment, pad_values], axis=1)
    elif segment.shape[1] > segment_length:
        # 如果片段长度过长，截取最后segment_length个点
        segment = segment[:, -segment_length:]
    
    return segment


def normalize_segment(
    segment: np.ndarray,
    method: str = "zscore",
    scaler: Optional[Union[StandardScaler, MinMaxScaler]] = None,
    fit: bool = False
) -> Tuple[np.ndarray, Optional[Union[StandardScaler, MinMaxScaler]]]:
    """
    标准化时间序列片段

    参数:
        segment: 时间序列片段，形状为 (n_features, segment_length)
        method: 标准化方法，可选 "zscore" 或 "minmax"
        scaler: 预先拟合的缩放器，如果为None则创建新的
        fit: 是否拟合缩放器

    返回:
        标准化后的片段和缩放器
    """
    # 转置片段以便于标准化（sklearn的缩放器期望样本在行上，特征在列上）
    segment_T = segment.T
    
    # 创建或使用缩放器
    if scaler is None:
        if method == "zscore":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"不支持的标准化方法: {method}")
    
    # 拟合和转换或仅转换
    if fit:
        segment_normalized_T = scaler.fit_transform(segment_T)
    else:
        segment_normalized_T = scaler.transform(segment_T)
    
    # 转置回原始形状
    segment_normalized = segment_normalized_T.T
    
    return segment_normalized, scaler


def prepare_dataset_from_signals(
    df: pd.DataFrame,
    segment_length: int = 40,
    feature_columns: List[str] = ["open", "high", "low", "close", "volume"],
    normalize: bool = True,
    normalize_method: str = "zscore",
    transpose_for_aeon: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从信号数据中准备数据集

    参数:
        df: 包含信号和标签的DataFrame
        segment_length: 时间序列片段长度
        feature_columns: 特征列名列表
        normalize: 是否标准化数据
        normalize_method: 标准化方法，可选 "zscore" 或 "minmax"
        transpose_for_aeon: 是否转置数据以适应aeon库的要求

    返回:
        X: 特征数组，形状为 (n_samples, n_features, segment_length) 或 (n_samples, segment_length, n_features)
        y: 标签数组
    """
    # 获取有效的信号（已标注的信号）
    valid_signals = df.dropna(subset=["success"])
    
    # 初始化特征和标签列表
    X_list = []
    y_list = []
    
    # 初始化缩放器字典（每个特征一个缩放器）
    scalers = {}
    
    # 遍历每个信号
    for idx in valid_signals.index:
        # 提取时间序列片段
        segment = extract_time_series_segment(
            df=df,
            index=idx,
            segment_length=segment_length,
            feature_columns=feature_columns
        )
        
        # 标准化片段（如果需要）
        if normalize:
            normalized_segment = np.zeros_like(segment)
            
            # 对每个特征单独标准化
            for i, feature in enumerate(feature_columns):
                feature_segment = segment[i:i+1, :]
                
                # 第一次遇到这个特征时创建缩放器
                if feature not in scalers:
                    normalized_feature, scaler = normalize_segment(
                        segment=feature_segment,
                        method=normalize_method,
                        fit=True
                    )
                    scalers[feature] = scaler
                else:
                    # 使用已有的缩放器
                    normalized_feature, _ = normalize_segment(
                        segment=feature_segment,
                        method=normalize_method,
                        scaler=scalers[feature],
                        fit=False
                    )
                
                normalized_segment[i:i+1, :] = normalized_feature
            
            segment = normalized_segment
        
        # 添加到特征列表
        X_list.append(segment)
        
        # 添加到标签列表
        y_list.append(valid_signals.loc[idx, "success"])
    
    # 转换为numpy数组
    X = np.array(X_list)
    y = np.array(y_list)
    
    # 转置以适应aeon库的要求
    if transpose_for_aeon:
        # aeon期望的形状是 (n_samples, n_timepoints, n_features)
        X = np.transpose(X, (0, 2, 1))
    
    return X, y


def create_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: Optional[int] = None,
    shuffle: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    创建训练集、验证集和测试集

    参数:
        X: 特征数组
        y: 标签数组
        train_size: 训练集比例
        val_size: 验证集比例
        test_size: 测试集比例
        random_state: 随机种子
        shuffle: 是否打乱数据

    返回:
        X_train: 训练集特征
        y_train: 训练集标签
        X_val: 验证集特征
        y_val: 验证集标签
        X_test: 测试集特征
        y_test: 测试集标签
    """
    # 检查比例之和是否为1
    if abs(train_size + val_size + test_size - 1.0) > 1e-10:
        raise ValueError("train_size、val_size和test_size之和必须为1")
    
    # 样本数量
    n_samples = len(X)
    
    # 计算分割点
    train_end = int(n_samples * train_size)
    val_end = train_end + int(n_samples * val_size)
    
    if shuffle and random_state is not None:
        # 设置随机种子
        np.random.seed(random_state)
        
        # 生成随机索引
        indices = np.random.permutation(n_samples)
        
        # 打乱数据
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # 划分数据集
        X_train = X_shuffled[:train_end]
        y_train = y_shuffled[:train_end]
        X_val = X_shuffled[train_end:val_end]
        y_val = y_shuffled[train_end:val_end]
        X_test = X_shuffled[val_end:]
        y_test = y_shuffled[val_end:]
    else:
        # 按顺序划分数据集
        X_train = X[:train_end]
        y_train = y[:train_end]
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]
        X_test = X[val_end:]
        y_test = y[val_end:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test
