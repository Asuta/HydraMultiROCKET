"""
评估指标模块
"""
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算评估指标

    参数:
        y_true: 真实标签
        y_pred: 预测标签

    返回:
        包含各种指标的字典
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
    }

    # 如果是二分类问题，添加二分类指标
    if len(np.unique(y_true)) == 2:
        # 获取标签值
        labels = np.unique(y_true)
        pos_label = labels[1]  # 使用第二个标签作为正类
        metrics.update({
            'precision': precision_score(y_true, y_pred, pos_label=pos_label),
            'recall': recall_score(y_true, y_pred, pos_label=pos_label),
            'f1': f1_score(y_true, y_pred, pos_label=pos_label),
        })

    return metrics


def print_metrics(metrics: Dict[str, float]) -> None:
    """
    打印评估指标

    参数:
        metrics: 包含各种指标的字典
    """
    print("\n===== 评估指标 =====")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    打印分类报告

    参数:
        y_true: 真实标签
        y_pred: 预测标签
    """
    print("\n===== 分类报告 =====")
    print(classification_report(y_true, y_pred))


def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    打印混淆矩阵

    参数:
        y_true: 真实标签
        y_pred: 预测标签
    """
    cm = confusion_matrix(y_true, y_pred)
    print("\n===== 混淆矩阵 =====")
    print(cm)
