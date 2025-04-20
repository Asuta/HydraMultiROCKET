"""
可视化模块
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict, Any
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_time_series(X: np.ndarray, y: Optional[np.ndarray] = None, 
                    class_names: Optional[List[str]] = None,
                    title: str = "时间序列数据",
                    save_path: Optional[str] = None) -> None:
    """
    绘制时间序列数据
    
    参数:
        X: 时间序列数据
        y: 标签
        class_names: 类别名称
        title: 图表标题
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 6))
    
    # 如果是多变量时间序列
    if len(X.shape) == 3:
        n_samples, n_channels, n_timepoints = X.shape
        
        # 选择前几个样本进行可视化
        n_samples_to_plot = min(5, n_samples)
        
        for i in range(n_samples_to_plot):
            plt.subplot(n_samples_to_plot, 1, i+1)
            
            for j in range(min(3, n_channels)):  # 最多显示3个通道
                plt.plot(X[i, j, :], label=f'通道 {j+1}')
            
            if y is not None:
                class_label = y[i]
                if class_names is not None and class_label < len(class_names):
                    class_label = class_names[class_label]
                plt.title(f'样本 {i+1}, 类别: {class_label}')
            else:
                plt.title(f'样本 {i+1}')
            
            plt.legend()
    
    # 如果是单变量时间序列
    else:
        n_samples, n_timepoints = X.shape
        
        # 选择前几个样本进行可视化
        n_samples_to_plot = min(5, n_samples)
        
        for i in range(n_samples_to_plot):
            plt.subplot(n_samples_to_plot, 1, i+1)
            plt.plot(X[i, :])
            
            if y is not None:
                class_label = y[i]
                if class_names is not None and class_label < len(class_names):
                    class_label = class_names[class_label]
                plt.title(f'样本 {i+1}, 类别: {class_label}')
            else:
                plt.title(f'样本 {i+1}')
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.9)
    
    if save_path:
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"图表已保存到 {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         class_names: Optional[List[str]] = None,
                         title: str = "混淆矩阵",
                         save_path: Optional[str] = None) -> None:
    """
    绘制混淆矩阵
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称
        title: 图表标题
        save_path: 保存路径
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # 添加文本注释
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    
    if save_path:
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"图表已保存到 {save_path}")
    
    plt.show()


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray,
                  class_names: Optional[List[str]] = None,
                  title: str = "ROC曲线",
                  save_path: Optional[str] = None) -> None:
    """
    绘制ROC曲线
    
    参数:
        y_true: 真实标签
        y_score: 预测概率
        class_names: 类别名称
        title: 图表标题
        save_path: 保存路径
    """
    # 确保y_true是一维数组
    y_true = np.array(y_true).ravel()
    
    # 获取类别数
    n_classes = y_score.shape[1]
    
    # 如果没有提供类别名称，使用默认名称
    if class_names is None:
        class_names = [f'类别 {i}' for i in range(n_classes)]
    
    # 计算每个类别的ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        # 对于多分类问题，使用one-vs-rest方法
        y_true_binary = (y_true == i).astype(int)
        fpr[i], tpr[i], _ = roc_curve(y_true_binary, y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 绘制所有ROC曲线
    plt.figure(figsize=(10, 8))
    
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正例率')
    plt.ylabel('真正例率')
    plt.title(title)
    plt.legend(loc="lower right")
    
    if save_path:
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"图表已保存到 {save_path}")
    
    plt.show()
