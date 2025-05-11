"""
查看.npz文件内容的工具脚本
"""
import numpy as np
import argparse
import matplotlib.pyplot as plt
import sys
import os
from typing import Tuple, Optional
import json

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入中文字体修复模块
from src.visualization.fix_chinese_font import fix_chinese_display

# 修复中文显示问题
fix_chinese_display()

def load_npz_file(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载.npz文件并返回其中的数组

    参数:
        file_path: .npz文件路径

    返回:
        X: 特征数组
        y: 标签数组
    """
    data = np.load(file_path)
    X = data['X']
    y = data['y']
    return X, y

def print_npz_info(file_path: str, display_all_x_data: bool = False) -> None:
    """
    打印.npz文件的基本信息 (主要用于CLI模式)

    参数:
        file_path: .npz文件路径
        display_all_x_data: 是否显示X的所有样本数据 (默认为False，只显示少量)
    """
    print(f"\n查看文件: {file_path}")
    X, y = load_npz_file(file_path)

    print(f"数据形状:")
    print(f"  X 形状: {X.shape}")
    print(f"  y 形状: {y.shape}")

    print(f"\n数据类型:")
    print(f"  X 类型: {X.dtype}")
    print(f"  y 类型: {y.dtype}")

    print(f"\n标签分布:")
    unique_labels, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"  标签 {label}: {count} 个样本 ({count/len(y)*100:.2f}%)")

    print(f"\nX 数据统计:")
    if X.size > 0:
        print(f"  最小值: {X.min()}")
        print(f"  最大值: {X.max()}")
        print(f"  均值: {X.mean()}")
        print(f"  标准差: {X.std()}")
    else:
        print("  X 数据为空, 无法计算统计信息。")
        
    # 打印X的数据示例
    num_samples_to_show_cli = X.shape[0] if display_all_x_data else min(2, X.shape[0])
    print(f"\nX 数据示例 (前{num_samples_to_show_cli}个样本的特征数据):")
    if X.shape[0] == 0:
        print("  X 中没有样本数据。")
    else:
        for i in range(num_samples_to_show_cli):
            print(f"  样本 {i+1}:")
            if X.ndim == 1: # (samples,)
                print(f"    值: {X[i]}")
                continue
            
            if X.shape[1] == 0: # (samples, 0, ...)
                print(f"    样本 {i+1} 中没有特征。")
            else:
                for j in range(X.shape[1]): # Iterate over features
                    feature_label = f"    特征 {j+1}"
                    if X.ndim == 2: # (samples, features)
                        print(f"{feature_label}: {X[i, j]}")
                    elif X.ndim >= 3: # (samples, features, timepoints, ...)
                        feature_values = X[i, j, :] # Assume 3rd dim is time series
                        max_timepoints_to_show_cli = 15 # CLI截断
                        if len(feature_values) > max_timepoints_to_show_cli:
                            print(f"{feature_label} (前{max_timepoints_to_show_cli}个时间点): {feature_values[:max_timepoints_to_show_cli]} ... (共{len(feature_values)}个时间点)")
                        else:
                            print(f"{feature_label}: {feature_values}")
                    else: # Should not happen based on typical NPZ structure
                        print(f"{feature_label}: (数据维度无法直接显示)")
        
        if not display_all_x_data and X.shape[0] > num_samples_to_show_cli:
            print(f"  ... (共 {X.shape[0]} 个样本, 使用 --display-all-x 参数在CLI中显示更多样本的摘要)")
        elif display_all_x_data and X.shape[0] > num_samples_to_show_cli : # Should be equal if display_all_x_data is true
             print(f"  ... (显示了全部 {X.shape[0]} 个样本的摘要)")

def visualize_sample(file_path: str, sample_idx: int = 0, feature_indices: Optional[list] = None, output_dir: str = '.') -> None:
    """
    可视化一个样本的时间序列数据

    参数:
        file_path: .npz文件路径
        sample_idx: 要可视化的样本索引
        feature_indices: 要可视化的特征索引列表，如果为None则可视化所有特征
        output_dir: 图像输出目录
    """
    X, y = load_npz_file(file_path)

    if sample_idx >= X.shape[0]:
        print(f"错误: 样本索引 {sample_idx} 超出范围 (0-{X.shape[0]-1})")
        return

    # 获取样本
    sample = X[sample_idx]
    label = y[sample_idx]

    # 确定要可视化的特征
    if feature_indices is None:
        feature_indices = list(range(sample.shape[0]))

    # 创建图形
    plt.figure(figsize=(12, 8))

    # 绘制每个特征的时间序列
    for i, feature_idx in enumerate(feature_indices):
        if feature_idx >= sample.shape[0]:
            print(f"警告: 特征索引 {feature_idx} 超出范围 (0-{sample.shape[0]-1})")
            continue

        plt.subplot(len(feature_indices), 1, i+1)
        plt.plot(sample[feature_idx])
        plt.title(f"特征 {feature_idx+1}")

        if i == 0:
            plt.suptitle(f"样本 {sample_idx+1} (标签: {label})")

    plt.tight_layout()

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 保存图像
    output_path = os.path.join(output_dir, f"sample_{sample_idx}_visualization.png")
    plt.savefig(output_path)
    print(f"已保存可视化图像到: {output_path}")
    plt.close()

def visualize_feature_distribution(file_path: str, feature_idx: int = 0, output_dir: str = '.') -> None:
    """
    可视化一个特征在所有样本中的分布

    参数:
        file_path: .npz文件路径
        feature_idx: 要可视化的特征索引
        output_dir: 图像输出目录
    """
    X, y = load_npz_file(file_path)

    if feature_idx >= X.shape[1]:
        print(f"错误: 特征索引 {feature_idx} 超出范围 (0-{X.shape[1]-1})")
        return

    # 提取特定特征的所有样本数据
    feature_data = X[:, feature_idx, :]

    # 计算每个时间点的统计信息
    mean_values = np.mean(feature_data, axis=0)
    std_values = np.std(feature_data, axis=0)

    # 创建图形
    plt.figure(figsize=(12, 6))

    # 绘制均值线
    plt.plot(mean_values, 'b-', label='均值')

    # 绘制标准差区间
    plt.fill_between(
        range(len(mean_values)),
        mean_values - std_values,
        mean_values + std_values,
        alpha=0.2,
        color='b',
        label='±1 标准差'
    )

    plt.title(f"特征 {feature_idx+1} 在所有样本中的分布")
    plt.xlabel('时间点')
    plt.ylabel('值')
    plt.legend()

    plt.tight_layout()

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 保存图像
    output_path = os.path.join(output_dir, f"feature_{feature_idx}_distribution.png")
    plt.savefig(output_path)
    print(f"已保存特征分布图像到: {output_path}")
    plt.close()

def visualize_class_distribution(file_path: str, feature_idx: int = 0, output_dir: str = '.') -> None:
    """
    可视化不同类别的特征分布

    参数:
        file_path: .npz文件路径
        feature_idx: 要可视化的特征索引
        output_dir: 图像输出目录
    """
    X, y = load_npz_file(file_path)

    if feature_idx >= X.shape[1]:
        print(f"错误: 特征索引 {feature_idx} 超出范围 (0-{X.shape[1]-1})")
        return

    # 获取唯一的类别
    unique_classes = np.unique(y)

    # 创建图形
    plt.figure(figsize=(12, 6))

    # 为每个类别绘制均值线
    for class_label in unique_classes:
        # 获取该类别的样本索引
        class_indices = np.where(y == class_label)[0]

        # 提取该类别的特征数据
        class_data = X[class_indices, feature_idx, :]

        # 计算均值
        class_mean = np.mean(class_data, axis=0)

        # 绘制均值线
        plt.plot(class_mean, label=f'类别 {int(class_label)}')

    plt.title(f"特征 {feature_idx+1} 在不同类别中的分布")
    plt.xlabel('时间点')
    plt.ylabel('均值')
    plt.legend()

    plt.tight_layout()

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 保存图像
    output_path = os.path.join(output_dir, f"feature_{feature_idx}_class_distribution.png")
    plt.savefig(output_path)
    print(f"已保存类别分布图像到: {output_path}")
    plt.close()

def parse_args_for_script():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='查看.npz文件内容。\n默认启动UI模式。使用 --no-ui 进入命令行模式。\n如果提供了文件路径且输出重定向，也会尝试进入CLI模式。',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('file_path', type=str, nargs='?', default=None,
                        help='.npz文件路径 (UI模式下可选, CLI模式下必需)')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='(CLI模式) 是否可视化样本')
    parser.add_argument('--sample', '-s', type=int, default=0,
                        help='(CLI模式) 要可视化的样本索引')
    parser.add_argument('--features', '-f', type=str,
                        help='(CLI模式) 要可视化的特征索引，用逗号分隔 (例如: 0,1,2)')
    parser.add_argument('--feature-dist', '-fd', type=int, default=None,
                        help='(CLI模式) 要可视化分布的特征索引')
    parser.add_argument('--class-dist', '-cd', type=int, default=None,
                        help='(CLI模式) 要可视化类别分布的特征索引')
    parser.add_argument('--output-dir', '-o', type=str, default='.',
                        help='(CLI/UI) 图像输出目录. 默认: 当前目录')
    parser.add_argument('--no-ui', action='store_true',
                        help='禁用UI，强制使用命令行模式')
    parser.add_argument('--display-all-x', action='store_true',
                        help='(CLI模式) 在命令行模式下显示所有X数据样本的摘要')
    return parser.parse_args()

# --- UI相关的导入和代码将从这里开始 ---
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

# 全局变量存储加载的数据，以便UI各部分访问
loaded_X_ui: Optional[np.ndarray] = None
loaded_y_ui: Optional[np.ndarray] = None
current_file_path_ui: Optional[str] = None
root_tk_window: Optional[tk.Tk] = None


def get_npz_info_for_ui(file_path: str) -> dict:
    """
    加载.npz文件并返回用于UI显示的信息字典
    """
    global loaded_X_ui, loaded_y_ui, current_file_path_ui
    data = np.load(file_path)
    # Make sure X and y exist, otherwise create empty arrays with appropriate types
    loaded_X_ui = data.get('X') 
    loaded_y_ui = data.get('y')

    if loaded_X_ui is None:
        loaded_X_ui = np.array([])
        print("警告: .npz 文件中未找到 'X' 数据。")
    if loaded_y_ui is None:
        loaded_y_ui = np.array([])
        print("警告: .npz 文件中未找到 'y' 数据。")
        
    current_file_path_ui = file_path

    info = {}
    info['file_path'] = file_path
    info['X_shape'] = loaded_X_ui.shape
    info['y_shape'] = loaded_y_ui.shape
    info['X_dtype'] = loaded_X_ui.dtype
    info['y_dtype'] = loaded_y_ui.dtype

    if loaded_y_ui.size > 0:
        unique_labels, counts = np.unique(loaded_y_ui, return_counts=True)
        label_dist_str_list = []
        for label, count in zip(unique_labels, counts):
            label_dist_str_list.append(f"  标签 {label}: {count} 个样本 ({count/len(loaded_y_ui)*100:.2f}%)")
        info['label_distribution'] = "\n".join(label_dist_str_list)
    else:
        info['label_distribution'] = "  y 数据为空或不存在。"


    if loaded_X_ui.size > 0:
        info['X_min'] = loaded_X_ui.min()
        info['X_max'] = loaded_X_ui.max()
        info['X_mean'] = loaded_X_ui.mean()
        info['X_std'] = loaded_X_ui.std()
    else:
        info['X_min'] = "N/A (X为空)"
        info['X_max'] = "N/A (X为空)"
        info['X_mean'] = "N/A (X为空)"
        info['X_std'] = "N/A (X为空)"
    return info

def display_full_x_data_ui():
    """在新的窗口中显示完整的X数据 (会进行截断以优化性能)"""
    global root_tk_window
    if loaded_X_ui is None or current_file_path_ui is None:
        messagebox.showinfo("信息", "请先加载一个.npz文件。")
        return
    if loaded_X_ui.size == 0:
        messagebox.showinfo("信息", "X 数据为空。")
        return

    top = tk.Toplevel(root_tk_window)
    top.title(f"X 数据详情 - {os.path.basename(current_file_path_ui)}")
    top.geometry("800x600")

    text_area = scrolledtext.ScrolledText(top, wrap=tk.NONE) # tk.NONE for horizontal scroll
    text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
    
    data_str_lines = []
    data_str_lines.append(f"X 数据 (形状: {loaded_X_ui.shape}):\\n")

    MAX_SAMPLES_DISPLAY = 100       # UI中最多直接显示的样本数
    MAX_FEATURES_DISPLAY = 20       # UI中每个样本最多直接显示的特征数
    MAX_TIMEPOINTS_DISPLAY = 50     # UI中每个特征最多直接显示的时间点数
    
    actual_num_samples = loaded_X_ui.shape[0]
    samples_to_show = min(actual_num_samples, MAX_SAMPLES_DISPLAY)

    truncated_due_to_performance = False

    for i in range(samples_to_show):
        data_str_lines.append(f"  样本 {i+1}:")
        
        if loaded_X_ui.ndim == 1: # Shape: (n_samples,)
            data_str_lines.append(f"    值: {loaded_X_ui[i]}")
            continue

        actual_num_features = loaded_X_ui.shape[1] if loaded_X_ui.ndim > 1 else 0
        features_to_show = min(actual_num_features, MAX_FEATURES_DISPLAY)

        if actual_num_features == 0:
            data_str_lines.append(f"    此样本无特征数据。")
            continue

        for j in range(features_to_show):
            feature_label = f"    特征 {j+1}"
            if loaded_X_ui.ndim == 2: # Shape: (n_samples, n_features)
                data_str_lines.append(f"{feature_label}: {loaded_X_ui[i, j]}")
            elif loaded_X_ui.ndim >= 3: # Shape: (n_samples, n_features, n_timepoints, ...)
                # Assuming the third dimension is time series
                time_series_data = loaded_X_ui[i, j, :] 
                actual_num_timepoints = len(time_series_data)
                timepoints_to_show = min(actual_num_timepoints, MAX_TIMEPOINTS_DISPLAY)
                
                displayed_time_series = time_series_data[:timepoints_to_show]
                
                if actual_num_timepoints > timepoints_to_show:
                    data_str_lines.append(f"{feature_label}: {displayed_time_series} ... (共{actual_num_timepoints}个时间点)")
                    truncated_due_to_performance = True
                else:
                    data_str_lines.append(f"{feature_label}: {displayed_time_series}")
            else: # Should not happen for typical data
                 data_str_lines.append(f"{feature_label}: (数据维度异常)")
        
        if actual_num_features > features_to_show:
            data_str_lines.append(f"    ... (还有 {actual_num_features - features_to_show} 个特征未在此处显示)")
            truncated_due_to_performance = True
            
    if actual_num_samples > samples_to_show:
         data_str_lines.append(f"... (还有 {actual_num_samples - samples_to_show} 个样本未在此处显示)")
         truncated_due_to_performance = True

    text_area.insert(tk.INSERT, "\n".join(data_str_lines))
    text_area.config(state=tk.DISABLED)
    
    if truncated_due_to_performance:
        messagebox.showwarning("数据截断", "为优化性能，部分X数据已在窗口中截断显示。完整数据已加载到内存中。", parent=top)

def export_x_data_to_json():
    """将加载的X数据导出为JSON文件"""
    global loaded_X_ui, current_file_path_ui, root_tk_window

    if loaded_X_ui is None or current_file_path_ui is None:
        messagebox.showerror("错误", "没有加载任何.npz文件，或者X数据不存在。")
        return

    if loaded_X_ui.size == 0:
        messagebox.showinfo("信息", "X 数据为空，将导出一个空的JSON列表。")
        x_data_list = []
    else:
        try:
            x_data_list = loaded_X_ui.tolist()
        except AttributeError:
            messagebox.showerror("错误", "X数据无法转换为列表。可能数据类型不受支持。")
            return
        except Exception as e:
            messagebox.showerror("转换错误", f"将X数据转换为列表时发生未知错误: {e}")
            return

    default_filename = os.path.splitext(os.path.basename(current_file_path_ui))[0] + "_X_data.json"
    filepath = filedialog.asksaveasfilename(
        title="保存X数据为JSON",
        defaultextension=".json",
        initialfile=default_filename,
        filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
        parent=root_tk_window
    )

    if filepath:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(x_data_list, f, ensure_ascii=False, indent=4)
            messagebox.showinfo("成功", f"X数据已成功导出到:\\n{filepath}")
        except IOError as e:
            messagebox.showerror("保存错误", f"无法写入文件: {e}")
        except Exception as e:
            messagebox.showerror("导出错误", f"导出X数据到JSON时发生未知错误: {e}")

def main_ui(cli_args: argparse.Namespace):
    global loaded_X_ui, loaded_y_ui, current_file_path_ui, root_tk_window
    root_tk_window = tk.Tk()
    root_tk_window.title("NPZ 数据查看器")
    root_tk_window.geometry("750x800")

    # --- 文件选择 ---
    file_frame = ttk.LabelFrame(root_tk_window, text="文件操作")
    file_frame.pack(padx=10, pady=10, fill="x")

    file_path_display_var = tk.StringVar(value="未选择文件")
    file_path_label = ttk.Label(file_frame, textvariable=file_path_display_var, wraplength=500) # wraplength for long paths
    file_path_label.pack(side=tk.LEFT, padx=5, pady=5, fill="x", expand=True)
    
    # --- 右侧信息显示区 (ScrolledText) ---
    info_display_frame = ttk.LabelFrame(root_tk_window, text="文件信息")
    info_display_frame.pack(padx=10, pady=5, fill="both", expand=True)
    
    info_text_area = scrolledtext.ScrolledText(info_display_frame, wrap=tk.WORD, height=15, state=tk.DISABLED)
    info_text_area.pack(padx=5, pady=5, fill="both", expand=True)

    show_x_data_button = ttk.Button(info_display_frame, text="查看详细X数据", command=display_full_x_data_ui, state=tk.DISABLED)
    show_x_data_button.pack(pady=(5,0)) # Adjusted padding

    export_x_json_button = ttk.Button(info_display_frame, text="导出X数据 (JSON)", command=export_x_data_to_json, state=tk.DISABLED)
    export_x_json_button.pack(pady=(2,5)) # Add some padding

    # --- 可视化控制 ---
    vis_control_frame = ttk.LabelFrame(root_tk_window, text="可视化操作")
    vis_control_frame.pack(padx=10, pady=10, fill="x")
    
    # Output directory
    ttk.Label(vis_control_frame, text="输出目录:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    output_dir_var = tk.StringVar(value=cli_args.output_dir if cli_args.output_dir else ".")
    output_dir_entry = ttk.Entry(vis_control_frame, textvariable=output_dir_var, width=40)
    output_dir_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    def browse_output_dir():
        dir_path = filedialog.askdirectory(title="选择图像输出目录", initialdir=output_dir_var.get())
        if dir_path:
            output_dir_var.set(dir_path)
    ttk.Button(vis_control_frame, text="浏览...", command=browse_output_dir).grid(row=0, column=2, padx=5, pady=5)

    # Sample visualization
    ttk.Label(vis_control_frame, text="样本可视化:").grid(row=1, column=0, columnspan=3, sticky="w", padx=5, pady=(10,0))
    ttk.Label(vis_control_frame, text="样本索引:").grid(row=2, column=0, padx=5, pady=2, sticky="w")
    sample_idx_var = tk.StringVar(value="0")
    sample_idx_entry = ttk.Entry(vis_control_frame, textvariable=sample_idx_var, width=10)
    sample_idx_entry.grid(row=2, column=1, padx=5, pady=2, sticky="w")
    
    ttk.Label(vis_control_frame, text="特征索引 (逗号分隔, 可空):").grid(row=3, column=0, padx=5, pady=2, sticky="w")
    features_indices_var = tk.StringVar()
    features_indices_entry = ttk.Entry(vis_control_frame, textvariable=features_indices_var, width=30)
    features_indices_entry.grid(row=3, column=1, padx=5, pady=2, sticky="ew")

    def trigger_visualize_sample():
        if not current_file_path_ui:
            messagebox.showerror("错误", "请先加载.npz文件。")
            return
        try:
            s_idx = int(sample_idx_var.get())
            f_indices_str = features_indices_var.get()
            f_indices = [int(idx.strip()) for idx in f_indices_str.split(',') if idx.strip()] if f_indices_str else None
            out_dir = output_dir_var.get()
            if not os.path.isdir(out_dir):
                 os.makedirs(out_dir, exist_ok=True)
            visualize_sample(current_file_path_ui, s_idx, f_indices, out_dir) # Uses original visualize_sample
            messagebox.showinfo("成功", f"样本可视化图像已保存到: {os.path.join(out_dir, f'sample_{s_idx}_visualization.png')}")
        except ValueError:
            messagebox.showerror("输入错误", "样本索引或特征索引必须是有效的整数。")
        except Exception as e:
            messagebox.showerror("可视化错误", f"执行样本可视化时出错: {e}")
            
    vis_sample_button = ttk.Button(vis_control_frame, text="可视化样本", command=trigger_visualize_sample, state=tk.DISABLED)
    vis_sample_button.grid(row=3, column=2, padx=5, pady=2)

    # Feature distribution
    ttk.Label(vis_control_frame, text="特征分布可视化:").grid(row=4, column=0, columnspan=3, sticky="w", padx=5, pady=(10,0))
    ttk.Label(vis_control_frame, text="特征索引:").grid(row=5, column=0, padx=5, pady=2, sticky="w")
    feature_dist_idx_var = tk.StringVar(value="0")
    feature_dist_idx_entry = ttk.Entry(vis_control_frame, textvariable=feature_dist_idx_var, width=10)
    feature_dist_idx_entry.grid(row=5, column=1, padx=5, pady=2, sticky="w")
    
    def trigger_visualize_feature_distribution():
        if not current_file_path_ui:
            messagebox.showerror("错误", "请先加载.npz文件。")
            return
        try:
            f_idx = int(feature_dist_idx_var.get())
            out_dir = output_dir_var.get()
            if not os.path.isdir(out_dir):
                 os.makedirs(out_dir, exist_ok=True)
            visualize_feature_distribution(current_file_path_ui, f_idx, out_dir)
            messagebox.showinfo("成功", f"特征分布图像已保存到: {os.path.join(out_dir, f'feature_{f_idx}_distribution.png')}")
        except ValueError:
            messagebox.showerror("输入错误", "特征索引必须是有效的整数。")
        except Exception as e:
            messagebox.showerror("可视化错误", f"执行特征分布可视化时出错: {e}")

    vis_feature_dist_button = ttk.Button(vis_control_frame, text="可视化特征分布", command=trigger_visualize_feature_distribution, state=tk.DISABLED)
    vis_feature_dist_button.grid(row=5, column=2, padx=5, pady=2)

    # Class distribution
    ttk.Label(vis_control_frame, text="类别分布可视化:").grid(row=6, column=0, columnspan=3, sticky="w", padx=5, pady=(10,0))
    ttk.Label(vis_control_frame, text="特征索引:").grid(row=7, column=0, padx=5, pady=2, sticky="w")
    class_dist_idx_var = tk.StringVar(value="0")
    class_dist_idx_entry = ttk.Entry(vis_control_frame, textvariable=class_dist_idx_var, width=10)
    class_dist_idx_entry.grid(row=7, column=1, padx=5, pady=2, sticky="w")

    def trigger_visualize_class_distribution():
        if not current_file_path_ui:
            messagebox.showerror("错误", "请先加载.npz文件。")
            return
        try:
            f_idx = int(class_dist_idx_var.get())
            out_dir = output_dir_var.get()
            if not os.path.isdir(out_dir):
                 os.makedirs(out_dir, exist_ok=True)
            visualize_class_distribution(current_file_path_ui, f_idx, out_dir)
            messagebox.showinfo("成功", f"类别特征分布图像已保存到: {os.path.join(out_dir, f'feature_{f_idx}_class_distribution.png')}")
        except ValueError:
            messagebox.showerror("输入错误", "特征索引必须是有效的整数。")
        except Exception as e:
            messagebox.showerror("可视化错误", f"执行类别分布可视化时出错: {e}")
            
    vis_class_dist_button = ttk.Button(vis_control_frame, text="可视化类别分布", command=trigger_visualize_class_distribution, state=tk.DISABLED)
    vis_class_dist_button.grid(row=7, column=2, padx=5, pady=2)
    
    vis_control_frame.columnconfigure(1, weight=1) # Allow middle column with entry fields to expand

    def select_and_load_file(initial_path: Optional[str] = None):
        global loaded_X_ui, loaded_y_ui, current_file_path_ui # Ensure global scope for assignment
        filepath_to_load = initial_path
        if not filepath_to_load:
            filepath_to_load = filedialog.askopenfilename(
                title="选择 .npz 文件",
                filetypes=(("NPZ files", "*.npz"), ("All files", "*.*"))
            )
        
        if filepath_to_load:
            try:
                info = get_npz_info_for_ui(filepath_to_load) # This sets global X, y, path
                file_path_display_var.set(f"当前文件: {filepath_to_load}")
                
                info_text_area.config(state=tk.NORMAL)
                info_text_area.delete('1.0', tk.END)
                info_text_area.insert(tk.END, f"文件路径: {info['file_path']}\\n\\n")
                info_text_area.insert(tk.END, f"数据形状:\\n")
                info_text_area.insert(tk.END, f"  X 形状: {info['X_shape']}\\n")
                info_text_area.insert(tk.END, f"  y 形状: {info['y_shape']}\\n\\n")
                info_text_area.insert(tk.END, f"数据类型:\\n")
                info_text_area.insert(tk.END, f"  X 类型: {info['X_dtype']}\\n")
                info_text_area.insert(tk.END, f"  y 类型: {info['y_dtype']}\\n\\n")
                info_text_area.insert(tk.END, f"标签分布:\\n{info['label_distribution']}\\n\\n")
                info_text_area.insert(tk.END, f"X 数据统计:\\n")
                info_text_area.insert(tk.END, f"  最小值: {info['X_min']}\\n")
                info_text_area.insert(tk.END, f"  最大值: {info['X_max']}\\n")
                info_text_area.insert(tk.END, f"  均值: {info['X_mean']}\\n")
                info_text_area.insert(tk.END, f"  标准差: {info['X_std']}\\n")
                info_text_area.config(state=tk.DISABLED)

                # Enable buttons
                show_x_data_button.config(state=tk.NORMAL)
                vis_sample_button.config(state=tk.NORMAL)
                vis_feature_dist_button.config(state=tk.NORMAL)
                vis_class_dist_button.config(state=tk.NORMAL)
                export_x_json_button.config(state=tk.NORMAL)

            except FileNotFoundError:
                messagebox.showerror("错误", f"文件未找到: {filepath_to_load}")
                file_path_display_var.set(f"文件加载失败: {filepath_to_load}")
                loaded_X_ui, loaded_y_ui, current_file_path_ui = None, None, None
                # Disable buttons
                show_x_data_button.config(state=tk.DISABLED)
                vis_sample_button.config(state=tk.DISABLED)
                vis_feature_dist_button.config(state=tk.DISABLED)
                vis_class_dist_button.config(state=tk.DISABLED)
                export_x_json_button.config(state=tk.DISABLED)
        else: # No file selected or initial_path was None and dialog cancelled
            if not initial_path: # Only show if it wasn't an initial load attempt
                 file_path_display_var.set("未选择文件")


    select_button = ttk.Button(file_frame, text="选择 .npz 文件", command=lambda: select_and_load_file(None))
    select_button.pack(side=tk.RIGHT, padx=5, pady=5) # Move button to right

    # Initial load if file_path provided via CLI args
    if cli_args.file_path:
        def attempt_initial_load():
            if os.path.exists(cli_args.file_path):
                select_and_load_file(initial_path=cli_args.file_path)
            else:
                messagebox.showwarning("文件未找到", f"通过命令行指定的初始文件路径 '{cli_args.file_path}' 未找到或无法访问。请手动选择文件。")
                file_path_display_var.set(f"指定文件未找到: {cli_args.file_path}")
        root_tk_window.after(100, attempt_initial_load) # Use after to ensure main window is ready

    root_tk_window.mainloop()


def main_cli_operations(args: argparse.Namespace):
    """
    处理CLI模式下的操作，如打印信息和可视化。
    假设 args.file_path 是有效的。
    """
    # 首先打印文件基本信息
    print_npz_info(args.file_path, args.display_all_x)

    # 确保输出目录存在 (仅当需要可视化时)
    perform_visualization = args.visualize or args.feature_dist is not None or args.class_dist is not None
    if perform_visualization and args.output_dir:
        try:
            os.makedirs(args.output_dir, exist_ok=True)
        except OSError as e:
            print(f"错误: 创建输出目录 '{args.output_dir}' 失败: {e}")
            return # 不能继续进行需要输出目录的可视化

    # 可视化样本
    if args.visualize:
        feature_indices = None
        if args.features:
            try:
                feature_indices = [int(idx.strip()) for idx in args.features.split(',') if idx.strip()]
            except ValueError:
                print("错误: 特征索引 '--features' 格式错误。应为逗号分隔的整数。")
                return 
        visualize_sample(args.file_path, args.sample, feature_indices, args.output_dir)

    # 可视化特征分布
    if args.feature_dist is not None:
        visualize_feature_distribution(args.file_path, args.feature_dist, args.output_dir)

    # 可视化类别分布
    if args.class_dist is not None:
        visualize_class_distribution(args.file_path, args.class_dist, args.output_dir)


def main():
    """主函数，根据参数选择CLI或UI模式"""
    args = parse_args_for_script()

    is_cli_mode = args.no_ui
    # 如果提供了文件路径且输出不是TTY（例如重定向到文件），并且没有明确请求可视化，则倾向于CLI模式
    if not is_cli_mode and args.file_path and not sys.stdout.isatty():
        if not args.visualize and args.feature_dist is None and args.class_dist is None:
            is_cli_mode = True
            # print("通知: 检测到非交互式输出且指定了文件路径，切换到CLI模式进行信息转储。", file=sys.stderr)

    if is_cli_mode:
        if not args.file_path:
             # 重新创建解析器以打印帮助信息，或者可以传递原始解析器对象
             parser_for_help = argparse.ArgumentParser(description='查看.npz文件内容')
             # Manually add file_path argument for help context
             parser_for_help.add_argument('file_path', type=str, help='.npz文件路径 (CLI模式下必需)')
             parser_for_help.print_help()
             print("\n错误: 在命令行模式下, 必须提供 file_path 参数。")
             sys.exit(1)
        main_cli_operations(args)
    else:
        # 对于UI模式，中文显示修复应该在Tkinter根窗口创建前完成
        fix_chinese_display() 
        main_ui(args) # 将解析的参数传递给UI函数，以便处理初始文件路径等

if __name__ == "__main__":
    main()
