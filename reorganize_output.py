"""
重新组织output文件夹结构

此脚本用于将output文件夹中的文件重新组织为更清晰的结构：
1. 模型文件 -> output/models/
2. 准备数据文件 -> output/data/
3. 预测结果文件 -> output/predictions/
"""

import os
import shutil
from pathlib import Path


def create_directory(path):
    """创建目录（如果不存在）"""
    os.makedirs(path, exist_ok=True)
    print(f"已创建目录: {path}")


def move_directory(src, dst):
    """移动目录"""
    if not os.path.exists(src):
        print(f"源目录不存在: {src}")
        return
    
    # 如果目标目录已存在，先删除
    if os.path.exists(dst):
        shutil.rmtree(dst)
    
    # 创建父目录（如果不存在）
    parent_dir = os.path.dirname(dst)
    os.makedirs(parent_dir, exist_ok=True)
    
    # 移动目录
    shutil.move(src, dst)
    print(f"已移动目录: {src} -> {dst}")


def main():
    """主函数"""
    # 当前工作目录
    cwd = os.getcwd()
    output_dir = os.path.join(cwd, "output")
    
    # 如果output目录不存在，则退出
    if not os.path.exists(output_dir):
        print(f"output目录不存在: {output_dir}")
        return
    
    # 创建新的目录结构
    create_directory(os.path.join(output_dir, "models"))
    create_directory(os.path.join(output_dir, "data"))
    create_directory(os.path.join(output_dir, "predictions"))
    
    # 移动prepared_data目录
    if os.path.exists(os.path.join(output_dir, "prepared_data")):
        move_directory(
            os.path.join(output_dir, "prepared_data"),
            os.path.join(output_dir, "data", "prepared_data")
        )
    
    # 移动prediction_data_2024目录
    if os.path.exists(os.path.join(output_dir, "prediction_data_2024")):
        move_directory(
            os.path.join(output_dir, "prediction_data_2024"),
            os.path.join(output_dir, "data", "prediction_data_2024")
        )
    
    # 移动predictions_2024目录
    if os.path.exists(os.path.join(output_dir, "predictions_2024")):
        move_directory(
            os.path.join(output_dir, "predictions_2024"),
            os.path.join(output_dir, "predictions", "predictions_2024")
        )
    
    print("文件结构重组完成！")


if __name__ == "__main__":
    main()
