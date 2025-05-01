"""
修复matplotlib中文显示问题
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import platform
import sys

def setup_chinese_font():
    """
    设置matplotlib支持中文显示
    """
    system = platform.system()
    
    # 检测操作系统类型并设置合适的中文字体
    if system == 'Windows':
        # Windows系统常见中文字体
        font_list = [
            'Microsoft YaHei',  # 微软雅黑
            'SimHei',           # 黑体
            'SimSun',           # 宋体
            'KaiTi',            # 楷体
            'FangSong',         # 仿宋
            'Arial Unicode MS'  # 通用字体
        ]
    elif system == 'Darwin':  # macOS
        font_list = [
            'PingFang SC',      # 苹方
            'Heiti SC',         # 黑体-简
            'STHeiti',          # 华文黑体
            'STSong',           # 华文宋体
            'STKaiti',          # 华文楷体
            'Arial Unicode MS'  # 通用字体
        ]
    else:  # Linux或其他系统
        font_list = [
            'WenQuanYi Micro Hei',  # 文泉驿微米黑
            'WenQuanYi Zen Hei',    # 文泉驿正黑
            'Droid Sans Fallback',  # 安卓后备字体
            'Noto Sans CJK SC',     # 谷歌Noto字体
            'Noto Sans SC',         # 谷歌Noto字体简体中文版
            'DejaVu Sans'           # 通用字体
        ]
    
    # 查找可用的中文字体
    chinese_font = None
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font in font_list:
        if font in available_fonts:
            chinese_font = font
            break
    
    if chinese_font is None:
        print("警告: 未找到支持中文的字体，尝试安装中文字体...")
        # 如果没有找到中文字体，尝试使用自带的中文字体
        try:
            # 检查是否有自带的中文字体文件
            font_path = os.path.join(os.path.dirname(__file__), 'fonts', 'SimHei.ttf')
            if not os.path.exists(font_path):
                # 如果没有，创建fonts目录
                os.makedirs(os.path.dirname(font_path), exist_ok=True)
                print(f"请将中文字体文件(如SimHei.ttf)放置在 {font_path} 路径下")
                return False
            
            # 添加字体文件
            font_prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
            print(f"已使用自带字体: {font_path}")
            return True
        except Exception as e:
            print(f"设置自带字体失败: {e}")
            return False
    else:
        # 设置找到的中文字体
        plt.rcParams['font.family'] = chinese_font
        print(f"已设置中文字体: {chinese_font}")
        return True

def fix_chinese_display():
    """
    修复matplotlib中文显示问题
    """
    # 设置中文字体
    success = setup_chinese_font()
    
    if not success:
        # 如果设置字体失败，使用另一种方法
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        print("已使用备选方案设置中文字体")
    
    return success
