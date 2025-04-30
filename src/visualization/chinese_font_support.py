"""
为matplotlib添加中文字体支持的模块
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import platform

def set_chinese_font():
    """
    设置matplotlib使用中文字体
    """
    system = platform.system()
    
    if system == 'Windows':
        # Windows系统常见中文字体
        font_list = [
            'Microsoft YaHei',  # 微软雅黑
            'SimHei',           # 黑体
            'SimSun',           # 宋体
            'KaiTi',            # 楷体
            'FangSong',         # 仿宋
            'Arial Unicode MS'  # Arial Unicode
        ]
    elif system == 'Darwin':  # macOS
        font_list = [
            'PingFang SC',      # 苹方
            'Heiti SC',         # 黑体-简
            'STHeiti',          # 华文黑体
            'STSong',           # 华文宋体
            'STKaiti',          # 华文楷体
            'Arial Unicode MS'  # Arial Unicode
        ]
    else:  # Linux
        font_list = [
            'WenQuanYi Micro Hei',  # 文泉驿微米黑
            'WenQuanYi Zen Hei',    # 文泉驿正黑
            'Noto Sans CJK SC',     # Noto Sans CJK SC
            'Noto Sans CJK TC',     # Noto Sans CJK TC
            'Droid Sans Fallback',  # Droid Sans Fallback
            'AR PL UMing CN',       # AR PL UMing CN
            'AR PL UKai CN',        # AR PL UKai CN
            'Arial Unicode MS'      # Arial Unicode
        ]
    
    # 尝试设置字体
    font_found = False
    for font_name in font_list:
        try:
            # 检查字体是否存在
            font_path = fm.findfont(fm.FontProperties(family=font_name))
            if os.path.exists(font_path) and not font_path.endswith('DejaVuSans.ttf'):
                plt.rcParams['font.family'] = font_name
                print(f"成功设置中文字体: {font_name}")
                font_found = True
                break
        except:
            continue
    
    if not font_found:
        print("警告: 未找到支持中文的字体，将尝试使用matplotlib内置的中文支持")
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
            plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
            print("已设置默认中文字体配置")
        except:
            print("错误: 无法设置中文字体，图表中的中文可能无法正确显示")

def set_chinese_font_with_custom_path(font_path):
    """
    使用自定义字体文件路径设置中文字体
    
    参数:
        font_path: 字体文件的路径
    """
    if not os.path.exists(font_path):
        print(f"错误: 字体文件不存在: {font_path}")
        return False
    
    try:
        # 添加字体文件
        font_prop = fm.FontProperties(fname=font_path)
        # 获取字体名称
        font_name = font_prop.get_name()
        # 设置为默认字体
        plt.rcParams['font.family'] = font_name
        print(f"成功设置自定义中文字体: {font_name}")
        return True
    except Exception as e:
        print(f"错误: 设置自定义字体失败: {e}")
        return False

# 默认在导入时设置中文字体
set_chinese_font()
