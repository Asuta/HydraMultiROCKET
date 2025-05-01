"""
测试中文字体显示
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from src.visualization.fix_chinese_font import fix_chinese_display

def test_chinese_font():
    """测试中文字体显示"""
    # 修复中文显示问题
    fix_chinese_display()
    
    # 创建示例数据
    categories = ['类别 0', '类别 1']
    values = {
        '类别 0': [0.5, 0.43],
        '类别 1': [0.37, 0.43],
        '类别 2': [0.13, 0.14]
    }
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(categories))  # 类别位置
    width = 0.2  # 柱状图宽度
    
    # 绘制柱状图
    plt.bar(x - width, values['类别 0'], width, label='类别 0的概率')
    plt.bar(x, values['类别 1'], width, label='类别 1的概率')
    plt.bar(x + width, values['类别 2'], width, label='类别 2的概率')
    
    # 添加标签和标题
    plt.xlabel('真实类别')
    plt.ylabel('预测概率')
    plt.title('各类别的预测概率分布')
    plt.xticks(x, categories)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图表
    os.makedirs('output/test', exist_ok=True)
    plt.savefig('output/test/chinese_font_test.png')
    print("测试图表已保存到 output/test/chinese_font_test.png")
    
    # 显示图表
    plt.show()

if __name__ == "__main__":
    test_chinese_font()
