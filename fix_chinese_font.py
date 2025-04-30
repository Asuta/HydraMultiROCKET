"""
修复中文字体显示问题的脚本
"""
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import numpy as np

def list_system_fonts():
    """列出系统中的所有字体"""
    fonts = fm.findSystemFonts()
    print(f"系统中共有 {len(fonts)} 个字体")
    
    # 打印前10个字体
    for i, font in enumerate(fonts[:10]):
        try:
            font_name = fm.FontProperties(fname=font).get_name()
            print(f"{i+1}. {font_name} ({font})")
        except:
            print(f"{i+1}. 无法获取字体名称 ({font})")
    
    return fonts

def find_chinese_fonts():
    """查找支持中文的字体"""
    system = platform.system()
    
    if system == 'Windows':
        # Windows系统常见中文字体
        chinese_font_names = [
            'Microsoft YaHei',  # 微软雅黑
            'SimHei',           # 黑体
            'SimSun',           # 宋体
            'KaiTi',            # 楷体
            'FangSong',         # 仿宋
            'Arial Unicode MS'  # Arial Unicode
        ]
    elif system == 'Darwin':  # macOS
        chinese_font_names = [
            'PingFang SC',      # 苹方
            'Heiti SC',         # 黑体-简
            'STHeiti',          # 华文黑体
            'STSong',           # 华文宋体
            'STKaiti',          # 华文楷体
            'Arial Unicode MS'  # Arial Unicode
        ]
    else:  # Linux
        chinese_font_names = [
            'WenQuanYi Micro Hei',  # 文泉驿微米黑
            'WenQuanYi Zen Hei',    # 文泉驿正黑
            'Noto Sans CJK SC',     # Noto Sans CJK SC
            'Noto Sans CJK TC',     # Noto Sans CJK TC
            'Droid Sans Fallback',  # Droid Sans Fallback
            'AR PL UMing CN',       # AR PL UMing CN
            'AR PL UKai CN',        # AR PL UKai CN
            'Arial Unicode MS'      # Arial Unicode
        ]
    
    # 查找字体文件
    chinese_fonts = []
    for font_name in chinese_font_names:
        try:
            font_path = fm.findfont(fm.FontProperties(family=font_name))
            if os.path.exists(font_path) and not font_path.endswith('DejaVuSans.ttf'):
                chinese_fonts.append((font_name, font_path))
                print(f"找到中文字体: {font_name} ({font_path})")
        except:
            continue
    
    return chinese_fonts

def set_font_directly():
    """直接设置matplotlib的字体配置"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    print("已直接设置字体配置")

def test_font_with_plot(font_name=None):
    """测试字体是否支持中文"""
    plt.figure(figsize=(10, 6))
    
    if font_name:
        plt.rcParams['font.family'] = font_name
        print(f"测试字体: {font_name}")
    
    # 创建一个简单的图表
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x)
    
    plt.plot(x, y)
    plt.title('中文标题测试')
    plt.xlabel('横轴标签')
    plt.ylabel('纵轴标签')
    
    # 添加一些中文文本
    plt.text(np.pi, 0, '这是一段中文文本')
    
    # 保存图表
    output_path = 'output/font_test.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"测试图表已保存到: {output_path}")
    
    return output_path

def fix_probability_distribution_plots():
    """修复概率分布图的中文显示问题"""
    # 设置字体
    set_font_directly()
    
    # 重新生成2023年数据的概率分布图
    try:
        # 加载数据
        data = np.load('output/prepared_data_new/test_dataset.npz')
        X = data['X']
        y = data['y']
        
        # 过滤无效类别
        valid_mask = (y != 2)
        X = X[valid_mask]
        y = y[valid_mask]
        
        # 加载模型
        import pickle
        with open('output/models_new/multirocket_hydra_20250429_222207/model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # 预测
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        
        # 绘制概率分布图
        plt.figure(figsize=(10, 6))
        
        # 获取正类（标签1）的概率
        proba_class_1 = y_proba[:, 1] if y_proba.shape[1] > 1 else 1 - y_proba[:, 0]
        
        # 根据真实标签分组
        proba_class_1_true_0 = proba_class_1[y == 0]
        proba_class_1_true_1 = proba_class_1[y == 1]
        
        # 绘制直方图
        plt.hist(proba_class_1_true_0, bins=20, alpha=0.5, label='真实标签 = 0 (失败)')
        plt.hist(proba_class_1_true_1, bins=20, alpha=0.5, label='真实标签 = 1 (成功)')
        
        plt.title('预测为成功的概率分布')
        plt.xlabel('预测为成功的概率')
        plt.ylabel('样本数量')
        plt.legend()
        plt.grid(True)
        
        # 保存图表
        plt.savefig('output/probability_distribution_new_fixed.png')
        print("2023年数据的概率分布图已修复并保存到 output/probability_distribution_new_fixed.png")
    except Exception as e:
        print(f"修复2023年数据的概率分布图时出错: {e}")
    
    # 重新生成2024年数据的概率分布图
    try:
        # 加载数据
        data = np.load('output/prepared_data_2024/test_dataset.npz')
        X = data['X']
        y = data['y']
        
        # 过滤无效类别
        valid_mask = (y != 2)
        X = X[valid_mask]
        y = y[valid_mask]
        
        # 加载模型
        import pickle
        with open('output/models_new/multirocket_hydra_20250429_222207/model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # 预测
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        
        # 绘制概率分布图
        plt.figure(figsize=(10, 6))
        
        # 获取正类（标签1）的概率
        proba_class_1 = y_proba[:, 1] if y_proba.shape[1] > 1 else 1 - y_proba[:, 0]
        
        # 根据真实标签分组
        proba_class_1_true_0 = proba_class_1[y == 0]
        proba_class_1_true_1 = proba_class_1[y == 1]
        
        # 绘制直方图
        plt.hist(proba_class_1_true_0, bins=20, alpha=0.5, label='真实标签 = 0 (失败)')
        plt.hist(proba_class_1_true_1, bins=20, alpha=0.5, label='真实标签 = 1 (成功)')
        
        plt.title('预测为成功的概率分布 (2024年数据)')
        plt.xlabel('预测为成功的概率')
        plt.ylabel('样本数量')
        plt.legend()
        plt.grid(True)
        
        # 保存图表
        plt.savefig('output/probability_distribution_2024_fixed.png')
        print("2024年数据的概率分布图已修复并保存到 output/probability_distribution_2024_fixed.png")
    except Exception as e:
        print(f"修复2024年数据的概率分布图时出错: {e}")

def main():
    """主函数"""
    print("开始修复中文字体显示问题...")
    
    # 列出系统字体
    print("\n=== 系统字体 ===")
    list_system_fonts()
    
    # 查找中文字体
    print("\n=== 中文字体 ===")
    chinese_fonts = find_chinese_fonts()
    
    if chinese_fonts:
        print(f"\n找到 {len(chinese_fonts)} 个中文字体")
        
        # 使用第一个找到的中文字体进行测试
        font_name, font_path = chinese_fonts[0]
        print(f"将使用 {font_name} 进行测试")
        
        # 测试字体
        test_font_with_plot(font_name)
    else:
        print("\n未找到中文字体，将使用直接设置的方法")
        
        # 直接设置字体
        set_font_directly()
        
        # 测试字体
        test_font_with_plot()
    
    # 修复概率分布图
    print("\n=== 修复概率分布图 ===")
    fix_probability_distribution_plots()
    
    print("\n中文字体修复完成！")

if __name__ == "__main__":
    main()
