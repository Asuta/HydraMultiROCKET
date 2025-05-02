"""
数据准备主脚本

此脚本用于准备Hydra+MultiROCKET模型的训练数据，包括：
1. 获取币安BTC历史数据
2. 计算移动平均线
3. 识别金叉/死叉信号
4. 标注信号成功/失败
5. 提取特征
6. 划分数据集
7. 保存处理后的数据集
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split

from src.visualization.fix_chinese_font import fix_chinese_display

# 修复中文显示问题
fix_chinese_display()

from data_processing.binance_data_fetcher import get_historical_klines, save_to_csv
from data_processing.signal_generator import (
    calculate_moving_averages,
    identify_crossover_signals,
    label_signals,
    get_signal_statistics
)
from data_processing.feature_extractor import (
    prepare_dataset_from_signals,
    create_train_val_test_split
)
from src.data.data_loader import save_dataset


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='准备Hydra+MultiROCKET模型的训练数据')

    # 数据获取参数
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='交易对')
    parser.add_argument('--interval', type=str, default='15m', help='K线间隔')
    parser.add_argument('--start_date', type=str, default='2023-01-01', help='开始日期')
    parser.add_argument('--end_date', type=str, default='2023-12-31', help='结束日期')
    parser.add_argument('--use_existing_data', action='store_true', help='使用现有数据文件而不是重新获取')
    parser.add_argument('--data_file', type=str, help='现有数据文件路径')

    # 移动平均线参数
    parser.add_argument('--ma_short', type=int, default=20, help='短期移动平均线周期')
    parser.add_argument('--ma_long', type=int, default=50, help='长期移动平均线周期')
    parser.add_argument('--ma_type', type=str, default='sma', choices=['sma', 'ema'], help='移动平均线类型')

    # 信号标注参数
    parser.add_argument('--profit_target', type=float, default=0.005, help='盈利目标（百分比，如0.005表示0.5%）')
    parser.add_argument('--stop_loss', type=float, default=0.005, help='止损水平（百分比，如0.005表示0.5%）')
    parser.add_argument('--max_bars', type=int, default=20, help='最大观察K线数量')

    # 特征提取参数
    parser.add_argument('--segment_length', type=int, default=40, help='时间序列片段长度')
    parser.add_argument('--normalize', action='store_true', help='是否标准化数据')
    parser.add_argument('--normalize_method', type=str, default='zscore', choices=['zscore', 'minmax'], help='标准化方法')

    # 数据集划分参数
    parser.add_argument('--train_size', type=float, default=0.7, help='训练集比例')
    parser.add_argument('--val_size', type=float, default=0.15, help='验证集比例')
    parser.add_argument('--test_size', type=float, default=0.15, help='测试集比例')
    parser.add_argument('--random_state', type=int, default=42, help='随机种子')

    # 输出参数
    parser.add_argument('--output_dir', type=str, default='output/prepared_data', help='输出目录')

    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置日志文件
    log_file = os.path.join(args.output_dir, 'prepare_data.log')

    # 记录开始时间
    start_time = datetime.now()
    print(f"开始数据准备: {start_time}")

    # 1. 获取或加载数据
    if args.use_existing_data and args.data_file:
        print(f"从文件加载数据: {args.data_file}")
        df = pd.read_csv(args.data_file)

        # 确保时间列是日期时间类型
        if 'open_time' in df.columns:
            df['open_time'] = pd.to_datetime(df['open_time'])
        if 'close_time' in df.columns:
            df['close_time'] = pd.to_datetime(df['close_time'])
    else:
        print(f"从币安获取 {args.symbol} 的历史数据...")
        df = get_historical_klines(
            symbol=args.symbol,
            interval=args.interval,
            start_date=args.start_date,
            end_date=args.end_date
        )

        # 保存原始数据
        raw_data_file = os.path.join(args.output_dir, f"{args.symbol}_{args.interval}_raw_data.csv")
        save_to_csv(df, raw_data_file)

    # 打印数据基本信息
    print(f"数据加载完成，共 {len(df)} 条记录")
    print(f"数据时间范围: {df['open_time'].min()} 到 {df['open_time'].max()}")

    # 2. 计算移动平均线
    print(f"计算移动平均线 (短期: {args.ma_short}, 长期: {args.ma_long}, 类型: {args.ma_type})...")
    df = calculate_moving_averages(
        df=df,
        short_period=args.ma_short,
        long_period=args.ma_long,
        price_col='close',
        ma_type=args.ma_type
    )

    # 3. 识别金叉/死叉信号
    print("识别金叉/死叉信号...")
    df = identify_crossover_signals(df)

    # 4. 标注信号成功/失败
    print(f"标注信号成功/失败 (盈利目标: {args.profit_target*100}%, 止损: {args.stop_loss*100}%, 最大观察K线: {args.max_bars})...")
    df = label_signals(
        df=df,
        profit_target=args.profit_target,
        stop_loss=args.stop_loss,
        max_bars=args.max_bars,
        price_col='close'
    )

    # 保存处理后的数据
    processed_data_file = os.path.join(args.output_dir, f"{args.symbol}_{args.interval}_processed_data.csv")
    save_to_csv(df, processed_data_file)

    # 5. 获取信号统计信息
    print("计算信号统计信息...")
    stats = get_signal_statistics(df)
    print(f"总信号数: {stats['total_signals']}")
    print(f"金叉信号: {stats['golden_cross_count']}")
    print(f"  - 成功率: {stats['golden_cross_success_rate']*100:.2f}%")
    print(f"  - 失败率: {stats['golden_cross_failure_rate']*100:.2f}%")
    print(f"  - 无效率: {stats['golden_cross_invalid_rate']*100:.2f}%")
    print(f"死叉信号: {stats['death_cross_count']}")
    print(f"  - 成功率: {stats['death_cross_success_rate']*100:.2f}%")
    print(f"  - 失败率: {stats['death_cross_failure_rate']*100:.2f}%")
    print(f"  - 无效率: {stats['death_cross_invalid_rate']*100:.2f}%")
    print(f"总体统计:")
    print(f"  - 成功信号: {stats['success_count']} ({stats['success_rate']*100:.2f}%)")
    print(f"  - 失败信号: {stats['failure_count']} ({stats['failure_rate']*100:.2f}%)")
    print(f"  - 无效信号: {stats['invalid_count']} ({stats['invalid_rate']*100:.2f}%)")

    # 6. 提取特征
    print(f"提取特征 (片段长度: {args.segment_length})...")
    feature_columns = ['open', 'high', 'low', 'close', 'volume', 'ma_short', 'ma_long']
    X, y = prepare_dataset_from_signals(
        df=df,
        segment_length=args.segment_length,
        feature_columns=feature_columns,
        normalize=args.normalize,
        normalize_method=args.normalize_method,
        for_aeon=True  # 直接生成适合aeon库的数据格式
    )

    print(f"提取的特征形状: {X.shape}, 标签形状: {y.shape}")

    # 如果没有足够的数据，则退出
    if len(X) < 10:
        print("警告: 提取的特征数量太少，无法继续处理")
        return

    # 7. 按时间顺序划分数据集
    print("划分数据集...")
    X_train, y_train, X_val, y_val, X_test, y_test = create_train_val_test_split(
        X=X,
        y=y,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state,
        shuffle=False  # 不打乱数据，保持时间顺序
    )

    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")

    # 8. 保存数据集
    print("保存数据集...")
    train_file = os.path.join(args.output_dir, "train_dataset.npz")
    val_file = os.path.join(args.output_dir, "val_dataset.npz")
    test_file = os.path.join(args.output_dir, "test_dataset.npz")

    save_dataset(X_train, y_train, train_file)
    save_dataset(X_val, y_val, val_file)
    save_dataset(X_test, y_test, test_file)

    # 9. 绘制数据分布图
    print("绘制数据分布图...")
    plt.figure(figsize=(12, 8))

    # 绘制价格和移动平均线
    plt.subplot(2, 1, 1)
    plt.plot(df['open_time'], df['close'], label='收盘价')
    plt.plot(df['open_time'], df['ma_short'], label=f'短期MA ({args.ma_short})')
    plt.plot(df['open_time'], df['ma_long'], label=f'长期MA ({args.ma_long})')

    # 标记金叉和死叉
    golden_cross = df[df['signal'] == 1]
    death_cross = df[df['signal'] == -1]

    plt.scatter(golden_cross['open_time'], golden_cross['close'],
               color='green', marker='^', label='金叉')
    plt.scatter(death_cross['open_time'], death_cross['close'],
               color='red', marker='v', label='死叉')

    plt.title(f'{args.symbol} 价格和移动平均线')
    plt.xlabel('时间')
    plt.ylabel('价格')
    plt.legend()
    plt.grid(True)

    # 绘制信号成功/失败/无效分布
    plt.subplot(2, 1, 2)

    # 获取有效的信号
    valid_signals = df.dropna(subset=['success'])

    # 成功的信号
    success_signals = valid_signals[valid_signals['success'] == 1]
    # 失败的信号
    failure_signals = valid_signals[valid_signals['success'] == 0]
    # 无效的信号
    invalid_signals = valid_signals[valid_signals['success'] == 2]

    # 绘制成功、失败和无效的信号
    plt.scatter(success_signals['open_time'], success_signals['signal'],
               color='green', marker='o', label='成功信号')
    plt.scatter(failure_signals['open_time'], failure_signals['signal'],
               color='red', marker='x', label='失败信号')
    plt.scatter(invalid_signals['open_time'], invalid_signals['signal'],
               color='blue', marker='s', label='无效信号')

    plt.title('信号分类分布')
    plt.xlabel('时间')
    plt.ylabel('信号类型 (1=金叉, -1=死叉)')
    plt.yticks([-1, 1], ['死叉', '金叉'])
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"{args.symbol}_signals_distribution.png"))

    # 记录结束时间
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"数据准备完成: {end_time}")
    print(f"总耗时: {duration}")

    # 保存配置信息
    config = vars(args)
    config['start_time'] = start_time.strftime("%Y-%m-%d %H:%M:%S")
    config['end_time'] = end_time.strftime("%Y-%m-%d %H:%M:%S")
    config['duration'] = str(duration)
    config['total_samples'] = len(X)
    config['train_samples'] = len(X_train)
    config['val_samples'] = len(X_val)
    config['test_samples'] = len(X_test)
    config['signal_stats'] = stats

    # 将配置保存为文本文件
    with open(os.path.join(args.output_dir, 'config.txt'), 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

    print(f"所有输出已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
