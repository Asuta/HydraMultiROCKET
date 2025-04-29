"""
币安数据获取脚本

此脚本用于从币安获取比特币历史价格数据、交易量和订单簿信息。
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
from binance.client import Client
import csv
from dateutil import parser

# 币安API配置
# 注意：这里使用的是公共API，不需要API密钥也可以获取历史数据
# 如果需要获取账户信息或下单，则需要提供API密钥
api_key = ""
api_secret = ""

# 初始化币安客户端
client = Client(api_key, api_secret)

def get_historical_klines(symbol, interval, start_date, end_date):
    """
    获取历史K线数据

    参数:
        symbol (str): 交易对，例如 'BTCUSDT'
        interval (str): K线间隔，例如 '15m', '1h', '1d'
        start_date (str): 开始日期，例如 '2023-06-01'
        end_date (str): 结束日期，例如 '2023-12-31'

    返回:
        pandas.DataFrame: 包含历史K线数据的DataFrame
    """
    print(f"获取 {symbol} 从 {start_date} 到 {end_date} 的 {interval} K线数据...")

    # 将日期字符串转换为时间戳（毫秒）
    start_ts = int(parser.parse(start_date).timestamp() * 1000)
    end_ts = int(parser.parse(end_date).timestamp() * 1000)

    # 获取历史K线数据
    klines = client.get_historical_klines(
        symbol=symbol,
        interval=interval,
        start_str=start_ts,
        end_str=end_ts
    )

    # 将数据转换为DataFrame
    columns = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ]
    df = pd.DataFrame(klines, columns=columns)

    # 转换时间戳为日期时间
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

    # 转换数值列为浮点数
    numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                       'quote_asset_volume', 'number_of_trades',
                       'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
    df[numeric_columns] = df[numeric_columns].astype(float)

    return df

def get_order_book(symbol, limit=1000):
    """
    获取当前订单簿数据

    参数:
        symbol (str): 交易对，例如 'BTCUSDT'
        limit (int): 订单簿深度，最大值为5000

    返回:
        dict: 包含订单簿数据的字典
    """
    print(f"获取 {symbol} 的订单簿数据（深度: {limit}）...")
    return client.get_order_book(symbol=symbol, limit=limit)

def save_to_csv(df, filename):
    """
    将DataFrame保存为CSV文件

    参数:
        df (pandas.DataFrame): 要保存的DataFrame
        filename (str): 文件名
    """
    print(f"保存数据到 {filename}...")
    df.to_csv(filename, index=False)
    print(f"数据已保存到 {filename}")

def plot_price_volume(df, symbol):
    """
    绘制价格和交易量图表

    参数:
        df (pandas.DataFrame): 包含价格和交易量数据的DataFrame
        symbol (str): 交易对，例如 'BTCUSDT'
    """
    plt.figure(figsize=(14, 10))

    # 绘制价格图
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(df['open_time'], df['close'], label='收盘价')
    ax1.set_title(f'{symbol} 价格走势')
    ax1.set_ylabel('价格 (USDT)')
    ax1.legend()
    ax1.grid(True)

    # 绘制交易量图
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.bar(df['open_time'], df['volume'], label='交易量')
    ax2.set_title(f'{symbol} 交易量')
    ax2.set_xlabel('时间')
    ax2.set_ylabel('交易量 (BTC)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    chart_path = f"output/binance_data/{symbol}_price_volume_chart.png"
    plt.savefig(chart_path)
    print(f"图表已保存为 {chart_path}")

def main():
    """主函数"""
    # 设置参数
    symbol = 'BTCUSDT'
    interval = '15m'  # 15分钟K线
    start_date = '2023-06-01'
    end_date = '2023-12-31'

    # 创建输出目录
    output_dir = 'output/binance_data'
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 获取历史K线数据
        klines_df = get_historical_klines(symbol, interval, start_date, end_date)

        # 保存K线数据到CSV
        klines_csv = f"{output_dir}/{symbol}_{interval}_{start_date}_to_{end_date}.csv"
        save_to_csv(klines_df, klines_csv)

        # 打印数据统计信息
        print("\n数据统计信息:")
        print(f"总记录数: {len(klines_df)}")
        print(f"日期范围: {klines_df['open_time'].min()} 到 {klines_df['open_time'].max()}")
        print(f"价格范围: {klines_df['low'].min()} 到 {klines_df['high'].max()}")
        print(f"平均交易量: {klines_df['volume'].mean()}")
        print(f"总交易量: {klines_df['volume'].sum()}")

        # 绘制价格和交易量图表
        plot_price_volume(klines_df, symbol)

        # 获取当前订单簿数据
        order_book = get_order_book(symbol)

        # 将订单簿数据转换为DataFrame
        bids_df = pd.DataFrame(order_book['bids'], columns=['price', 'quantity'])
        asks_df = pd.DataFrame(order_book['asks'], columns=['price', 'quantity'])

        # 转换为数值类型
        bids_df = bids_df.astype(float)
        asks_df = asks_df.astype(float)

        # 保存订单簿数据到CSV
        bids_csv = f"{output_dir}/{symbol}_bids_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        asks_csv = f"{output_dir}/{symbol}_asks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        save_to_csv(bids_df, bids_csv)
        save_to_csv(asks_df, asks_csv)

        print("\n币安API可获取的数据类型:")
        print("1. K线数据 (OHLCV) - 可获取历史数据")
        print("   - 开盘价、最高价、最低价、收盘价")
        print("   - 交易量 (基础资产和报价资产)")
        print("   - 交易笔数")
        print("   - 买方成交量 (基础资产和报价资产)")
        print("2. 订单簿数据 - 仅当前数据，历史数据有限制")
        print("   - 买单 (价格和数量)")
        print("   - 卖单 (价格和数量)")
        print("3. 最新成交 - 可获取近期成交")
        print("4. 24小时价格变动统计")
        print("5. 交易对信息 (价格精度、数量精度等)")

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()
