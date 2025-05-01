"""
币安数据获取模块

此模块用于从币安API获取历史K线数据
"""
import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple


def get_historical_klines(
    symbol: str = "BTCUSDT",
    interval: str = "15m",
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
    limit: int = 1000,
    base_url: str = "https://api.binance.com"
) -> pd.DataFrame:
    """
    从币安API获取历史K线数据

    参数:
        symbol: 交易对，如 "BTCUSDT"
        interval: K线间隔，如 "15m", "1h", "1d" 等
        start_date: 开始日期，格式为 "YYYY-MM-DD"
        end_date: 结束日期，格式为 "YYYY-MM-DD"
        limit: 每次请求的最大数据条数，最大为1000
        base_url: 币安API基础URL

    返回:
        包含历史K线数据的DataFrame
    """
    # 转换日期为毫秒时间戳
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
    
    # 准备API端点
    endpoint = f"/api/v3/klines"
    
    # 准备存储所有K线数据的列表
    all_klines = []
    
    # 分批获取数据
    current_ts = start_ts
    while current_ts < end_ts:
        # 准备请求参数
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_ts,
            "endTime": end_ts,
            "limit": limit
        }
        
        # 发送请求
        url = base_url + endpoint
        response = requests.get(url, params=params)
        
        # 检查响应
        if response.status_code != 200:
            raise Exception(f"API请求失败: {response.text}")
        
        # 解析响应
        klines = response.json()
        
        # 如果没有数据，退出循环
        if not klines:
            break
        
        # 添加到总列表
        all_klines.extend(klines)
        
        # 更新时间戳，准备下一批请求
        current_ts = klines[-1][0] + 1
        
        print(f"已获取 {len(all_klines)} 条K线数据，当前时间: {datetime.fromtimestamp(current_ts/1000)}")
    
    # 如果没有数据，返回空DataFrame
    if not all_klines:
        return pd.DataFrame()
    
    # 转换为DataFrame
    df = pd.DataFrame(all_klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    
    # 转换数据类型
    numeric_columns = ["open", "high", "low", "close", "volume", 
                      "quote_asset_volume", "number_of_trades",
                      "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"]
    
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col])
    
    # 转换时间戳为日期时间
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    
    # 删除不需要的列
    df = df.drop(columns=["ignore"])
    
    return df


def save_to_csv(df: pd.DataFrame, file_path: str) -> None:
    """
    将DataFrame保存为CSV文件

    参数:
        df: 要保存的DataFrame
        file_path: 保存路径
    """
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # 保存为CSV
    df.to_csv(file_path, index=False)
    
    print(f"数据已保存到: {file_path}")


def load_from_csv(file_path: str) -> pd.DataFrame:
    """
    从CSV文件加载数据

    参数:
        file_path: CSV文件路径

    返回:
        加载的DataFrame
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 加载CSV
    df = pd.read_csv(file_path)
    
    # 转换时间列为日期时间类型
    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"])
    if "close_time" in df.columns:
        df["close_time"] = pd.to_datetime(df["close_time"])
    
    return df
