"""
信号生成模块

此模块用于生成交易信号和标签
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple


def calculate_moving_averages(
    df: pd.DataFrame,
    short_period: int = 20,
    long_period: int = 50,
    price_col: str = "close",
    ma_type: str = "sma"
) -> pd.DataFrame:
    """
    计算移动平均线

    参数:
        df: 包含价格数据的DataFrame
        short_period: 短期移动平均线周期
        long_period: 长期移动平均线周期
        price_col: 价格列名
        ma_type: 移动平均线类型，可选 "sma" 或 "ema"

    返回:
        添加了移动平均线的DataFrame
    """
    # 创建DataFrame的副本，避免修改原始数据
    result_df = df.copy()
    
    # 计算短期移动平均线
    if ma_type.lower() == "sma":
        result_df["ma_short"] = result_df[price_col].rolling(window=short_period).mean()
    elif ma_type.lower() == "ema":
        result_df["ma_short"] = result_df[price_col].ewm(span=short_period, adjust=False).mean()
    else:
        raise ValueError(f"不支持的移动平均线类型: {ma_type}")
    
    # 计算长期移动平均线
    if ma_type.lower() == "sma":
        result_df["ma_long"] = result_df[price_col].rolling(window=long_period).mean()
    elif ma_type.lower() == "ema":
        result_df["ma_long"] = result_df[price_col].ewm(span=long_period, adjust=False).mean()
    
    return result_df


def identify_crossover_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    识别金叉和死叉信号

    参数:
        df: 包含移动平均线的DataFrame

    返回:
        添加了信号的DataFrame
    """
    # 创建DataFrame的副本，避免修改原始数据
    result_df = df.copy()
    
    # 初始化信号列
    result_df["signal"] = 0
    
    # 计算前一天的MA差值
    result_df["prev_ma_diff"] = result_df["ma_short"].shift(1) - result_df["ma_long"].shift(1)
    
    # 计算当天的MA差值
    result_df["curr_ma_diff"] = result_df["ma_short"] - result_df["ma_long"]
    
    # 识别金叉信号（短期MA从下方穿过长期MA）
    golden_cross = (result_df["prev_ma_diff"] < 0) & (result_df["curr_ma_diff"] > 0)
    result_df.loc[golden_cross, "signal"] = 1
    
    # 识别死叉信号（短期MA从上方穿过长期MA）
    death_cross = (result_df["prev_ma_diff"] > 0) & (result_df["curr_ma_diff"] < 0)
    result_df.loc[death_cross, "signal"] = -1
    
    # 删除临时列
    result_df = result_df.drop(columns=["prev_ma_diff", "curr_ma_diff"])
    
    return result_df


def label_signals(
    df: pd.DataFrame,
    profit_target: float = 0.005,  # 0.5%
    stop_loss: float = 0.005,      # 0.5%
    max_bars: int = 20,
    price_col: str = "close"
) -> pd.DataFrame:
    """
    标注信号成功或失败

    参数:
        df: 包含信号的DataFrame
        profit_target: 盈利目标（百分比）
        stop_loss: 止损水平（百分比）
        max_bars: 最大观察K线数量
        price_col: 价格列名

    返回:
        添加了标签的DataFrame
    """
    # 创建DataFrame的副本，避免修改原始数据
    result_df = df.copy()
    
    # 初始化成功列
    result_df["success"] = np.nan
    
    # 获取所有信号的索引
    signal_indices = result_df[result_df["signal"] != 0].index
    
    # 遍历每个信号
    for idx in signal_indices:
        signal_type = result_df.loc[idx, "signal"]
        entry_price = result_df.loc[idx, price_col]
        
        # 计算目标价格
        if signal_type == 1:  # 金叉，做多
            target_price = entry_price * (1 + profit_target)
            stop_price = entry_price * (1 - stop_loss)
        else:  # 死叉，做空
            target_price = entry_price * (1 - profit_target)
            stop_price = entry_price * (1 + stop_loss)
        
        # 获取后续K线
        if idx + max_bars < len(result_df):
            future_bars = result_df.loc[idx+1:idx+max_bars]
        else:
            future_bars = result_df.loc[idx+1:]
        
        # 如果没有足够的后续K线，标记为无效
        if len(future_bars) == 0:
            result_df.loc[idx, "success"] = 2  # 2表示无效
            continue
        
        # 检查是否达到盈利目标或止损
        if signal_type == 1:  # 金叉，做多
            # 检查是否先达到盈利目标
            hit_target = future_bars[future_bars[price_col] >= target_price]
            if len(hit_target) > 0:
                first_target_idx = hit_target.index[0]
                
                # 检查是否先达到止损
                hit_stop = future_bars[future_bars[price_col] <= stop_price]
                if len(hit_stop) > 0:
                    first_stop_idx = hit_stop.index[0]
                    
                    # 比较哪个先达到
                    if first_target_idx < first_stop_idx:
                        result_df.loc[idx, "success"] = 1  # 成功
                    else:
                        result_df.loc[idx, "success"] = 0  # 失败
                else:
                    result_df.loc[idx, "success"] = 1  # 成功
            else:
                # 检查是否达到止损
                hit_stop = future_bars[future_bars[price_col] <= stop_price]
                if len(hit_stop) > 0:
                    result_df.loc[idx, "success"] = 0  # 失败
                else:
                    result_df.loc[idx, "success"] = 2  # 无效（既没有达到盈利目标也没有达到止损）
        else:  # 死叉，做空
            # 检查是否先达到盈利目标
            hit_target = future_bars[future_bars[price_col] <= target_price]
            if len(hit_target) > 0:
                first_target_idx = hit_target.index[0]
                
                # 检查是否先达到止损
                hit_stop = future_bars[future_bars[price_col] >= stop_price]
                if len(hit_stop) > 0:
                    first_stop_idx = hit_stop.index[0]
                    
                    # 比较哪个先达到
                    if first_target_idx < first_stop_idx:
                        result_df.loc[idx, "success"] = 1  # 成功
                    else:
                        result_df.loc[idx, "success"] = 0  # 失败
                else:
                    result_df.loc[idx, "success"] = 1  # 成功
            else:
                # 检查是否达到止损
                hit_stop = future_bars[future_bars[price_col] >= stop_price]
                if len(hit_stop) > 0:
                    result_df.loc[idx, "success"] = 0  # 失败
                else:
                    result_df.loc[idx, "success"] = 2  # 无效（既没有达到盈利目标也没有达到止损）
    
    return result_df


def get_signal_statistics(df: pd.DataFrame) -> Dict[str, Union[int, float]]:
    """
    获取信号统计信息

    参数:
        df: 包含信号和标签的DataFrame

    返回:
        包含统计信息的字典
    """
    # 获取有效的信号（已标注的信号）
    valid_signals = df.dropna(subset=["success"])
    
    # 总信号数
    total_signals = len(valid_signals)
    
    if total_signals == 0:
        return {
            "total_signals": 0,
            "golden_cross_count": 0,
            "death_cross_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "invalid_count": 0,
            "success_rate": 0.0,
            "failure_rate": 0.0,
            "invalid_rate": 0.0,
            "golden_cross_success_rate": 0.0,
            "golden_cross_failure_rate": 0.0,
            "golden_cross_invalid_rate": 0.0,
            "death_cross_success_rate": 0.0,
            "death_cross_failure_rate": 0.0,
            "death_cross_invalid_rate": 0.0
        }
    
    # 金叉信号
    golden_cross = valid_signals[valid_signals["signal"] == 1]
    golden_cross_count = len(golden_cross)
    
    # 死叉信号
    death_cross = valid_signals[valid_signals["signal"] == -1]
    death_cross_count = len(death_cross)
    
    # 成功信号
    success_signals = valid_signals[valid_signals["success"] == 1]
    success_count = len(success_signals)
    
    # 失败信号
    failure_signals = valid_signals[valid_signals["success"] == 0]
    failure_count = len(failure_signals)
    
    # 无效信号
    invalid_signals = valid_signals[valid_signals["success"] == 2]
    invalid_count = len(invalid_signals)
    
    # 计算比率
    success_rate = success_count / total_signals if total_signals > 0 else 0
    failure_rate = failure_count / total_signals if total_signals > 0 else 0
    invalid_rate = invalid_count / total_signals if total_signals > 0 else 0
    
    # 金叉成功率
    golden_cross_success = golden_cross[golden_cross["success"] == 1]
    golden_cross_success_rate = len(golden_cross_success) / golden_cross_count if golden_cross_count > 0 else 0
    
    # 金叉失败率
    golden_cross_failure = golden_cross[golden_cross["success"] == 0]
    golden_cross_failure_rate = len(golden_cross_failure) / golden_cross_count if golden_cross_count > 0 else 0
    
    # 金叉无效率
    golden_cross_invalid = golden_cross[golden_cross["success"] == 2]
    golden_cross_invalid_rate = len(golden_cross_invalid) / golden_cross_count if golden_cross_count > 0 else 0
    
    # 死叉成功率
    death_cross_success = death_cross[death_cross["success"] == 1]
    death_cross_success_rate = len(death_cross_success) / death_cross_count if death_cross_count > 0 else 0
    
    # 死叉失败率
    death_cross_failure = death_cross[death_cross["success"] == 0]
    death_cross_failure_rate = len(death_cross_failure) / death_cross_count if death_cross_count > 0 else 0
    
    # 死叉无效率
    death_cross_invalid = death_cross[death_cross["success"] == 2]
    death_cross_invalid_rate = len(death_cross_invalid) / death_cross_count if death_cross_count > 0 else 0
    
    # 返回统计信息
    return {
        "total_signals": total_signals,
        "golden_cross_count": golden_cross_count,
        "death_cross_count": death_cross_count,
        "success_count": success_count,
        "failure_count": failure_count,
        "invalid_count": invalid_count,
        "success_rate": success_rate,
        "failure_rate": failure_rate,
        "invalid_rate": invalid_rate,
        "golden_cross_success_rate": golden_cross_success_rate,
        "golden_cross_failure_rate": golden_cross_failure_rate,
        "golden_cross_invalid_rate": golden_cross_invalid_rate,
        "death_cross_success_rate": death_cross_success_rate,
        "death_cross_failure_rate": death_cross_failure_rate,
        "death_cross_invalid_rate": death_cross_invalid_rate
    }
