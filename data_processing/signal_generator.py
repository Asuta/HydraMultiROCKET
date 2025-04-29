"""
信号生成器模块

此模块用于计算移动平均线、识别金叉/死叉信号，并根据未来价格走势标注信号成功/失败。
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Union, Optional


def calculate_moving_averages(df: pd.DataFrame,
                             short_period: int = 20,
                             long_period: int = 50,
                             price_col: str = 'close',
                             ma_type: str = 'sma') -> pd.DataFrame:
    """
    计算短期和长期移动平均线

    参数:
        df: 包含价格数据的DataFrame
        short_period: 短期移动平均线周期
        long_period: 长期移动平均线周期
        price_col: 用于计算移动平均线的价格列名
        ma_type: 移动平均线类型，'sma'表示简单移动平均线，'ema'表示指数移动平均线

    返回:
        添加了移动平均线的DataFrame
    """
    # 创建DataFrame的副本，避免修改原始数据
    result_df = df.copy()

    # 计算移动平均线
    if ma_type.lower() == 'sma':
        # 简单移动平均线
        result_df[f'ma_short'] = result_df[price_col].rolling(window=short_period).mean()
        result_df[f'ma_long'] = result_df[price_col].rolling(window=long_period).mean()
    elif ma_type.lower() == 'ema':
        # 指数移动平均线
        result_df[f'ma_short'] = result_df[price_col].ewm(span=short_period, adjust=False).mean()
        result_df[f'ma_long'] = result_df[price_col].ewm(span=long_period, adjust=False).mean()
    else:
        raise ValueError(f"不支持的移动平均线类型: {ma_type}，支持的类型有 'sma' 和 'ema'")

    return result_df


def identify_crossover_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    识别移动平均线金叉和死叉信号

    金叉: 短期均线从下方穿过长期均线
    死叉: 短期均线从上方穿过长期均线

    参数:
        df: 包含移动平均线的DataFrame，必须包含'ma_short'和'ma_long'列

    返回:
        添加了信号列的DataFrame，其中:
        - signal = 1 表示金叉（买入信号）
        - signal = -1 表示死叉（卖出信号）
        - signal = 0 表示无信号
    """
    # 检查必要的列是否存在
    required_cols = ['ma_short', 'ma_long']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"输入DataFrame缺少必要的列: {col}")

    # 创建DataFrame的副本
    result_df = df.copy()

    # 初始化信号列
    result_df['signal'] = 0

    # 计算前一天的短期和长期均线差值
    result_df['prev_diff'] = result_df['ma_short'].shift(1) - result_df['ma_long'].shift(1)

    # 计算当天的短期和长期均线差值
    result_df['curr_diff'] = result_df['ma_short'] - result_df['ma_long']

    # 识别金叉信号（短期均线从下方穿过长期均线）
    golden_cross = (result_df['prev_diff'] < 0) & (result_df['curr_diff'] > 0)
    result_df.loc[golden_cross, 'signal'] = 1

    # 识别死叉信号（短期均线从上方穿过长期均线）
    death_cross = (result_df['prev_diff'] > 0) & (result_df['curr_diff'] < 0)
    result_df.loc[death_cross, 'signal'] = -1

    # 删除临时列
    result_df = result_df.drop(['prev_diff', 'curr_diff'], axis=1)

    return result_df


def label_signals(df: pd.DataFrame,
                 profit_target: float = 0.005,
                 stop_loss: float = 0.005,
                 max_bars: int = 20,
                 price_col: str = 'close') -> pd.DataFrame:
    """
    根据未来价格走势标注信号成功、失败或无效

    参数:
        df: 包含信号的DataFrame，必须包含'signal'列
        profit_target: 盈利目标（百分比，如0.005表示0.5%）
        stop_loss: 止损水平（百分比，如0.005表示0.5%）
        max_bars: 最大观察K线数量
        price_col: 价格列名

    返回:
        添加了标签列的DataFrame，其中:
        - success = 1 表示信号成功（达到盈利目标）
        - success = 0 表示信号失败（触发止损位）
        - success = 2 表示信号无效（在观察期内既没有达到盈利目标，也没有触发止损位）
    """
    # 检查必要的列是否存在
    required_cols = ['signal', price_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"输入DataFrame缺少必要的列: {col}")

    # 创建DataFrame的副本
    result_df = df.copy()

    # 初始化成功标签列
    result_df['success'] = np.nan

    # 获取所有信号的索引
    signal_indices = result_df[result_df['signal'] != 0].index

    # 对每个信号进行标注
    for idx in signal_indices:
        signal_type = result_df.loc[idx, 'signal']
        entry_price = result_df.loc[idx, price_col]

        # 确定观察范围（不超过DataFrame的末尾）
        end_idx = min(idx + max_bars, len(result_df) - 1)

        # 如果信号太靠近数据末尾，无法完整观察，则跳过
        if end_idx - idx < max_bars:
            continue

        # 提取未来价格数据
        future_prices = result_df.loc[idx+1:end_idx, price_col].values

        # 根据信号类型计算目标价格
        if signal_type == 1:  # 金叉（买入信号）
            profit_price = entry_price * (1 + profit_target)
            stop_price = entry_price * (1 - stop_loss)

            # 检查是否达到盈利目标或止损
            reached_profit = any(price >= profit_price for price in future_prices)
            reached_stop = any(price <= stop_price for price in future_prices)

            # 判断信号结果
            if reached_profit and (not reached_stop or
                                  np.argmax(future_prices >= profit_price) <
                                  np.argmax(future_prices <= stop_price)):
                # 先达到盈利目标，标记为成功
                result_df.loc[idx, 'success'] = 1
            elif reached_stop and (not reached_profit or
                                  np.argmax(future_prices <= stop_price) <
                                  np.argmax(future_prices >= profit_price)):
                # 先触及止损位，标记为失败
                result_df.loc[idx, 'success'] = 0
            else:
                # 既没有达到盈利目标，也没有触及止损位，标记为无效
                result_df.loc[idx, 'success'] = 2

        elif signal_type == -1:  # 死叉（卖出信号）
            profit_price = entry_price * (1 - profit_target)
            stop_price = entry_price * (1 + stop_loss)

            # 检查是否达到盈利目标或止损
            reached_profit = any(price <= profit_price for price in future_prices)
            reached_stop = any(price >= stop_price for price in future_prices)

            # 判断信号结果
            if reached_profit and (not reached_stop or
                                  np.argmax(future_prices <= profit_price) <
                                  np.argmax(future_prices >= stop_price)):
                # 先达到盈利目标，标记为成功
                result_df.loc[idx, 'success'] = 1
            elif reached_stop and (not reached_profit or
                                  np.argmax(future_prices >= stop_price) <
                                  np.argmax(future_prices <= profit_price)):
                # 先触及止损位，标记为失败
                result_df.loc[idx, 'success'] = 0
            else:
                # 既没有达到盈利目标，也没有触及止损位，标记为无效
                result_df.loc[idx, 'success'] = 2

    return result_df


def get_signal_statistics(df: pd.DataFrame) -> Dict[str, Union[int, float]]:
    """
    获取信号统计信息

    参数:
        df: 包含信号和标签的DataFrame

    返回:
        包含统计信息的字典
    """
    # 过滤有效的信号（已标注的信号）
    valid_signals = df.dropna(subset=['success'])

    # 计算统计信息
    total_signals = len(valid_signals)
    if total_signals == 0:
        return {
            'total_signals': 0,
            'golden_cross_count': 0,
            'death_cross_count': 0,
            'success_count': 0,
            'failure_count': 0,
            'invalid_count': 0,
            'success_rate': 0.0,
            'failure_rate': 0.0,
            'invalid_rate': 0.0,
            'golden_cross_success_rate': 0.0,
            'golden_cross_failure_rate': 0.0,
            'golden_cross_invalid_rate': 0.0,
            'death_cross_success_rate': 0.0,
            'death_cross_failure_rate': 0.0,
            'death_cross_invalid_rate': 0.0
        }

    # 按信号类型分组
    golden_cross = valid_signals[valid_signals['signal'] == 1]
    death_cross = valid_signals[valid_signals['signal'] == -1]

    golden_cross_count = len(golden_cross)
    death_cross_count = len(death_cross)

    # 按结果分组
    success_signals = valid_signals[valid_signals['success'] == 1]  # 成功
    failure_signals = valid_signals[valid_signals['success'] == 0]  # 失败
    invalid_signals = valid_signals[valid_signals['success'] == 2]  # 无效

    success_count = len(success_signals)
    failure_count = len(failure_signals)
    invalid_count = len(invalid_signals)

    # 计算总体比率
    success_rate = success_count / total_signals if total_signals > 0 else 0
    failure_rate = failure_count / total_signals if total_signals > 0 else 0
    invalid_rate = invalid_count / total_signals if total_signals > 0 else 0

    # 计算金叉信号的比率
    golden_cross_success = len(golden_cross[golden_cross['success'] == 1])
    golden_cross_failure = len(golden_cross[golden_cross['success'] == 0])
    golden_cross_invalid = len(golden_cross[golden_cross['success'] == 2])

    golden_cross_success_rate = golden_cross_success / golden_cross_count if golden_cross_count > 0 else 0
    golden_cross_failure_rate = golden_cross_failure / golden_cross_count if golden_cross_count > 0 else 0
    golden_cross_invalid_rate = golden_cross_invalid / golden_cross_count if golden_cross_count > 0 else 0

    # 计算死叉信号的比率
    death_cross_success = len(death_cross[death_cross['success'] == 1])
    death_cross_failure = len(death_cross[death_cross['success'] == 0])
    death_cross_invalid = len(death_cross[death_cross['success'] == 2])

    death_cross_success_rate = death_cross_success / death_cross_count if death_cross_count > 0 else 0
    death_cross_failure_rate = death_cross_failure / death_cross_count if death_cross_count > 0 else 0
    death_cross_invalid_rate = death_cross_invalid / death_cross_count if death_cross_count > 0 else 0

    return {
        'total_signals': total_signals,
        'golden_cross_count': golden_cross_count,
        'death_cross_count': death_cross_count,
        'success_count': success_count,
        'failure_count': failure_count,
        'invalid_count': invalid_count,
        'success_rate': success_rate,
        'failure_rate': failure_rate,
        'invalid_rate': invalid_rate,
        'golden_cross_success_rate': golden_cross_success_rate,
        'golden_cross_failure_rate': golden_cross_failure_rate,
        'golden_cross_invalid_rate': golden_cross_invalid_rate,
        'death_cross_success_rate': death_cross_success_rate,
        'death_cross_failure_rate': death_cross_failure_rate,
        'death_cross_invalid_rate': death_cross_invalid_rate
    }
