"""
数据处理模块

此包包含用于数据预处理、转换和准备的各种功能。
"""

from data_processing.binance_data_fetcher import (
    get_historical_klines,
    get_order_book,
    save_to_csv,
    plot_price_volume
)

from data_processing.signal_generator import (
    calculate_moving_averages,
    identify_crossover_signals,
    label_signals,
    get_signal_statistics
)

from data_processing.feature_extractor import (
    extract_time_series_segments,
    normalize_time_series,
    prepare_dataset_from_signals
)

__version__ = '0.1.0'
