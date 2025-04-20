"""
日志模块
"""
import os
import logging
from typing import Optional


def setup_logger(name: str, 
                log_file: Optional[str] = None, 
                level: int = logging.INFO) -> logging.Logger:
    """
    设置日志记录器
    
    参数:
        name: 日志记录器名称
        log_file: 日志文件路径
        level: 日志级别
        
    返回:
        日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(console_handler)
    
    # 如果指定了日志文件，添加文件处理器
    if log_file:
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # 创建文件处理器
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        
        # 添加处理器到日志记录器
        logger.addHandler(file_handler)
    
    return logger
