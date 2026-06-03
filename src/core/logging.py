"""
日志管理模块

提供统一的日志记录和管理功能
"""

import logging
import sys
from pathlib import Path
from typing import Optional


class LoggerManager:
    """
    日志管理器（单例模式）
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._loggers = {}
        self.log_dir = Path('./logs')
        self.log_dir.mkdir(exist_ok=True)
        self.default_level = logging.INFO
        self.formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def get_logger(self, name: str, level: Optional[int] = None) -> logging.Logger:
        """
        获取日志记录器
        
        Args:
            name: 日志器名称
            level: 日志级别
            
        Returns:
            日志记录器
        """
        if name in self._loggers:
            return self._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(level or self.default_level)
        
        if not logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(self.formatter)
            logger.addHandler(console_handler)
            
            try:
                file_handler = logging.FileHandler(
                    self.log_dir / f'{name}.log',
                    encoding='utf-8'
                )
                file_handler.setFormatter(self.formatter)
                logger.addHandler(file_handler)
            except Exception:
                pass  # 日志目录不可用时仅使用控制台
        
        self._loggers[name] = logger
        return logger
    
    def set_level(self, level: int) -> None:
        """设置全局日志级别"""
        self.default_level = level
        for logger in self._loggers.values():
            logger.setLevel(level)


logger_manager = LoggerManager()


def get_logger(name: str) -> logging.Logger:
    """获取日志记录器的便捷函数"""
    return logger_manager.get_logger(name)
