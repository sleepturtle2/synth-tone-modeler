import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

class LoggingUtil:
    _initialized = False
    
    @classmethod
    def initialize(cls, config: Optional[dict] = None):
        """Initialize logging system with optional config override"""
        if not cls._initialized:
            if not config:
                from .config_loader import get_config
                config = get_config("config.toml", "logging")
            
            log_file = Path(config.get("file", "logs/neural_synth.log"))
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            logging.basicConfig(
                level=config.get("level", "INFO"),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    RotatingFileHandler(
                        filename=str(log_file),
                        maxBytes=config.get("max_bytes", 10 * 1024 * 1024),  # 10MB
                        backupCount=config.get("backup_count", 5)
                    ),
                    logging.StreamHandler()
                ]
            )
            cls._initialized = True
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Get a configured logger instance"""
        if not LoggingUtil._initialized:
            LoggingUtil.initialize()
        return logging.getLogger(name)
    
    @staticmethod
    def setup_logger(name: str) -> logging.Logger:
        """Alternative name for get_logger for compatibility"""
        return LoggingUtil.get_logger(name)