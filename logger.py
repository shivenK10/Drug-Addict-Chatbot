import logging
from logging.handlers import RotatingFileHandler
from typing import Literal

class Logger:
    """
    Logger implementation
    """
    def __init__(self, name: str, log_file_needed: bool = False, log_file_path: str = "", level: Literal['DEV', 'PROD'] = 'DEV'):
        """
        Initialization of Logger class:
        - name -> string -> Define the name of the logger, eg. a logger for monitoring data cleaning pipeline, name = "Data Cleaning Logs"
        - log_file_needed -> boolean (True or False) -> Whether or not you want to store the logs in ".log" file.
        - log_file_path -> string -> Mention the path for the log file.
        - level -> Literal (either "DEV" or "PROD") -> Select the level of logging, by default it is "DEV".
        """

        self.name = name
        self.log_file_needed = log_file_needed
        self.log_file_path = log_file_path

        self.logger = logging.getLogger(name)
        if level.upper() == 'DEV':
            self.logger.setLevel(logging.DEBUG)
        elif level.upper() == 'PROD':
            self.logger.setLevel(logging.INFO)
        else:
            raise ValueError("The value of level must be 'DEV' or 'PROD'")
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Clear any existing handlers
        self.logger.handlers.clear()

        if self.log_file_needed:
            # Validate that log_file is provided when log_file_needed is True
            if log_file_needed and not log_file_path.strip():
                raise ValueError("A file name is required when log_file_needed is set to True")
            
            # Rotating file handler with 5MB cap
            file_handler = RotatingFileHandler(
                log_file_path, 
                maxBytes=5*1024*1024,  # 5MB
                backupCount=5  # Keep 5 backup files
            )
            file_handler.setLevel(logging.DEBUG)

            file_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)
            

    def debug(self, message):
        """Log debug message"""
        self.logger.debug(message)

    def info(self, message):
        """Log info message"""
        self.logger.info(message)

    def warning(self, message):
        """Log warning message"""
        self.logger.warning(message)

    def error(self, message):
        """Log error message"""
        self.logger.error(message)

    def critical(self, message):
        """Log critical message"""
        self.logger.critical(message)
