import os
import json
import logging
from pythonjsonlogger import jsonlogger  # pyre-ignore

def setup_logger(name="sortiq_logger", log_file="sortiq.log", level=logging.INFO):
    """
    Sets up JSON formatted logging for the application.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Don't add handlers if they already exist
    if not logger.handlers:
        # File handler for JSON logs
        file_handler = logging.FileHandler(log_file)
        
        # Console handler for local viewing
        console_handler = logging.StreamHandler()
        
        # Define JSON formatter
        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(levelname)s %(name)s %(module)s %(message)s'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger

# Create a singleton logger instance
logger = setup_logger()
