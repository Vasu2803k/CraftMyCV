import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging(log_level: str, log_dir: str, log_file_name: str, logger_name: str) -> logging.Logger:
    """Set up logging configuration with console and file output."""
    try:
        # Ensure the log directory exists
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Define log file path
        log_file = os.path.join(log_dir, f"{log_file_name}.log")

        # Get the logger specific to the current module (file)
        logger = logging.getLogger(logger_name)

        # Validate log level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if log_level.upper() not in valid_log_levels:
            logger.warning(f"Invalid log level '{log_level}', defaulting to 'INFO'")
            log_level = 'INFO'

        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        # Log the setup information
        logger.info(f"Setting up Logs for '{log_file}'")

        # Check if a file handler already exists
        file_handler = next((handler for handler in logger.handlers if isinstance(handler, RotatingFileHandler)), None)

        # Check if a console handler already exists
        console_handler = next((handler for handler in logger.handlers if isinstance(handler, logging.StreamHandler)), None)

        # Create and configure file handler if it doesn't exist
        if file_handler is None:
            file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5, mode='a')
            file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # Create and configure console handler if it doesn't exist
        if console_handler is None:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, log_level.upper(), logging.ERROR))
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        logger.info(f"Logging initialized for Log file: '{log_file}'")

        return logger
    except Exception as e:
        # Log the error to a fallback log file or console
        logging.basicConfig(level=logging.ERROR)
        logging.error(f"Error setting up logging: {e}")
        raise