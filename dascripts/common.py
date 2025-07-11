from typing import Optional
import logging


def get_logger(name: str,
               level: int = logging.INFO,
               log_format: str = "%(asctime)s|%(name)s|%(levelname)s|%(message)s",
               filename: Optional[str] = None,
               filemode: str = "a",
) -> logging.Logger:
    """
    Returns a logger object with a full setup.

    Parameters:
        name (str): The name of the logger.
        level (int): Logging level (e.g., logging.DEBUG, logging.INFO, etc.)
        log_format (str): Log message format.
        filename (str): If provided, the logger will write logs to the specified file;
                        otherwise, logs will be output to the console (StreamHandler).

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create or get a logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding multiple handlers if the logger already has them
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create a formatter
    formatter = logging.Formatter(log_format)
    
    # Determine handler type: file or stream
    if filename is not None:
        handler = logging.FileHandler(filename, mode=filemode, encoding="utf-8")
    else:
        handler = logging.StreamHandler()
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
