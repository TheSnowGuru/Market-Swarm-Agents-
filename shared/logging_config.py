import logging
import os
from config import LOGGING_CONFIG

def setup_logging():
    """
    Configure logging based on the configuration in config.py
    """
    log_dir = os.path.dirname(LOGGING_CONFIG['file_path'])
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG['level']),
        format=LOGGING_CONFIG['format'],
        filename=LOGGING_CONFIG['file_path'],
        filemode='a'
    )

    return logging.getLogger(__name__)
