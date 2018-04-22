
import logging
import os

def setup_logger(name, log_file, level=logging.INFO):

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    f = open(log_file, "w+")
    f.close()
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)

    return logger
