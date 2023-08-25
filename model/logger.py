import logging
log_format = '%(asctime)s %(levelname)s %(name)s:: %(message)s'


def create_logger(logger: logging.Logger, logging_level: int = logging.INFO,
                  log_format: str = log_format):

    logger.setLevel(level=logging_level)
    formatter = logging.Formatter(log_format)

    # add file handler
    file_handler = logging.FileHandler('./logs.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # add stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
