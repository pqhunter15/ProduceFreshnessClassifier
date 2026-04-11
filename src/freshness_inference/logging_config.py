"""Logging configuration for freshness inference."""
import logging
import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
        "file": {
            "level": "DEBUG",
            "class": "logging.FileHandler",
            "filename": "freshness_inference.log",
            "formatter": "detailed",
        },
    },
    "loggers": {
        "src.freshness_inference": {
            "handlers": ["default", "file"],
            "level": "DEBUG",
            "propagate": False,
        }
    },
}


def setup_logging():
    """Initialize logging configuration."""
    logging.config.dictConfig(LOGGING_CONFIG)
    return logging.getLogger("src.freshness_inference")