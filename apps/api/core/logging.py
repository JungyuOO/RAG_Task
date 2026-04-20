from __future__ import annotations

import logging
from logging.config import dictConfig


def configure_logging() -> None:
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s %(levelname)s [%(name)s] %(message)s",
                    "datefmt": "%H:%M:%S",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "level": "INFO",
                }
            },
            "root": {
                "handlers": ["console"],
                "level": "INFO",
            },
            "loggers": {
                "uvicorn.access": {
                    "handlers": ["console"],
                    "level": "WARNING",
                    "propagate": False,
                },
                "httpx": {
                    "handlers": ["console"],
                    "level": "WARNING",
                    "propagate": False,
                },
                "rag.chat": {
                    "handlers": ["console"],
                    "level": "INFO",
                    "propagate": False,
                },
                "rag.llm": {
                    "handlers": ["console"],
                    "level": "INFO",
                    "propagate": False,
                },
                "rag.retrieval": {
                    "handlers": ["console"],
                    "level": "INFO",
                    "propagate": False,
                },
                "rag.batch_index": {
                    "handlers": ["console"],
                    "level": "INFO",
                    "propagate": False,
                },
                "rag.new_index": {
                    "handlers": ["console"],
                    "level": "INFO",
                    "propagate": False,
                },
                "rag.index_writer": {
                    "handlers": ["console"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    logging.getLogger("rag.batch_index").info("batch index logger configured")
