"""Tiny logging helper.

A library should not configure the root logger on import, so we don't. Callers
(the CLI, notebooks) opt in via ``setup_logging``. Everything else just grabs a
module logger and stays quiet until someone attaches a handler.
"""

from __future__ import annotations

import logging
import os

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: int | str | None = None) -> None:
    """Configure root logging once, for entry points only.

    Honors ``DUPLEX_BOL_LOG_LEVEL`` so a notebook can crank up verbosity without
    editing code. Idempotent — calling it twice won't stack handlers.
    """
    if level is None:
        level = os.environ.get("DUPLEX_BOL_LOG_LEVEL", "INFO")
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format=_DEFAULT_FORMAT)
    else:
        root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
