"""Non-blocking logging setup.

All log records go through a ``QueueHandler`` so that ``logger.info(...)`` in any thread
(including the 50 Hz motion streamer) only enqueues; the actual formatting and I/O run
on a ``QueueListener`` background thread.
"""

from __future__ import annotations

import logging
import logging.handlers
import queue
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-7s %(threadName)-12s %(name)s: %(message)s"
DATE_FORMAT = "%H:%M:%S"


def setup_logging(
    level: int = logging.INFO, log_file: Path | None = None
) -> logging.handlers.QueueListener:
    """Install a queue-based root handler. Call ``listener.stop()`` at shutdown."""
    log_queue: queue.Queue[logging.LogRecord] = queue.Queue(-1)
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

    handlers: list[logging.Handler] = []
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    handlers.append(console)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        handlers.append(fh)

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(logging.handlers.QueueHandler(log_queue))
    root.setLevel(level)

    listener = logging.handlers.QueueListener(log_queue, *handlers, respect_handler_level=False)
    listener.start()
    return listener
