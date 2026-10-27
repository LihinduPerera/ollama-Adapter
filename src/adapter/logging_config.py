from __future__ import annotations

import logging
import re

_BEARER_RE = re.compile(r"(?i)(authorization:\s*bearer\s+)\S+")


class _SecretFilter(logging.Filter):
    def __init__(self, secret: str | None) -> None:
        super().__init__()
        self._secret = secret

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        cleaned = _BEARER_RE.sub(r"\1[REDACTED]", message)
        if self._secret:
            cleaned = cleaned.replace(self._secret, "[REDACTED]")
        if cleaned != message:
            record.msg = cleaned
            record.args = ()
        return True


def configure_logging(level: str = "INFO", secret: str | None = None) -> None:
    """Configure process logging without ever logging request bodies."""
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    handler.addFilter(_SecretFilter(secret))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # httpx/httpcore can log URLs at INFO. Keep them quiet; the adapter logs
    # its own request summaries without prompts, headers, or tool arguments.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
