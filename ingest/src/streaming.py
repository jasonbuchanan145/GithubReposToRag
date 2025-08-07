from __future__ import annotations
import logging
from typing import Any, Dict


def stream_event(event: str, payload: Dict[str, Any] | None = None) -> None:
    logging.info("[stream] %s %s", event, payload or {})

def stream_step(message: str, **fields: Any) -> None:
    logging.info("[stream] %s %s", message, fields)
