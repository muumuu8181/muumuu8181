import logging
import json
from datetime import datetime
from typing import Any, Dict
from uuid import uuid4

class StructuredLogger:
    def __init__(self):
        self.logger = logging.getLogger("api")
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def log(self, level: str, event: str, correlation_id: str = None, **kwargs):
        if correlation_id is None:
            correlation_id = str(uuid4())
            
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "event": event,
            "correlation_id": correlation_id,
            **kwargs
        }
        self.logger.log(
            getattr(logging, level.upper()),
            json.dumps(log_data, ensure_ascii=False)
        )

logger = StructuredLogger()
