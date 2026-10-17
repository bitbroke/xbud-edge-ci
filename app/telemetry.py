import logging
import json
import uuid
import re
import threading
from datetime import datetime

class TelemetryManager:
    _instance = None
    
    @classmethod
    def get(cls):
        if not cls._instance:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.metrics = {
            "asr_failures": 0,
            "tts_failures": 0,
            "asr_timeouts": 0,
            "total_intents": 0,
            "db_errors": 0
        }
        self.latency_samples = []
        self.lock = threading.Lock()
        
        self.logger = logging.getLogger("xbud.telemetry")
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler("data/xbud_telemetry.log")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(fh)

    def redact_pii(self, text: str) -> str:
        # Simple redaction for transcripts (remove digits, specific names)
        # In a real system, this might use NLP NER to mask entities.
        redacted = re.sub(r'\b\d{10,12}\b', '[PHONE]', text)
        redacted = re.sub(r'\b\d+\b', '[NUM]', redacted)
        return redacted

    def log_event(self, session_id: str, event_type: str, details: dict):
        # Redact any transcript fields
        safe_details = dict(details)
        if "transcript" in safe_details:
            safe_details["transcript"] = self.redact_pii(safe_details["transcript"])
            
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "session_id": session_id,
            "event": event_type,
            "details": safe_details
        }
        self.logger.info(json.dumps(log_entry))
        
    def record_metric(self, key: str, value=1):
        with self.lock:
            if key in self.metrics:
                self.metrics[key] += value
            else:
                self.metrics[key] = value

    def record_latency(self, component: str, ms: float):
        with self.lock:
            self.latency_samples.append({
                "component": component,
                "latency_ms": ms,
                "timestamp": datetime.utcnow().isoformat()
            })
            # keep bounded
            if len(self.latency_samples) > 1000:
                self.latency_samples.pop(0)

def get_session_id():
    return uuid.uuid4().hex
