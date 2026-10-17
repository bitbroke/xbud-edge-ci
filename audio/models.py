import time
from enum import Enum
from dataclasses import dataclass
import numpy as np

class CaptureState(Enum):
    STARTING = "starting"
    RECORDING = "recording"
    DISCONNECTED = "disconnected"
    STORAGE_FULL = "storage-full"
    FAILED = "failed"
    STOPPED = "stopped"

@dataclass
class AudioFrame:
    stream_id: str
    seq: int
    channels: int
    sample_rate: int
    sample_count: int
    capture_timestamp: float  # monotonic time
    data: np.ndarray          # The actual PCM audio block

@dataclass
class AudioConfig:
    device_id: int = None
    input_channels: int = 1
    input_rate: int = 48000
    asr_rate: int = 16000
    queue_size_frames: int = 200
    storage_limit_bytes: int = 100 * 1024 * 1024  # 100 MB default
    chunk_size: int = 1024
