import time
import numpy as np

class VirtualMicrophoneStream:
    """Simulates real-time 16kHz mono hardware mic stream from memory buffer."""
    def __init__(self, pcm_data: np.ndarray, sample_rate: int = 16000, chunk_size: int = 512):
        self.pcm_data = pcm_data
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.cursor = 0
        self.chunk_duration = chunk_size / sample_rate

    def read_chunk(self, tts_active_flag):
        """Simulates ALSA read with half-duplex mute support."""
        time.sleep(self.chunk_duration) # Enforce hardware realtime clock pace
        
        if tts_active_flag.is_set():
            # Drop frame: assistant is speaking, mic blinded
            return np.zeros(self.chunk_size, dtype=np.float32)

        if self.cursor >= len(self.pcm_data):
            return None # Stream finished

        end = self.cursor + self.chunk_size
        chunk = self.pcm_data[self.cursor:end]
        if len(chunk) < self.chunk_size:
            chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)))
        self.cursor = end
        return chunk
