import os
import time
import queue
import logging
import threading
import soundfile as sf
import uuid

from audio.capture import LiveAudioAdapter, DeviceManager
from audio.models import AudioConfig

logger = logging.getLogger(__name__)

class HardwareRecorder:
    def __init__(self, config: AudioConfig, max_duration: int = 30, temp_dir: str = "."):
        self.config = config
        self.max_duration = max_duration
        self.temp_dir = temp_dir
        self.frame_queue = queue.Queue(maxsize=100) # bounded buffer
        self.adapter = LiveAudioAdapter(config, self.frame_queue)
        self.is_recording = False
        self.thread = None
        self.temp_wav_path = None
        self.start_time = None
        self.max_amp = 0.0
        self.file_returned = False
        self.on_limit_reached = None

    def start_recording(self):
        if self.is_recording:
            return
            
        self.temp_wav_path = os.path.join(self.temp_dir, f"temp_rec_{uuid.uuid4().hex[:8]}.wav")
        
        # Clear any stale frames
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
                
        try:
            self.adapter.start()
        except Exception as e:
            logger.error(f"Failed to start audio adapter: {e}")
            return
            
        self.is_recording = True
        self.file_returned = False
        self.max_amp = 0.0
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.thread.start()

    def stop_recording(self) -> str:
        if self.file_returned or not self.temp_wav_path:
            return None
            
        self.is_recording = False
        self.adapter.stop()
        
        if self.thread:
            self.thread.join(timeout=2.0)
            
        self.file_returned = True
            
        # Check silence or empty recording
        if not os.path.exists(self.temp_wav_path) or os.path.getsize(self.temp_wav_path) < 100 or self.max_amp < 0.01:
            logger.warning(f"Recording was empty, too short, or silent (max amp {self.max_amp:.4f}).")
            if os.path.exists(self.temp_wav_path):
                os.remove(self.temp_wav_path)
            return None
            
        return self.temp_wav_path

    def shutdown(self):
        self.is_recording = False
        try:
            self.adapter.stop()
        except Exception:
            pass
        if self.thread:
            self.thread.join(timeout=2.0)

    def _writer_loop(self):
        try:
            with sf.SoundFile(self.temp_wav_path, mode='w', samplerate=self.config.input_rate, channels=self.config.input_channels) as f:
                while self.is_recording:
                    if time.time() - self.start_time > self.max_duration:
                        logger.info("Max recording duration reached. Auto-stopping adapter.")
                        self.adapter.stop()
                        self.is_recording = False
                        if self.on_limit_reached:
                            self.on_limit_reached()
                        break
                        
                    try:
                        frame = self.frame_queue.get(timeout=0.1)
                        import numpy as np
                        amp = np.max(np.abs(frame.data))
                        if amp > self.max_amp:
                            self.max_amp = amp
                        f.write(frame.data)
                    except queue.Empty:
                        pass
        except Exception as e:
            logger.error(f"Error during audio recording: {e}", exc_info=True)
            self.is_recording = False

class ALSARecorder:
    """Pure ALSA implementation for Linux CM5 boards without PortAudio/sounddevice dependencies."""
    def __init__(self, config: AudioConfig, max_duration: int = 30, temp_dir: str = "."):
        self.config = config
        self.max_duration = max_duration
        self.temp_dir = temp_dir
        self.is_recording = False
        self.process = None
        self.temp_wav_path = None
        self.start_time = None
        
    def start_recording(self):
        if self.is_recording:
            return
            
        import subprocess
        
        self.temp_wav_path = os.path.join(self.temp_dir, f"temp_rec_{uuid.uuid4().hex[:8]}.wav")
        # Use arecord for pure ALSA capture.
        # -D hw:1,0 (default or config), -f S16_LE (16-bit), -c channels, -r rate, -d duration
        device_id = self.config.device_id
        device = f"hw:{device_id}" if isinstance(device_id, int) else (device_id or "default")
        
        cmd = [
            "arecord",
            "-D", str(device),
            "-f", "S16_LE",
            "-c", str(self.config.input_channels),
            "-r", str(self.config.input_rate),
            "-d", str(self.max_duration),
            "-q", # quiet
            self.temp_wav_path
        ]
        
        try:
            self.process = subprocess.Popen(cmd)
            self.is_recording = True
            self.start_time = time.time()
        except FileNotFoundError:
            logger.error("arecord not found! Please install alsa-utils.")
        except Exception as e:
            logger.error(f"ALSA recorder failed to start: {e}")

    def stop_recording(self) -> str:
        if not self.is_recording:
            return None
            
        self.is_recording = False
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except Exception:
                self.process.kill()
                self.process.wait(timeout=2.0)
            self.process = None
            
        if not os.path.exists(self.temp_wav_path) or os.path.getsize(self.temp_wav_path) < 100:
            logger.warning("ALSA recording was empty or failed.")
            if os.path.exists(self.temp_wav_path):
                os.remove(self.temp_wav_path)
            return None
            
        # Optional: Implement purely file-based silence detection here if needed (e.g. read WAV chunk)
        # We assume the external pipeline can handle silence or VAD if needed.
        return self.temp_wav_path
        
    def shutdown(self):
        self.stop_recording()
