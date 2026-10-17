import time
import queue
import threading
import sounddevice as sd
import soundfile as sf
import numpy as np
import logging

from .models import AudioFrame, CaptureState, AudioConfig
from .resampler import StatefulResampler
from .journal import RecordingJournal, StorageFullError

logger = logging.getLogger(__name__)

class DeviceManager:
    @staticmethod
    def list_devices():
        return sd.query_devices()
        
    @staticmethod
    def validate_format(device_id: int, channels: int, rate: int):
        try:
            sd.check_input_settings(device=device_id, channels=channels, samplerate=rate)
            return True
        except Exception as e:
            logger.error(f"Unsupported format: {e}")
            return False

class LiveAudioAdapter:
    def __init__(self, config: AudioConfig, frame_queue: queue.Queue):
        self.config = config
        self.frame_queue = frame_queue
        self.stream = None
        self.seq = 0
        self.stream_id = str(time.time())

    def _callback(self, indata, frames, time_info, status):
        # time_info.inputBufferAdcTime is monotonic
        capture_time = time_info.inputBufferAdcTime if time_info else time.monotonic()
        if status:
            logger.warning(f"Audio status: {status}")
            
        frame = AudioFrame(
            stream_id=self.stream_id,
            seq=self.seq,
            channels=self.config.input_channels,
            sample_rate=self.config.input_rate,
            sample_count=frames,
            capture_timestamp=capture_time,
            data=indata.copy()
        )
        self.seq += 1
        
        try:
            from app.conversation import tts_active_flag
            if tts_active_flag.is_set():
                # Dumping audio frame, TTS is active
                return
        except ImportError:
            pass

        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            logger.warning("Capture queue overflow! Dropping frame.")

    def start(self):
        if not DeviceManager.validate_format(self.config.device_id, self.config.input_channels, self.config.input_rate):
            raise ValueError("Unsupported audio format for selected device.")
            
        self.stream = sd.InputStream(
            device=self.config.device_id,
            channels=self.config.input_channels,
            samplerate=self.config.input_rate,
            blocksize=self.config.chunk_size,
            callback=self._callback,
            dtype='float32'
        )
        self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

class WavReplayAdapter:
    def __init__(self, filepath: str, config: AudioConfig, frame_queue: queue.Queue):
        self.filepath = filepath
        self.config = config
        self.frame_queue = frame_queue
        self.running = False
        self.thread = None
        self.seq = 0
        self.stream_id = f"wav_{time.time()}"

    def _replay_loop(self):
        try:
            with sf.SoundFile(self.filepath, 'r') as f:
                # Update config to match file if needed, or strictly pace it
                rate = f.samplerate
                channels = f.channels
                
                while self.running:
                    data = f.read(self.config.chunk_size, dtype='float32')
                    if len(data) == 0:
                        break
                        
                    frame = AudioFrame(
                        stream_id=self.stream_id,
                        seq=self.seq,
                        channels=channels,
                        sample_rate=rate,
                        sample_count=len(data),
                        capture_timestamp=time.monotonic(),
                        data=data
                    )
                    self.seq += 1
                    
                    try:
                        self.frame_queue.put(frame, timeout=1.0)
                    except queue.Full:
                        logger.warning("Replay queue overflow!")
                    
                    # Sleep to simulate real-time pacing
                    time.sleep(len(data) / rate)
        except Exception as e:
            logger.error(f"Replay error: {e}")
        finally:
            self.running = False

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._replay_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

class CaptureEngine:
    def __init__(self, config: AudioConfig, adapter_type="live", wav_path=None, out_dir="out"):
        self.config = config
        self.queue = queue.Queue(maxsize=config.queue_size_frames)
        self.state = CaptureState.STARTING
        self.session_id = f"session_{time.time():.3f}"
        
        if adapter_type == "live":
            self.adapter = LiveAudioAdapter(config, self.queue)
        else:
            self.adapter = WavReplayAdapter(wav_path, config, self.queue)
            
        self.resampler = StatefulResampler(config.input_rate, config.asr_rate, config.input_channels)
        self.journal = RecordingJournal(out_dir, self.session_id, config.input_rate, config.asr_rate, config.input_channels, config.storage_limit_bytes)
        
        self.running = False
        self.worker = None

    def _worker_loop(self):
        while self.running or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.1)
                
                # Resample to ASR rate
                asr_data = self.resampler.process(frame.data)
                
                # Journal write
                try:
                    self.journal.write_frames(frame, asr_data)
                except StorageFullError:
                    self.state = CaptureState.STORAGE_FULL
                    self.running = False
                    if self.adapter:
                        self.adapter.stop()
                    break
                except Exception:
                    self.state = CaptureState.FAILED
                    self.running = False
                    if self.adapter:
                        self.adapter.stop()
                    break
                    
            except queue.Empty:
                continue

        # Finalize journal when loop exits
        if self.state not in (CaptureState.FAILED, CaptureState.STORAGE_FULL):
            self.state = CaptureState.STOPPED
            self.journal.finalize(status=self.state.value)
        else:
            self.journal.finalize(status=self.state.value)

    def start(self):
        self.state = CaptureState.STARTING
        try:
            self.adapter.start()
            self.running = True
            
            # Resample explicitly to mono (1 channel) for ASR
            self.resampler = StatefulResampler(self.config.input_rate, self.config.asr_rate, channels=1)
            
            self.worker = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker.start()
            self.state = CaptureState.RECORDING
        except Exception as e:
            self.state = CaptureState.FAILED
            logger.error(f"Failed to start capture: {e}")
            raise

    def stop(self):
        self.running = False
        if self.adapter:
            self.adapter.stop()
        if self.worker and threading.current_thread() != self.worker:
            self.worker.join(timeout=2.0)
