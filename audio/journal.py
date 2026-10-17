import os
import hashlib
import soundfile as sf
import numpy as np
import logging

logger = logging.getLogger(__name__)

class StorageFullError(Exception):
    pass

class RecordingJournal:
    def __init__(self, out_dir: str, session_id: str, orig_rate: int, asr_rate: int, orig_channels: int, storage_limit_bytes: int):
        self.out_dir = out_dir
        self.session_id = session_id
        self.orig_rate = orig_rate
        self.asr_rate = asr_rate
        self.orig_channels = orig_channels
        self.storage_limit = storage_limit_bytes
        
        os.makedirs(out_dir, exist_ok=True)
        self.orig_path = os.path.join(out_dir, f"{session_id}_orig.wav")
        self.asr_path = os.path.join(out_dir, f"{session_id}_asr.wav")
        self.meta_path = os.path.join(out_dir, f"{session_id}.json")
        
        self.orig_file = sf.SoundFile(self.orig_path, mode='w', samplerate=orig_rate, channels=orig_channels)
        self.asr_file = sf.SoundFile(self.asr_path, mode='w', samplerate=asr_rate, channels=1)
        
        self.sha_orig = hashlib.sha256()
        self.sha_asr = hashlib.sha256()
        
        self.written_bytes = 0
        self.last_seq = -1
        self.dropouts = 0
        self.dropped_samples = 0
        self.is_finalized = False
        self.failed = False

    def write_frames(self, orig_frame, asr_data):
        if self.is_finalized or self.failed:
            return
            
        # Check dropouts
        if self.last_seq != -1 and orig_frame.seq != self.last_seq + 1:
            self.dropouts += 1
            # We don't know exactly how many samples dropped unless we use timestamps,
            # but we record the dropout event.
            self.dropped_samples += (orig_frame.seq - self.last_seq - 1) * orig_frame.sample_count
            
        self.last_seq = orig_frame.seq
        
        # Check storage limit
        # 16-bit PCM = 2 bytes per sample
        approx_bytes = (orig_frame.data.size * 2) + (asr_data.size * 2)
        if self.written_bytes + approx_bytes > self.storage_limit:
            self.failed = True
            self.finalize(status="storage-full")
            raise StorageFullError(f"Storage limit of {self.storage_limit} bytes reached.")

        try:
            self.orig_file.write(orig_frame.data)
            self.asr_file.write(asr_data)
            
            # Update checksums on the raw PCM data (in bytes)
            self.sha_orig.update(orig_frame.data.tobytes())
            self.sha_asr.update(asr_data.tobytes())
            
            self.written_bytes += approx_bytes
        except Exception as e:
            self.failed = True
            logger.error(f"Write failure: {e}")
            self.finalize(status="failed")
            raise

    def finalize(self, status="completed"):
        if self.is_finalized:
            return
            
        self.orig_file.close()
        self.asr_file.close()
        
        # Atomic meta write
        meta = {
            "session_id": self.session_id,
            "status": status,
            "orig_sha256": self.sha_orig.hexdigest(),
            "asr_sha256": self.sha_asr.hexdigest(),
            "dropouts": self.dropouts,
            "dropped_samples": self.dropped_samples,
            "written_bytes": self.written_bytes,
        }
        
        import json
        tmp_meta = self.meta_path + ".tmp"
        with open(tmp_meta, 'w') as f:
            json.dump(meta, f)
        os.replace(tmp_meta, self.meta_path)
        
        self.is_finalized = True

    def __del__(self):
        if not self.is_finalized:
            self.finalize(status="interrupted")
