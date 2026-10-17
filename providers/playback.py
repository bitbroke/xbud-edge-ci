import os
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class PlaybackAdapter(ABC):
    @abstractmethod
    def play(self, filepath: str):
        """Plays the audio file synchronously. Should raise an exception if playback fails."""
        pass

class WindowsWinsoundPlayback(PlaybackAdapter):
    def play(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Audio file not found: {filepath}")
            
        try:
            import winsound
            winsound.PlaySound(filepath, winsound.SND_FILENAME)
        except Exception as e:
            logger.error(f"Windows playback failed: {e}")
            raise RuntimeError(f"Playback failed: {e}")

class LinuxALSAPlayback(PlaybackAdapter):
    def __init__(self, device=None, timeout=30.0):
        self.device = device
        self.timeout = timeout

    def play(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Audio file not found: {filepath}")
            
        import subprocess
        try:
            # -q: quiet, -V mono: mono output (if needed)
            command = ["aplay", "-q"]
            if self.device is not None:
                command.extend(["-D", str(self.device)])
            command.append(filepath)
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"Playback timed out after {self.timeout} seconds") from e
        except subprocess.CalledProcessError as e:
            logger.error(f"ALSA playback failed: {e.stderr}")
            raise RuntimeError(f"Playback failed: {e.stderr}")
        except FileNotFoundError:
            raise RuntimeError("aplay not found. Is alsa-utils installed?")
