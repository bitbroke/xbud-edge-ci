import os
import subprocess
import logging
import time

logger = logging.getLogger(__name__)

class TTSWorker:
    def __init__(self, engine: str = "espeak", exec_path: str = None, voice: str = "hi", strict_real: bool = False):
        """
        engine: 'espeak', 'piper', or 'sapi5'
        exec_path: Path to the executable (not needed for sapi5).
        If exec_path is missing or engine is mock, runs in deterministic mock mode unless strict_real=True or engine is 'sapi5'.
        """
        self.engine = engine
        self.exec_path = exec_path
        self.voice = voice
        
        has_real = bool(exec_path and os.path.exists(exec_path))
        if engine == "sapi5":
            has_real = True # Native Windows SAPI
            
        if strict_real and not has_real:
            raise EnvironmentError(f"Model Unavailable: Missing {engine} binary in strict mode.")
            
        self.is_mock = not has_real
        
        if self.is_mock:
            logger.warning(f"TTSWorker running in MOCK mode. Provide exec_path to actual {engine} binary.")

    def synthesize(self, text: str, out_wav_path: str) -> bool:
        output_wav = out_wav_path
        if self.is_mock:
            # Deterministic mock TTS taking roughly 0.1s
            time.sleep(0.1)
            # Create a 16kHz dummy wav
            import numpy as np
            import soundfile as sf
            dummy = np.zeros((16000, 1), dtype=np.float32)
            sf.write(output_wav, dummy, 16000)
            return True
            
        if self.engine == "sapi5":
            try:
                script = """import sys, json, pyttsx3
data = json.loads(sys.stdin.read())
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.save_to_file(data['text'], data['output_wav'])
engine.runAndWait()
"""
                import sys
                import json
                result = subprocess.run(
                    [sys.executable, "-c", script], 
                    input=json.dumps({"text": text, "output_wav": output_wav}), 
                    text=True, 
                    capture_output=True, 
                    timeout=10.0
                )
                if result.returncode != 0:
                    logger.error(f"SAPI5 TTS Failed: {result.stderr}")
                    return False
                return True
            except Exception as e:
                logger.error(f"SAPI5 TTS Exception: {e}")
                return False

        args = []  
        try:
            if self.engine == "espeak":
                # espeak-ng -v hi -w out.wav "text"
                cmd = [self.exec_path, "-v", self.voice, "-w", out_wav_path, text]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5.0)
                if result.returncode != 0:
                    logger.error(f"eSpeak failed: {result.stderr}")
                    return False
                return True
                
            elif self.engine == "piper":
                # echo "text" | piper --model voice.onnx --output_file out.wav
                cmd = [self.exec_path, "--model", self.voice, "--output_file", out_wav_path]
                result = subprocess.run(cmd, input=text, capture_output=True, text=True, timeout=10.0)
                if result.returncode != 0:
                    logger.error(f"Piper failed: {result.stderr}")
                    return False
                return True
                
        except Exception as e:
            logger.error(f"TTS Exception: {e}")
            return False
        return False
