import os
import time
import logging
from typing import Dict, Any, Optional
import multiprocessing as mp

logger = logging.getLogger(__name__)

try:
    import psutil
except ImportError:
    psutil = None

from providers.asr_worker import BaseASR, MockASR

def _sense_voice_inference_worker(req_queue: mp.Queue, resp_queue: mp.Queue, model_dir: str, num_threads: int):
    try:
        import sherpa_onnx
        import soundfile as sf
    except Exception as e:
        resp_queue.put({"status": "error", "error": f"Failed importing dependencies: {e}"})
        return

    model = os.path.join(model_dir, "model.onnx")
    if not os.path.exists(model):
        model = os.path.join(model_dir, "model.int8.onnx")
    tokens = os.path.join(model_dir, "tokens.txt")

    if not os.path.exists(model) or not os.path.exists(tokens):
        resp_queue.put({"status": "error", "error": f"Missing SenseVoice model files in {model_dir}"})
        return

    try:
        recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model,
            tokens=tokens,
            num_threads=num_threads,
            sample_rate=16000,
            feature_dim=80,
            debug=False,
            use_itn=True
        )
        resp_queue.put({"status": "ready"})
    except Exception as e:
        resp_queue.put({"status": "error", "error": f"Failed initializing SenseVoice recognizer: {e}"})
        return

    while True:
        try:
            req = req_queue.get()
            if req is None or req.get("action") == "stop":
                break

            wav_path = req["wav_path"]
            audio, sample_rate = sf.read(wav_path, dtype="float32")
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate, audio)
            recognizer.decode_stream(stream)
            text = stream.result.text.strip()
            # SenseVoice often outputs language tags like <|en|> <|nospeech|><|withitn|> at the beginning. 
            # We clean them up via simple string replace if they exist, or they might be handled natively by sherpa-onnx.
            if text.startswith("<|"):
                # E.g. <|en|><|NEUTRAL|><|Speech|><|woitn|>
                import re
                text = re.sub(r'<\|.*?\|>', '', text).strip()

            audio_duration = float(len(audio)) / float(sample_rate)

            resp_queue.put({"status": "ok", "text": text, "audio_duration": audio_duration, "error": None})
        except Exception as e:
            resp_queue.put({"status": "error", "text": "", "error": str(e), "audio_duration": 0.0})


class SenseVoiceASR(BaseASR):
    """In-memory SenseVoice ASR using sherpa-onnx with managed process isolation for timeouts."""
    def __init__(self, model_dir: str = "models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
                 language: str = "auto", strict_real: bool = False, num_threads: int = 1):
        super().__init__(language=language, model_id="sensevoice-small")
        self.model_dir = os.path.abspath(model_dir) if model_dir else None
        self.strict_real = strict_real
        self.num_threads = num_threads
        self._proc = None
        self._req_queue = None
        self._resp_queue = None
        self._ctx = mp.get_context("spawn")

        has_model_dir = self.model_dir and os.path.isdir(self.model_dir)
        has_sherpa = False
        try:
            import sherpa_onnx
            has_sherpa = True
        except ImportError:
            has_sherpa = False

        if strict_real:
            if not has_model_dir:
                raise EnvironmentError(f"Model directory missing: {self.model_dir}")
            if not has_sherpa:
                raise EnvironmentError("sherpa-onnx package is not installed.")

            mod = os.path.exists(os.path.join(self.model_dir, "model.onnx")) or \
                  os.path.exists(os.path.join(self.model_dir, "model.int8.onnx"))
            tok = os.path.exists(os.path.join(self.model_dir, "tokens.txt"))

            if not (mod and tok):
                raise EnvironmentError(f"Required SenseVoice model files missing in {self.model_dir}")

        self.is_mock = not (has_model_dir and has_sherpa)

    def _ensure_worker(self):
        if self._proc is not None and self._proc.is_alive():
            return True

        self._req_queue = self._ctx.Queue()
        self._resp_queue = self._ctx.Queue()
        self._proc = self._ctx.Process(
            target=_sense_voice_inference_worker,
            args=(self._req_queue, self._resp_queue, self.model_dir, self.num_threads),
            daemon=True
        )
        self._proc.start()

        try:
            init_resp = self._resp_queue.get(timeout=10.0)
            if init_resp.get("status") != "ready":
                if self.strict_real:
                    raise RuntimeError(f"SenseVoice worker failed to initialize: {init_resp.get('error')}")
                self._kill_worker()
                self.is_mock = True
                return False
            return True
        except Exception as e:
            if self.strict_real:
                raise RuntimeError(f"SenseVoice worker startup timed out or failed: {e}")
            self._kill_worker()
            self.is_mock = True
            return False

    def _kill_worker(self):
        if self._proc is not None:
            try:
                if self._proc.is_alive():
                    self._proc.terminate()
                    self._proc.join(timeout=1.0)
                    if self._proc.is_alive():
                        self._proc.kill()
            except Exception:
                pass
            self._proc = None
            self._req_queue = None
            self._resp_queue = None

    def transcribe(self, wav_path: str, timeout: float = 10.0, language: Optional[str] = None) -> Dict[str, Any]:
        t0 = time.monotonic()
        run_language = language or self.language

        if self.is_mock:
            mock = MockASR(language=run_language, model_id=self.model_id)
            return mock.transcribe(wav_path, timeout=timeout)

        if not self._ensure_worker():
            mock = MockASR(language=run_language, model_id=self.model_id)
            return mock.transcribe(wav_path, timeout=timeout)

        wav_path = os.path.abspath(wav_path)
        self._req_queue.put({"wav_path": wav_path, "language": run_language})

        peak_bytes = None
        p = None
        if psutil and self._proc:
            try:
                p = psutil.Process(self._proc.pid)
            except Exception:
                p = None

        timed_out = False
        res = None

        start_wait = time.monotonic()
        while (time.monotonic() - start_wait) < timeout:
            if p and p.is_running():
                try:
                    mem = p.memory_info().rss
                    if peak_bytes is None or mem > peak_bytes:
                        peak_bytes = mem
                except Exception:
                    pass
            try:
                res = self._resp_queue.get_nowait()
                break
            except Exception:
                time.sleep(0.02)

        elapsed = time.monotonic() - t0

        if res is None:
            logger.warning(f"SenseVoice inference exceeded timeout ({timeout}s). Killing worker process.")
            self._kill_worker()
            return {
                "text": "",
                "language": run_language,
                "model_id": self.model_id,
                "duration_s": elapsed,
                "duration": elapsed,
                "error": "Timeout",
                "timeout": True,
                "peak_memory_bytes": peak_bytes,
                "peak_process_memory_bytes": peak_bytes,
                "segments": [],
            }

        text = res.get("text", "")
        error = res.get("error")
        audio_dur = res.get("audio_duration", 0.0)

        return {
            "text": text,
            "language": run_language,
            "model_id": self.model_id,
            "duration_s": elapsed,
            "duration": elapsed,
            "error": error,
            "timeout": False,
            "peak_memory_bytes": peak_bytes,
            "peak_process_memory_bytes": peak_bytes,
            "segments": [{"start": 0.0, "end": audio_dur, "text": text}] if text else [],
        }

    def __del__(self):
        self._kill_worker()
