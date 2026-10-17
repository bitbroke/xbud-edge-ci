import os
import sys
import time
import json
import logging
import subprocess
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import multiprocessing as mp

logger = logging.getLogger(__name__)

try:
    import psutil
except ImportError:
    psutil = None

STRICT_HARDWARE_AUDIT = os.getenv("STRICT_HARDWARE_AUDIT", "1") == "1"


def run_measured(argv: list[str], timeout: float = 45.0) -> Tuple[subprocess.CompletedProcess, bool, Optional[int]]:
    """
    Spawns a subprocess, tracks peak RSS memory via psutil, and enforces a strict timeout.
    Returns (CompletedProcess, timed_out_boolean, peak_memory_bytes).
    """
    start_time = time.monotonic()
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    peak_bytes = None
    timed_out = False

    p = None
    if psutil:
        try:
            p = psutil.Process(proc.pid)
        except Exception:
            p = None

    while True:
        if proc.poll() is not None:
            break
        if p and p.is_running():
            try:
                mem = p.memory_info().rss
                if peak_bytes is None or mem > peak_bytes:
                    peak_bytes = mem
            except Exception:
                pass
        if (time.monotonic() - start_time) > timeout:
            timed_out = True
            try:
                proc.kill()
            except Exception:
                pass
            break
        time.sleep(0.02)

    stdout, stderr = proc.communicate()
    completed = subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)
    return completed, timed_out, peak_bytes


class BaseASR(ABC):
    """Abstract base class for all ASR engine adapters."""
    def __init__(self, language: str = "en", model_id: str = "base-asr"):
        self.language = language
        self.model_id = model_id

    @abstractmethod
    def transcribe(self, wav_path: str, timeout: float = 10.0, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe a WAV file and return the standardized ASR contract:
        {
            "text": str,
            "language": str,
            "model_id": str,
            "duration_s": float,
            "error": str | None,
            "timeout": bool,
            "peak_memory_bytes": int | None,
        }
        """
        pass


class MockASR(BaseASR):
    """Deterministic mock ASR adapter for unit testing and offline fallbacks."""
    def __init__(self, language: str = "en", model_id: str = "mock-asr", mock_text: str = None):
        super().__init__(language=language, model_id=model_id)
        self.mock_text = mock_text or "कल सुबह नौ बजे रिया को कॉल करने की याद दिलाना"

    def transcribe(self, wav_path: str, timeout: float = 10.0, language: Optional[str] = None) -> Dict[str, Any]:
        t0 = time.monotonic()
        run_lang = language or self.language
        time.sleep(0.05)
        duration = time.monotonic() - t0
        text = self.mock_text if run_lang in ["hi", "auto"] else "remind me to call John tomorrow at 5 PM"
        if duration > timeout:
            return {
                "text": "",
                "language": run_lang,
                "model_id": self.model_id,
                "duration_s": duration,
                "duration": duration,
                "error": "Timeout",
                "timeout": True,
                "peak_memory_bytes": 1048576,
                "peak_process_memory_bytes": 1048576,
                "segments": [],
            }
        return {
            "text": text,
            "language": run_lang,
            "model_id": self.model_id,
            "duration_s": duration,
            "duration": duration,
            "error": None,
            "timeout": False,
            "peak_memory_bytes": 1048576,
            "peak_process_memory_bytes": 1048576,
            "segments": [{"start": 0.0, "end": 2.0, "text": text}],
        }


class WhisperCppASR(BaseASR):
    """Subprocess-based adapter executing whisper.cpp (e.g. main.exe / whisper-cli)."""
    def __init__(self, cli_path: str, model_path: str, language: str = "en", strict_real: bool = False):
        super().__init__(language=language, model_id=f"whisper-cpp:{os.path.basename(model_path) if model_path else 'none'}")
        self.cli_path = os.path.abspath(cli_path) if cli_path else None
        self.model_path = os.path.abspath(model_path) if model_path else None
        self.strict_real = strict_real

        cli_exists = self.cli_path and os.path.exists(self.cli_path)
        model_exists = self.model_path and os.path.exists(self.model_path)

        if strict_real:
            if not cli_exists:
                raise FileNotFoundError(f"Whisper CLI executable not found: {self.cli_path}")
            if not model_exists:
                raise FileNotFoundError(f"Whisper model file not found: {self.model_path}")

        self.is_mock = not (cli_exists and model_exists)

    def transcribe(self, wav_path: str, timeout: float = 45.0, language: Optional[str] = None) -> Dict[str, Any]:
        t0 = time.monotonic()
        run_language = language or self.language

        if self.is_mock:
            mock = MockASR(language=run_language, model_id=self.model_id)
            return mock.transcribe(wav_path, timeout=timeout)

        wav_path = os.path.abspath(wav_path)
        output_base = os.path.splitext(wav_path)[0]
        output_json = output_base + ".json"

        # Sidecar isolation: remove any stale output file before execution
        if os.path.exists(output_json):
            try:
                os.remove(output_json)
            except Exception:
                pass

        argv = [
            self.cli_path,
            "-m", self.model_path,
            "-f", wav_path,
            "-oj",
            "-of", output_base,
            "-l", run_language,
            "-t", "4",
        ]

        proc, timed_out, peak_bytes = run_measured(argv, timeout=timeout)
        elapsed = time.monotonic() - t0

        if timed_out:
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
                "raw_stdout": proc.stdout,
                "raw_stderr": proc.stderr,
                "argv": argv,
                "segments": [],
            }

        if not os.path.exists(output_json):
            return {
                "text": "",
                "language": run_language,
                "model_id": self.model_id,
                "duration_s": elapsed,
                "duration": elapsed,
                "error": "missing_output_json",
                "timeout": False,
                "peak_memory_bytes": peak_bytes,
                "peak_process_memory_bytes": peak_bytes,
                "raw_stdout": proc.stdout,
                "raw_stderr": proc.stderr,
                "argv": argv,
                "segments": [],
            }

        try:
            with open(output_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Remove json sidecar file to keep dataset pure
            try:
                os.remove(output_json)
            except Exception:
                pass

            transcription_items = data.get("transcription", [])
            text = " ".join(item.get("text", "").strip() for item in transcription_items).strip()
            segments = [
                {
                    "start": item.get("timestamps", {}).get("from", 0.0),
                    "end": item.get("timestamps", {}).get("to", 0.0),
                    "text": item.get("text", "").strip(),
                }
                for item in transcription_items
            ]

            return {
                "text": text,
                "language": run_language,
                "model_id": self.model_id,
                "duration_s": elapsed,
                "duration": elapsed,
                "error": None,
                "timeout": False,
                "peak_memory_bytes": peak_bytes,
                "peak_process_memory_bytes": peak_bytes,
                "raw_stdout": proc.stdout,
                "raw_stderr": proc.stderr,
                "argv": argv,
                "segments": segments,
            }
        except Exception as e:
            return {
                "text": "",
                "language": run_language,
                "model_id": self.model_id,
                "duration_s": elapsed,
                "duration": elapsed,
                "error": f"JSON parsing error: {e}",
                "timeout": False,
                "peak_memory_bytes": peak_bytes,
                "peak_process_memory_bytes": peak_bytes,
                "raw_stdout": proc.stdout,
                "raw_stderr": proc.stderr,
                "argv": argv,
                "segments": [],
            }


# Worker target function for managed Sherpa process isolation
def _sherpa_inference_worker(req_queue: mp.Queue, resp_queue: mp.Queue, model_dir: str, num_threads: int):
    try:
        import sherpa_onnx
        import soundfile as sf
    except Exception as e:
        resp_queue.put({"status": "error", "error": f"Failed importing dependencies: {e}"})
        return

    encoder = os.path.join(model_dir, "encoder-epoch-99-avg-1.onnx")
    decoder = os.path.join(model_dir, "decoder-epoch-99-avg-1.onnx")
    joiner = os.path.join(model_dir, "joiner-epoch-99-avg-1.onnx")

    if not os.path.exists(encoder):
        encoder = os.path.join(model_dir, "encoder-epoch-99-avg-1.int8.onnx")
    if not os.path.exists(decoder):
        decoder = os.path.join(model_dir, "decoder-epoch-99-avg-1.int8.onnx")
    if not os.path.exists(joiner):
        joiner = os.path.join(model_dir, "joiner-epoch-99-avg-1.int8.onnx")
    tokens = os.path.join(model_dir, "tokens.txt")

    try:
        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            num_threads=num_threads,
            sample_rate=16000,
            feature_dim=80,
            debug=False,
        )
        resp_queue.put({"status": "ready"})
    except Exception as e:
        resp_queue.put({"status": "error", "error": f"Failed initializing recognizer: {e}"})
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
            audio_duration = float(len(audio)) / float(sample_rate)

            resp_queue.put({"status": "ok", "text": text, "audio_duration": audio_duration, "error": None})
        except Exception as e:
            resp_queue.put({"status": "error", "text": "", "error": str(e), "audio_duration": 0.0})


class SherpaOnnxASR(BaseASR):
    """In-memory streaming Zipformer ASR using sherpa-onnx with managed process isolation for timeouts."""
    def __init__(self, model_dir: str = "models/sherpa-onnx-zipformer-en-2023-06-26",
                 language: str = "en", strict_real: bool = False, num_threads: int = 1):
        super().__init__(language=language, model_id="sherpa-onnx-zipformer")
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

            # Validate required files
            enc = os.path.exists(os.path.join(self.model_dir, "encoder-epoch-99-avg-1.onnx")) or \
                  os.path.exists(os.path.join(self.model_dir, "encoder-epoch-99-avg-1.int8.onnx"))
            dec = os.path.exists(os.path.join(self.model_dir, "decoder-epoch-99-avg-1.onnx")) or \
                  os.path.exists(os.path.join(self.model_dir, "decoder-epoch-99-avg-1.int8.onnx"))
            joi = os.path.exists(os.path.join(self.model_dir, "joiner-epoch-99-avg-1.onnx")) or \
                  os.path.exists(os.path.join(self.model_dir, "joiner-epoch-99-avg-1.int8.onnx"))
            tok = os.path.exists(os.path.join(self.model_dir, "tokens.txt"))

            if not (enc and dec and joi and tok):
                raise EnvironmentError(f"Required Sherpa-ONNX model files missing in {self.model_dir}")

        self.is_mock = not (has_model_dir and has_sherpa)

    def _ensure_worker(self):
        if self._proc is not None and self._proc.is_alive():
            return True

        self._req_queue = self._ctx.Queue()
        self._resp_queue = self._ctx.Queue()
        self._proc = self._ctx.Process(
            target=_sherpa_inference_worker,
            args=(self._req_queue, self._resp_queue, self.model_dir, self.num_threads),
            daemon=True
        )
        self._proc.start()

        # Wait for worker ready signal
        try:
            init_resp = self._resp_queue.get(timeout=10.0)
            if init_resp.get("status") != "ready":
                if self.strict_real:
                    raise RuntimeError(f"Sherpa worker failed to initialize: {init_resp.get('error')}")
                self._kill_worker()
                self.is_mock = True
                return False
            return True
        except Exception as e:
            if self.strict_real:
                raise RuntimeError(f"Sherpa worker startup timed out or failed: {e}")
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
            # Enforce hard timeout: kill worker and respawn on subsequent calls
            logger.warning(f"Sherpa-ONNX inference exceeded timeout ({timeout}s). Killing worker process.")
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


class ASRWorker:
    """
    Unified ASR facade and factory ensuring one stable contract across all callers.
    """
    def __init__(self, cli_path: str = None, model_path: str = None, language: str = "en",
                 strict_real: bool = False, engine: str = None, model_dir: str = None):
        self.language = language
        self.strict_real = True if STRICT_HARDWARE_AUDIT else strict_real

        # Resolve engine type
        if engine:
            self.engine = engine.lower()
        elif cli_path or (model_path and model_path.endswith(".bin")):
            self.engine = "whisper"
        else:
            self.engine = "sherpa"

        self.model_dir = model_dir or "models/sherpa-onnx-zipformer-en-2023-06-26"
        self.cli_path = cli_path
        self.model_path = model_path

        if self.engine == "mock":
            self.adapter = MockASR(language=language)
            self.is_mock = True
        elif self.engine in ["indic_conformer", "indic", "ai4bharat"]:
            from providers.indic_conformer import IndicConformerASR
            self.adapter = IndicConformerASR(
                model_dir=self.model_dir if "indic" in str(self.model_dir) else None,
                language=language,
                strict_real=strict_real
            )
            self.is_mock = self.adapter.is_mock
        elif self.engine in ["sense_voice", "sensevoice"]:
            from providers.sense_voice import SenseVoiceASR
            self.adapter = SenseVoiceASR(
                model_dir=self.model_dir if "sense" in str(self.model_dir) else "models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
                language=language,
                strict_real=strict_real
            )
            self.is_mock = self.adapter.is_mock
        elif self.engine == "whisper":
            self.adapter = WhisperCppASR(
                cli_path=cli_path,
                model_path=model_path,
                language=language,
                strict_real=strict_real
            )
            self.is_mock = self.adapter.is_mock
        else:
            # Default: SherpaOnnxASR
            self.adapter = SherpaOnnxASR(
                model_dir=self.model_dir,
                language=language,
                strict_real=strict_real
            )
            self.is_mock = self.adapter.is_mock

    def transcribe(self, wav_path: str, timeout: float = 10.0, language: str = None) -> Dict[str, Any]:
        return self.adapter.transcribe(wav_path=wav_path, timeout=timeout, language=language)
