import os
import sys
import time
import json
import logging
import subprocess
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import multiprocessing as mp
import queue
import threading
import uuid
import math

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
            raw_text = " ".join(item.get("text", "").strip() for item in transcription_items).strip()
            # Remove only this exact engine control marker, not bracketed user
            # content, words such as 'blank audio', or other annotations.
            def clean_text(value):
                return " ".join(value.replace("[BLANK_AUDIO]", " ").split())
            text = clean_text(raw_text)
            segments = [
                {
                    "start": item.get("timestamps", {}).get("from", 0.0),
                    "end": item.get("timestamps", {}).get("to", 0.0),
                    "text": clean_text(item.get("text", "")),
                    "raw_text": item.get("text", "").strip(),
                }
                for item in transcription_items if clean_text(item.get("text", ""))
            ]

            return {
                "text": text,
                "raw_text": raw_text,
                "non_speech_markers_removed": raw_text.count("[BLANK_AUDIO]"),
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

            resp_queue.put({"status": "ok", "request_id": req["request_id"], "text": text, "audio_duration": audio_duration, "error": None})
        except Exception as e:
            resp_queue.put({"status": "error", "request_id": req.get("request_id"), "text": "", "error": str(e), "audio_duration": 0.0})


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
        self._ready = False
        self._lifecycle_lock = threading.RLock()
        self.startup_timeout_s = 10.0

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

    def _ensure_worker(self, deadline=None):
        """A live process is usable only after its ready handshake."""
        with self._lifecycle_lock:
            if self._proc is not None and self._proc.is_alive() and self._ready:
                return True
            self._kill_worker()
            ready_deadline = time.monotonic() + self.startup_timeout_s
            if deadline is not None:
                ready_deadline = min(ready_deadline, deadline)
            try:
                if time.monotonic() >= ready_deadline:
                    raise TimeoutError("Sherpa startup deadline exhausted")
                self._req_queue = self._ctx.Queue()
                self._resp_queue = self._ctx.Queue()
                self._proc = self._ctx.Process(
                    target=_sherpa_inference_worker,
                    args=(self._req_queue, self._resp_queue, self.model_dir, self.num_threads),
                    daemon=True,
                )
                self._proc.start()
                while True:
                    remaining = ready_deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Sherpa ready handshake timed out")
                    try:
                        response = self._resp_queue.get(timeout=min(0.02, remaining))
                    except queue.Empty:
                        if not self._proc.is_alive():
                            raise RuntimeError(f"Sherpa exited during startup: {self._proc.exitcode}")
                        continue
                    if response.get("status") != "ready":
                        raise RuntimeError(f"Sherpa initialization failed: {response.get('error')}")
                    self._ready = True
                    return True
            except Exception:
                self._kill_worker()
                raise

    def _kill_worker(self):
        with self._lifecycle_lock:
            self._ready = False
            proc = self._proc
            if proc is not None:
                if proc.pid is not None:
                    if proc.is_alive():
                        proc.terminate()
                    proc.join(timeout=1.0)
                    if proc.is_alive():
                        proc.kill()
                        proc.join(timeout=1.0)
                    if proc.is_alive():
                        # Retain the handle and refuse to spawn another worker.
                        raise RuntimeError("Sherpa worker could not be reaped")
                proc.close()
                self._proc = None
            # A queue used by a killed writer is never reused. Do not join its
            # feeder: termination can leave the underlying pipe lock held.
            for name in ("_req_queue", "_resp_queue"):
                channel = getattr(self, name, None)
                if channel is not None:
                    channel.cancel_join_thread()
                    channel.close()
                    setattr(self, name, None)

    def transcribe(self, wav_path: str, timeout: float = 10.0, language: Optional[str] = None) -> Dict[str, Any]:
        t0 = time.monotonic()
        run_language = language or self.language
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if self.is_mock:
            mock = MockASR(language=run_language, model_id=self.model_id)
            return mock.transcribe(wav_path, timeout=timeout)
        deadline = t0 + timeout
        peak_bytes = None
        phase = "queue_wait"
        def result(text="", error=None, timed_out=False, audio_dur=0.0):
            elapsed = time.monotonic() - t0
            return {
                "text": text, "language": run_language, "model_id": self.model_id,
                "duration_s": elapsed, "duration": elapsed, "error": error,
                "timeout": timed_out, "failure_phase": phase if error else None,
                "peak_memory_bytes": peak_bytes, "peak_process_memory_bytes": peak_bytes,
                "segments": [{"start": 0.0, "end": audio_dur, "text": text}] if text else [],
            }
        if not self._lifecycle_lock.acquire(timeout=max(0.0, deadline-time.monotonic())):
            return result(error="Timeout", timed_out=True)
        try:
            phase = "startup"
            self._ensure_worker(deadline)
            phase = "inference"
            request_id = uuid.uuid4().hex
            self._req_queue.put({"wav_path": os.path.abspath(wav_path),
                                 "language": run_language, "request_id": request_id})
            p = psutil.Process(self._proc.pid) if psutil else None
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Sherpa request deadline exhausted")
                if p:
                    try:
                        peak_bytes = max(peak_bytes or 0, p.memory_info().rss)
                    except psutil.Error:
                        pass
                try:
                    res = self._resp_queue.get(timeout=min(0.02, remaining))
                except queue.Empty:
                    if not self._proc.is_alive():
                        raise RuntimeError(f"Sherpa exited during inference: {self._proc.exitcode}")
                    continue
                if res.get("request_id") != request_id or res.get("status") not in ("ok", "error"):
                    raise RuntimeError("Sherpa response protocol mismatch")
                return result(text=res.get("text", ""), error=res.get("error"),
                              audio_dur=res.get("audio_duration", 0.0))
        except Exception as exc:
            try:
                self._kill_worker()
            except Exception as cleanup_error:
                return result(error=f"CleanupError: {cleanup_error}")
            return result(error="Timeout" if isinstance(exc, TimeoutError) else f"WorkerError: {exc}",
                          timed_out=isinstance(exc, TimeoutError))
        finally:
            self._lifecycle_lock.release()

    def __del__(self):
        try:
            self._kill_worker()
        except Exception:
            pass  # Explicit cleanup reports errors; finalizers must not raise.


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
