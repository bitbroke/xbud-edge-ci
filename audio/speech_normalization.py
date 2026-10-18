"""Experimental, opt-in speech-gated gain. Never edits source audio or trust state."""
from pathlib import Path
import numpy as np


class BoundedSpeechNormalizer:
    def __init__(self, vad_model=None, speech_gate=None):
        self._gate = speech_gate
        self._vad = None
        if self._gate is None and vad_model and Path(vad_model).is_file():
            try:
                from audio.vad import VoiceActivityDetector
                self._vad = VoiceActivityDetector(
                    silero_model_path=str(vad_model), use_silero=True,
                    silero_threshold=0.6, min_speech_ms=300,
                    max_utterance_s=30,
                )
                if not self._vad.use_silero:
                    self._vad = None
            except Exception:
                self._vad = None

    def _neural_speech(self, original, sample_rate):
        if self._gate is not None:
            return bool(self._gate(original.copy(), sample_rate))
        if self._vad is None:
            return False
        # Direct neural call intentionally excludes heuristic fallback.
        result = self._vad._process_silero(
            np.round(original * 32767).astype(np.int16)
        )
        return bool(result and result.get("engine") == "silero_v5_onnx"
                    and result.get("has_speech")
                    and result.get("speech_duration_s", 0) >= 0.3)

    def process(self, audio, sample_rate):
        original = np.asarray(audio)
        output = original.copy()
        info = {"gain": 1.0, "applied": False, "reason": "invalid_input",
                "neural_speech": False}
        if (sample_rate != 16000 or original.ndim != 1
                or original.dtype.kind != "f" or len(original) < 5600
                or len(original) > sample_rate * 30
                or not np.isfinite(original).all()):
            return output, info
        peak = float(np.max(np.abs(original)))
        rms = float(np.sqrt(np.mean(original.astype(np.float64) ** 2)))
        info.update(input_peak=peak, input_rms=rms)
        if peak > 1 or peak < 0.0001 or rms < 0.00001:
            info["reason"] = "out_of_range_or_near_silence"
            return output, info
        if abs(float(np.mean(original))) > rms * 0.25:
            info["reason"] = "dc_offset"
            return output, info
        if rms >= 0.01 or peak >= 0.98:
            info["reason"] = "not_quiet_or_clipped"
            return output, info
        frames = original[:len(original)//320*320].reshape(-1, 320)
        energies = np.sqrt(np.mean(frames.astype(np.float64)**2, axis=1))
        active = frames[energies > max(float(energies.max()) * 0.1, 1e-6)]
        if len(active):
            crossings = np.mean(np.diff(np.signbit(active), axis=1), axis=1)
            if float(np.median(crossings)) > 0.35:
                info["reason"] = "broadband_noise"
                return output, info
        cv = float(np.std(energies) / max(np.mean(energies), 1e-12))
        if cv < 0.08:
            info["reason"] = "stationary_signal"
            return output, info
        try:
            confirmed = self._neural_speech(original, sample_rate)
        except Exception:
            info["reason"] = "neural_error"
            return output, info
        if not confirmed:
            info["reason"] = "no_neural_speech"
            return output, info
        gain = min(10 ** (30 / 20), 0.03 / rms, 0.8 / peak)
        info["neural_speech"] = True
        if gain <= 1.05:
            info["reason"] = "no_gain_needed"
            return output, info
        output = (original * gain).astype(original.dtype)
        info.update(gain=float(gain), applied=True, reason="bounded_neural_speech",
                    output_peak=float(np.max(np.abs(output))))
        return output, info
