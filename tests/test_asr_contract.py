import os
import pytest
from unittest.mock import patch, MagicMock
from providers.asr_worker import ASRWorker, MockASR, WhisperCppASR, SherpaOnnxASR

REQUIRED_KEYS = {
    "text",
    "language",
    "model_id",
    "duration_s",
    "error",
    "timeout",
    "peak_memory_bytes"
}

def test_mock_asr_contract():
    mock = MockASR(language="en")
    res = mock.transcribe("fake.wav", timeout=1.0)
    for k in REQUIRED_KEYS:
        assert k in res, f"Missing required contract key: {k}"
    assert res["timeout"] is False
    assert res["error"] is None
    assert isinstance(res["text"], str)
    assert isinstance(res["duration_s"], float)

def test_strict_real_fails_fast_on_missing_model(tmp_path):
    # Non-existent directory
    missing_dir = str(tmp_path / "non_existent_model")
    with pytest.raises(EnvironmentError):
        SherpaOnnxASR(model_dir=missing_dir, strict_real=True)

def test_strict_real_fails_fast_on_whisper(tmp_path):
    missing_cli = str(tmp_path / "missing_main.exe")
    missing_model = str(tmp_path / "missing_model.bin")
    with pytest.raises(FileNotFoundError):
        WhisperCppASR(cli_path=missing_cli, model_path=missing_model, strict_real=True)

def test_sherpa_timeout_enforcement():
    asr = SherpaOnnxASR(model_dir="models/sherpa-onnx-zipformer-en-2023-06-26", strict_real=False)
    # Simulate a timeout by specifying an impossible 0.0001s timeout
    res = asr.transcribe("tests/fixtures/silence.wav", timeout=0.0001)
    for k in REQUIRED_KEYS:
        assert k in res, f"Missing required contract key: {k}"
    assert res["timeout"] is True
    assert res["error"] == "Timeout"
    assert res["text"] == ""

def test_asr_worker_facade_contract():
    worker = ASRWorker(engine="mock", language="en")
    res = worker.transcribe("dummy.wav")
    for k in REQUIRED_KEYS:
        assert k in res, f"Missing required contract key: {k}"
    assert res["language"] == "en"
