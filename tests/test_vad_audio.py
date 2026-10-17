import os
import wave
import numpy as np
import pytest
from audio.recording_validation import inspect_recording

def write_wav(path, signal, sample_rate=16000):
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        # Ensure it's in the 16-bit range
        signal_int16 = np.clip(signal, -32768, 32767).astype('<i2')
        w.writeframes(signal_int16.tobytes())

@pytest.fixture
def tone_wav(tmp_path):
    path = tmp_path / "tone.wav"
    t = np.linspace(0, 1.0, 16000, False)
    signal = 5000 * np.sin(2 * np.pi * 440 * t)
    write_wav(path, signal)
    return path

@pytest.fixture
def static_wav(tmp_path):
    path = tmp_path / "static.wav"
    # White noise
    signal = np.random.normal(0, 2000, 16000)
    write_wav(path, signal)
    return path

@pytest.fixture
def quiet_speech_wav(tmp_path):
    path = tmp_path / "quiet_speech.wav"
    # To simulate speech, we want fluctuating energy and ZCR
    # We can create a bursty signal with varying frequencies
    signal = np.zeros(16000)
    t = np.linspace(0, 1.0, 16000, False)
    # 5 bursts of "speech"
    for start in [0.1, 0.3, 0.5, 0.7, 0.9]:
        idx_start = int(start * 16000)
        idx_end = idx_start + int(0.05 * 16000) # 50ms burst
        burst = 300 * np.sin(2 * np.pi * 300 * t[idx_start:idx_end]) # fundamental
        burst += 150 * np.sin(2 * np.pi * 1200 * t[idx_start:idx_end]) # formant
        burst += np.random.normal(0, 50, len(burst)) # fricative noise
        # Apply an envelope so it's not a block
        env = np.hanning(len(burst))
        signal[idx_start:idx_end] = burst * env
    write_wav(path, signal)
    return path

@pytest.fixture
def music_wav(tmp_path):
    path = tmp_path / "music.wav"
    # Simulate music as a sustained chord
    t = np.linspace(0, 1.0, 16000, False)
    # C major chord (C4, E4, G4)
    signal = 2000 * np.sin(2 * np.pi * 261.63 * t) + \
             2000 * np.sin(2 * np.pi * 329.63 * t) + \
             2000 * np.sin(2 * np.pi * 392.00 * t)
    write_wav(path, signal)
    return path

def test_vad_rejects_tone(tone_wav):
    res = inspect_recording(tone_wav)
    assert any("VAD rejected" in err for err in res["errors"])

def test_vad_rejects_static(static_wav):
    res = inspect_recording(static_wav)
    assert any("VAD rejected" in err for err in res["errors"])

def test_vad_rejects_music(music_wav):
    res = inspect_recording(music_wav)
    assert any("VAD rejected" in err for err in res["errors"])

def test_vad_accepts_quiet_speech(quiet_speech_wav):
    res = inspect_recording(quiet_speech_wav)
    assert not any("VAD rejected" in err for err in res["errors"])

def test_streaming_vad_speech_boundaries_and_hangover():
    from audio.vad import VoiceActivityDetector
    vad = VoiceActivityDetector(min_speech_ms=100, hangover_ms=300)
    
    # 2 seconds audio: 0.5s silence + 0.6s speech + 0.9s silence
    t = np.linspace(0, 2.0, 32000, False)
    speech = np.zeros(32000, dtype=np.int16)
    
    speech_start = int(0.5 * 16000)
    speech_end = int(1.1 * 16000)
    # Generate speech-like waveform with harmonics
    waveform = (1200 * np.sin(2 * np.pi * 200 * t[speech_start:speech_end]) +
                600 * np.sin(2 * np.pi * 800 * t[speech_start:speech_end]))
    speech[speech_start:speech_end] = waveform.astype(np.int16)
    
    res = vad.process_pcm(speech)
    assert res["has_speech"] is True
    assert res["speech_start_s"] is not None
    assert 0.45 <= res["speech_start_s"] <= 0.55
    # Speech ended around 1.1s
    assert res["speech_end_s"] is not None
    assert 1.05 <= res["speech_end_s"] <= 1.15

def test_streaming_vad_short_confirmation_accepted():
    from audio.vad import VoiceActivityDetector
    vad = VoiceActivityDetector(min_speech_ms=100)
    
    # Short confirmation ("yes" ~ 300ms)
    t = np.linspace(0, 0.5, 8000, False)
    pcm = np.zeros(8000, dtype=np.int16)
    yes_segment = (1500 * np.sin(2 * np.pi * 250 * t[1600:6400]) +
                   800 * np.sin(2 * np.pi * 1500 * t[1600:6400]))
    pcm[1600:6400] = yes_segment.astype(np.int16)
    
    res = vad.process_pcm(pcm)
    assert res["has_speech"] is True
    assert len(res["errors"]) == 0


def test_silero_vad_v5_neural_on_real_speech():
    from audio.vad import VoiceActivityDetector
    vad = VoiceActivityDetector()
    if not vad.use_silero:
        pytest.skip("Silero VAD v5 ONNX model not bundled in lightweight test harness")
    assert vad.use_silero is True

    # Test on real human speech sample
    sample_path = "datasets/fleurs_pilot_v1/clips/en_07f77109b3c8f11cac1a8d49.wav"
    if os.path.isfile(sample_path):
        res = vad.process_file(sample_path)
        assert res["has_speech"] is True
        assert res["engine"] == "silero_v5_onnx"
        assert res["speech_duration_s"] > 2.0



