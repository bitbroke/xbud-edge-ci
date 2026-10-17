"""Shared test utilities for generating speech-like synthetic audio.

Pure sine waves have near-zero energy variance (CV < 0.01), which the VAD
correctly rejects. Real speech is bursty with varying amplitude and frequency.
This module provides a helper to generate synthetic signals that pass the VAD
while remaining deterministic and reproducible.
"""
import numpy as np
import wave


def make_speech_like_signal(duration_s=0.5, sample_rate=16000, amplitude=1000):
    """Generate a bursty, speech-like synthetic signal that passes VAD.
    
    The signal consists of short bursts with varying frequency and amplitude,
    separated by brief pauses. This mimics the energy variance (CV > 0.5) and
    ZCR characteristics of real speech.
    """
    n_samples = int(duration_s * sample_rate)
    signal = np.zeros(n_samples)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    
    # Create 3-5 speech-like bursts with different characteristics
    n_bursts = max(3, int(duration_s * 5))
    burst_duration = int(0.06 * sample_rate)  # 60ms bursts
    
    for i in range(n_bursts):
        center = int((i + 0.5) / n_bursts * n_samples)
        start = max(0, center - burst_duration // 2)
        end = min(n_samples, start + burst_duration)
        length = end - start
        
        if length < 10:
            continue
        
        # Fundamental frequency varies per burst (100-400 Hz, typical speech range)
        f0 = 150 + 50 * (i % 5)
        burst_t = np.arange(length) / sample_rate
        
        # Fundamental + formant harmonics
        burst = amplitude * np.sin(2 * np.pi * f0 * burst_t)
        burst += amplitude * 0.5 * np.sin(2 * np.pi * f0 * 2.5 * burst_t)  # formant
        burst += np.random.normal(0, amplitude * 0.05, length)  # fricative noise
        
        # Apply Hanning envelope so energy ramps up and down
        env = np.hanning(length)
        signal[start:end] += burst * env
    
    return np.clip(signal, -32767, 32767).astype('<i2')


def write_speech_wav(path, duration_s=0.5, sample_rate=16000, amplitude=1000):
    """Write a speech-like WAV file that passes the VAD gate."""
    signal = make_speech_like_signal(duration_s, sample_rate, amplitude)
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(signal.tobytes())
    return signal
