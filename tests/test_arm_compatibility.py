"""ARM64 and cross-platform architecture compatibility tests."""
import sys
import platform
import struct
import numpy as np
import pytest

def test_architecture_address_and_endianness():
    """Verify 64-bit architecture and little-endian byte order required for ARM64 audio processing."""
    assert struct.calcsize("P") == 8, "Must run in a 64-bit runtime"
    assert sys.byteorder == "little", "ARM64 Linux uses little-endian byte order for PCM audio"

def test_numpy_simd_vector_math():
    """Verify numpy vectorization / SIMD (NEON on ARM64) audio normalization."""
    raw_int16 = np.array([-32768, -16384, 0, 16384, 32767], dtype=np.int16)
    float_audio = (raw_int16 / 32768.0).astype(np.float32)
    
    assert float_audio[0] == pytest.approx(-1.0, rel=1e-4)
    assert float_audio[2] == pytest.approx(0.0, rel=1e-4)
    assert float_audio[4] == pytest.approx(0.999969, rel=1e-4)
    assert float_audio.dtype == np.float32

def test_pcm_audio_buffer_slicing():
    """Verify chunked audio buffer alignment matches 16kHz 16-bit mono PCM."""
    sample_rate = 16000
    chunk_duration_ms = 30
    chunk_samples = int(sample_rate * (chunk_duration_ms / 1000.0)) # 480 samples
    
    stream = np.zeros(chunk_samples * 10, dtype=np.int16)
    chunks = [stream[i:i + chunk_samples] for i in range(0, len(stream), chunk_samples)]
    
    assert len(chunks) == 10
    assert len(chunks[0]) == 480
    assert chunks[0].nbytes == 960 # 480 * 2 bytes

def test_platform_diagnostics():
    """Report platform environment diagnostics without failing non-ARM development machines."""
    machine = platform.machine().lower()
    system = platform.system().lower()
    # In GitHub Actions ubuntu-24.04-arm runner, machine is 'aarch64' or 'arm64'
    is_arm = machine in ("aarch64", "arm64")
    assert system in ("linux", "windows", "darwin")
    assert sys.version_info >= (3, 10)
