"""Windows-runnable software contract checks, not Linux/hardware evidence."""
import subprocess
from unittest.mock import Mock

import pytest

from app.config import AppConfig
from audio.models import AudioConfig
from hardware.recorder import ALSARecorder
from linux_app import settings
from providers.playback import LinuxALSAPlayback


def test_playback_uses_configured_alsa_device_and_timeout(tmp_path, monkeypatch):
    wav = tmp_path/'sample.wav'
    wav.write_bytes(b'test')
    run = Mock()
    monkeypatch.setattr(subprocess, 'run', run)
    LinuxALSAPlayback('plughw:2,0', timeout=7).play(str(wav))
    run.assert_called_once_with(['aplay', '-q', '-D', 'plughw:2,0', str(wav)], check=True, capture_output=True, text=True, timeout=7)


def test_playback_timeout_propagates_failure(tmp_path, monkeypatch):
    wav = tmp_path/'sample.wav'
    wav.write_bytes(b'test')
    monkeypatch.setattr(subprocess, 'run', Mock(side_effect=subprocess.TimeoutExpired('aplay', 1)))
    with pytest.raises(RuntimeError, match='timed out'):
        LinuxALSAPlayback(timeout=1).play(str(wav))


@pytest.mark.parametrize('device,expected', [('plughw:2,0','plughw:2,0'), (2,'hw:2'), (None,'default')])
def test_capture_accepts_full_alsa_device_name(tmp_path, monkeypatch, device, expected):
    popen = Mock()
    monkeypatch.setattr(subprocess, 'Popen', popen)
    rec = ALSARecorder(AudioConfig(device_id=device), temp_dir=str(tmp_path))
    rec.start_recording()
    assert popen.call_args.args[0][1:3] == ['-D',expected]


def test_linux_paths_resolve_from_config_directory(tmp_path, monkeypatch):
    config = AppConfig(str(tmp_path/'config.json'))
    config.data.update(asr_bin='bin/whisper-cli', model_path='models/small.bin', storage_dir='state', language='en')
    monkeypatch.setattr('linux_app.shutil.which', lambda _: '/usr/bin/espeak-ng')
    result = settings(config)
    assert result['storage_dir'] == str(tmp_path/'state')
    assert result['asr_bin'] == str(tmp_path/'bin'/'whisper-cli')
    assert result['language'] == 'en'
