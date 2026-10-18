import sys
import time
import psutil
import pytest
from scripts.compare_human_local import measure, main


def test_meter_includes_parent_and_preserves_result():
    result, elapsed, peak = measure(lambda: (time.sleep(.03), {'text': 'ok'})[1])
    assert result['text'] == 'ok'
    assert elapsed >= .03
    assert peak >= psutil.Process().memory_info().rss * .8


def test_meter_preserves_failure_elapsed():
    def fail():
        time.sleep(.02)
        raise RuntimeError('intentional')
    result, elapsed, peak = measure(fail)
    assert 'intentional' in result['error']
    assert elapsed >= .02 and peak > 0


def test_missing_neural_model_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['compare', '--project', str(tmp_path),
        '--output', str(tmp_path/'output'), '--engines', 'sherpa', '--safe-normalize-sherpa'])
    with pytest.raises(RuntimeError, match='real neural VAD'):
        main()
    assert not (tmp_path/'output').exists()


def test_normalization_cannot_change_whisper(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['compare', '--project', str(tmp_path),
        '--output', str(tmp_path/'output'), '--engines', 'whisper', '--safe-normalize-sherpa'])
    with pytest.raises(ValueError, match='Sherpa-only'):
        main()
