import json
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from nlp.intent_parser import IntentParser
from app.conversation import ConversationManager, ConvoState
from storage.database import StorageManager, get_db_conn
from scheduler.scheduler import ReminderScheduler
from providers.asr_worker import WhisperCppASR
from audio.speech_normalization import BoundedSpeechNormalizer


@pytest.mark.parametrize("date", ["2026-10-18", "2026-12-31", "2028-02-28"])
def test_time_correction_preserves_tomorrow_exact_utc(tmp_path, monkeypatch, date):
    now = datetime.fromisoformat(date).replace(hour=18, tzinfo=ZoneInfo("Asia/Kolkata"))
    parser = IntentParser(now)
    db = StorageManager(str(tmp_path/"state.db"))
    cm = ConversationManager(db, ReminderScheduler(db.db_path), None, None, clock=lambda: now)
    cm.nlp_engine = SimpleNamespace(parse=parser.parse)
    monkeypatch.setattr(cm, "_speak", lambda text: None)
    try:
        cm._process_new_intent("Remind me to buy groceries tomorrow at two PM")
        previous = cm.pending_action_id
        cm._process_confirmation("Change time to four PM")
        assert cm.state == ConvoState.AWAITING_CONFIRMATION
        assert cm.pending_action_id != previous
        cm._process_confirmation("yes")
        with get_db_conn(db.db_path) as conn:
            due = conn.execute("SELECT due_time FROM reminders").fetchone()[0]
            old = conn.execute("SELECT state FROM actions WHERE id = ?", (previous,)).fetchone()[0]
        target = (now + timedelta(days=1)).replace(hour=16, minute=0).astimezone(timezone.utc)
        assert datetime.fromisoformat(due) == target
        assert old == "cancelled"
    finally:
        cm._cancel_ttl()


@pytest.mark.parametrize("correction,expected_state,expected_date", [
    ("no make it today at four PM", ConvoState.CLARIFYING, "2026-10-18"),
    ("no make it today at eight PM", ConvoState.AWAITING_CONFIRMATION, "2026-10-18"),
    ("change to tomorrow", ConvoState.AWAITING_CONFIRMATION, "2026-10-19"),
    ("change time to four", ConvoState.CLARIFYING, "2026-10-19"),
])
def test_corrections_revalidate_merged_instant(tmp_path, monkeypatch, correction, expected_state, expected_date):
    now = datetime(2026,10,18,18,tzinfo=ZoneInfo("Asia/Kolkata"))
    db = StorageManager(str(tmp_path/"state.db"))
    cm = ConversationManager(db, ReminderScheduler(db.db_path), None, None, clock=lambda: now)
    cm.nlp_engine = SimpleNamespace(parse=IntentParser(now).parse)
    monkeypatch.setattr(cm, "_speak", lambda text: None)
    try:
        cm._process_new_intent("Remind me to buy groceries tomorrow at two PM")
        cm._process_confirmation(correction)
        assert cm.state == expected_state
        assert cm.pending_intent_data["slots"]["date"] == expected_date
        with get_db_conn(db.db_path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM reminders").fetchone()[0] == 0
    finally:
        cm._cancel_ttl()


@pytest.mark.parametrize("text,expected,count", [
    ("[BLANK_AUDIO]", "", 1),
    (" [BLANK_AUDIO] [BLANK_AUDIO] ", "", 2),
    ("hello [BLANK_AUDIO] world", "hello world", 1),
    ("save [milk] and say blank audio", "save [milk] and say blank audio", 0),
    ("[MUSIC] hello", "[MUSIC] hello", 0),
])
def test_whisper_marker_contract(tmp_path, monkeypatch, text, expected, count):
    wav = tmp_path/"clip.wav"
    wav.write_bytes(b"fixture")
    cli, model = tmp_path/"cli.exe", tmp_path/"model.bin"
    cli.touch()
    model.touch()
    def measured(argv, timeout):
        Path(argv[argv.index("-of")+1]+".json").write_text(json.dumps({
            "transcription":[{"text":text,"timestamps":{"from":"00:00:00.000","to":"00:00:01.000"}}]
        }),encoding="utf-8")
        return subprocess.CompletedProcess(argv,0,b"raw",b""),False,123
    monkeypatch.setattr("providers.asr_worker.run_measured", measured)
    result = WhisperCppASR(str(cli),str(model),strict_real=True).transcribe(str(wav))
    assert result["text"] == expected
    assert result["non_speech_markers_removed"] == count
    assert result["raw_text"] == text.strip()
    assert result["error"] is None
    assert len(result["segments"]) == bool(expected)
    if expected:
        assert result["segments"][0]["text"] == expected
        assert result["segments"][0]["end"] == "00:00:01.000"


def quiet_variable_signal():
    t = np.arange(32000)/16000
    return (0.0005*(1+0.8*np.sin(2*np.pi*3*t))*np.sin(2*np.pi*150*t)).astype("float32")


def test_gain_bounds_and_original_neural_input():
    audio = quiet_variable_signal()
    before = audio.copy()
    seen = []
    normalizer = BoundedSpeechNormalizer(speech_gate=lambda x,sr: seen.append(x) or True)
    processed, info = normalizer.process(audio,16000)
    assert info["applied"] and 1 < info["gain"] <= 10**1.5
    assert np.max(abs(processed)) <= 0.800001
    assert processed.shape == audio.shape and processed.dtype == audio.dtype
    np.testing.assert_array_equal(before,audio)
    np.testing.assert_array_equal(seen[0],audio)


@pytest.mark.parametrize("kind",["silence","dc","tone","noise","bursty_noise","loud","empty","nan","inf","stereo","int16","rate","clipped"])
def test_unsafe_or_ineligible_signals_never_boosted(kind):
    rng = np.random.default_rng(42)
    t = np.arange(32000)/16000
    audio = quiet_variable_signal()
    sr = 16000
    if kind == "silence": audio[:] = 0
    elif kind == "dc": audio[:] = .001
    elif kind == "tone": audio = (.001*np.sin(2*np.pi*440*t)).astype("float32")
    elif kind == "noise": audio = rng.normal(0,.001,len(t)).astype("float32")
    elif kind == "bursty_noise": audio = (rng.normal(0,.005,len(t))*(np.sin(2*np.pi*2*t)>0)).astype("float32")
    elif kind == "loud": audio *= 100
    elif kind == "empty": audio = audio[:0]
    elif kind == "nan": audio[0] = np.nan
    elif kind == "inf": audio[0] = np.inf
    elif kind == "stereo": audio = np.stack([audio,audio],axis=1)
    elif kind == "int16": audio = (audio*32767).astype("int16")
    elif kind == "rate": sr = 8000
    elif kind == "clipped": audio[0] = 1.0
    result, info = BoundedSpeechNormalizer(speech_gate=lambda x,sr: True).process(audio,sr)
    assert not info["applied"]
    np.testing.assert_array_equal(result,audio)


@pytest.mark.parametrize("correction,reply,expected_hour", [
    ("no make it ten", "pm", 22),
    ("no make it today at four PM", "tomorrow", 16),
])
def test_correction_clarification_keeps_merged_slots(tmp_path, monkeypatch, correction, reply, expected_hour):
    now = datetime(2026,10,18,18,tzinfo=ZoneInfo("Asia/Kolkata"))
    db = StorageManager(str(tmp_path/"state.db"))
    cm = ConversationManager(db, ReminderScheduler(db.db_path), None, None, clock=lambda: now)
    cm.nlp_engine = SimpleNamespace(parse=IntentParser(now).parse)
    monkeypatch.setattr(cm, "_speak", lambda text: None)
    try:
        cm._process_new_intent("Remind me to call John tomorrow at nine AM")
        cm._process_confirmation(correction)
        assert cm.state == ConvoState.CLARIFYING
        cm._process_clarification("yes")
        assert cm.state == ConvoState.CLARIFYING
        if reply == "pm":
            cm._process_clarification("tomorrow")
            assert cm.state == ConvoState.CLARIFYING
            assert "am_pm" in cm.pending_intent_data["ambiguous"]
        with get_db_conn(db.db_path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM reminders").fetchone()[0] == 0
        cm._process_clarification(reply)
        assert cm.state == ConvoState.AWAITING_CONFIRMATION
        assert cm.pending_intent_data["slots"]["time"] == f"{expected_hour:02d}:00"
        cm._process_confirmation("yes")
        with get_db_conn(db.db_path) as conn:
            due = conn.execute("SELECT due_time FROM reminders").fetchone()[0]
        expected = datetime(2026,10,19,expected_hour,tzinfo=ZoneInfo("Asia/Kolkata"))
        assert datetime.fromisoformat(due) == expected.astimezone(timezone.utc)
    finally:
        cm._cancel_ttl()


@pytest.mark.parametrize("mode",["missing","reject","error"])
def test_neural_gate_fails_closed(mode):
    def broken(x,sr): raise RuntimeError("VAD failure")
    gate = None if mode=="missing" else broken if mode=="error" else lambda x,sr: False
    audio = quiet_variable_signal()
    result, info = BoundedSpeechNormalizer(vad_model="missing.onnx",speech_gate=gate).process(audio,16000)
    assert not info["applied"]
    np.testing.assert_array_equal(result,audio)
