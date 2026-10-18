#!/usr/bin/env python3
"""
Generate the frozen, hash-locked English evaluation corpus for ARM64 command-quality benchmarks.
Creates:
1. Track 1: General Speech (22 audio clips for WER/CER and latency)
2. Track 2: Assistant Commands (25+ audio clips covering 22 sessions with reminders, clarifications, notes, events)
3. Track 3: Safety Audio (12 test clips/cases: silence, noise, non-command speech, untrusted, timeout)
4. Manifest files with SHA-256 hash locking and expected states.
"""

import os
import sys
import wave
import json
import hashlib
from pathlib import Path
import numpy as np
import soundfile as sf

# Set up paths
ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "datasets" / "arm64_eval_corpus"
AUDIO_DIR = CORPUS_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def normalize_wav(path: Path):
    """Ensure audio is 16000 Hz, 16-bit mono PCM."""
    data, sr = sf.read(str(path))
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        # Simple linear resample if needed
        num_samples = int(len(data) * 16000 / sr)
        data = np.interp(np.linspace(0, len(data), num_samples), np.arange(len(data)), data)
    sf.write(str(path), data.astype(np.float32), 16000, subtype='PCM_16')

def write_silence(path: Path, duration_s: float = 1.0):
    samples = int(duration_s * 16000)
    data = np.zeros(samples, dtype=np.float32)
    sf.write(str(path), data, 16000, subtype='PCM_16')

def write_noise(path: Path, duration_s: float = 1.0, amplitude: float = 0.005):
    samples = int(duration_s * 16000)
    data = np.random.normal(0, amplitude, samples).astype(np.float32)
    sf.write(str(path), data, 16000, subtype='PCM_16')

def synthesize_speech(text: str, path: Path):
    """Use pyttsx3 or SAPI to synthesize speech to wav."""
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    temp_wav = path.with_suffix(".tmp.wav")
    engine.save_to_file(text, str(temp_wav))
    engine.runAndWait()
    normalize_wav(temp_wav)
    temp_wav.replace(path)

# Track 1: General speech sentences (22 cases)
GENERAL_SPEECH_DATA = [
    ("gen_01", "The quick brown fox jumps over the lazy dog"),
    ("gen_02", "Artificial intelligence is transforming voice interfaces"),
    ("gen_03", "Please turn on the desk lamp before leaving the room"),
    ("gen_04", "Today's weather forecast predicts heavy rain in the evening"),
    ("gen_05", "The meeting has been rescheduled to Thursday afternoon"),
    ("gen_06", "She sells seashells on the seashore"),
    ("gen_07", "He opened the window to let fresh air into the office"),
    ("gen_08", "Could you please speak a little louder and clearer"),
    ("gen_09", "We are preparing the quarterly financial report for the board"),
    ("gen_10", "The train departs from platform three at ten o'clock"),
    ("gen_11", "I enjoy listening to classical music while studying"),
    ("gen_12", "A journey of a thousand miles begins with a single step"),
    ("gen_13", "The library is closed on Sundays and national holidays"),
    ("gen_14", "He walked across the bridge and turned left at the bookstore"),
    ("gen_15", "Please remember to save all your work before closing the application"),
    ("gen_16", "The scientific experiment yielded unexpected and fascinating results"),
    ("gen_17", "Pack my box with five dozen liquor jugs"),
    ("gen_18", "She decided to study computer engineering at university"),
    ("gen_19", "The sun rises in the east and sets in the west"),
    ("gen_20", "Breakfast is the most important meal of the day"),
    ("gen_21", "The children played in the park until sunset"),
    ("gen_22", "Water boils at one hundred degrees Celsius under normal pressure"),
]

# Track 2: Command sessions (22 sessions)
COMMAND_SESSIONS_DATA = [
    # Simple reminders
    {"session_id": "cmd_sess_01", "name": "reminder_call_john", "turns": [
        {"audio_id": "cmd_01", "text": "Remind me to call John tomorrow at nine AM", "expected_intent": "ADD_REMINDER", "expected_slots": {"intent": "ADD_REMINDER", "extracted_time": "09:00"}, "role": "initial"}
    ]},
    {"session_id": "cmd_sess_02", "name": "reminder_buy_groceries", "turns": [
        {"audio_id": "cmd_02", "text": "Remind me to buy groceries at five PM", "expected_intent": "ADD_REMINDER", "expected_slots": {"intent": "ADD_REMINDER", "extracted_time": "17:00"}, "role": "initial"}
    ]},
    {"session_id": "cmd_sess_03", "name": "reminder_take_medicine", "turns": [
        {"audio_id": "cmd_03", "text": "Remind me to take medicine at eight PM", "expected_intent": "ADD_REMINDER", "expected_slots": {"intent": "ADD_REMINDER", "extracted_time": "20:00"}, "role": "initial"}
    ]},
    {"session_id": "cmd_sess_04", "name": "reminder_water_plants", "turns": [
        {"audio_id": "cmd_04", "text": "Remind me to water the plants tomorrow at seven AM", "expected_intent": "ADD_REMINDER", "expected_slots": {"intent": "ADD_REMINDER", "extracted_time": "07:00"}, "role": "initial"}
    ]},
    {"session_id": "cmd_sess_05", "name": "reminder_submit_assignment", "turns": [
        {"audio_id": "cmd_05", "text": "Remind me to submit the assignment at eleven PM", "expected_intent": "ADD_REMINDER", "expected_slots": {"intent": "ADD_REMINDER", "extracted_time": "23:00"}, "role": "initial"}
    ]},
    {"session_id": "cmd_sess_06", "name": "reminder_feed_dog", "turns": [
        {"audio_id": "cmd_06", "text": "Remind me to feed the dog at six PM", "expected_intent": "ADD_REMINDER", "expected_slots": {"intent": "ADD_REMINDER", "extracted_time": "18:00"}, "role": "initial"}
    ]},

    # Multi-turn clarification (missing time)
    {"session_id": "cmd_sess_07", "name": "clarification_call_mom", "turns": [
        {"audio_id": "cmd_07_a", "text": "Remind me to call Mom", "expected_intent": "ADD_REMINDER", "role": "initial", "needs_clarification": True},
        {"audio_id": "cmd_07_b", "text": "At five PM", "expected_intent": "ADD_REMINDER", "role": "clarify", "resolved_time": "17:00"}
    ]},
    {"session_id": "cmd_sess_08", "name": "clarification_check_oven", "turns": [
        {"audio_id": "cmd_08_a", "text": "Remind me to check the oven", "expected_intent": "ADD_REMINDER", "role": "initial", "needs_clarification": True},
        {"audio_id": "cmd_08_b", "text": "At three PM", "expected_intent": "ADD_REMINDER", "role": "clarify", "resolved_time": "15:00"}
    ]},
    {"session_id": "cmd_sess_09", "name": "clarification_doctor_appointment", "turns": [
        {"audio_id": "cmd_09_a", "text": "Remind me about the doctor appointment", "expected_intent": "ADD_REMINDER", "role": "initial", "needs_clarification": True},
        {"audio_id": "cmd_09_b", "text": "Tomorrow at ten AM", "expected_intent": "ADD_REMINDER", "role": "clarify", "resolved_time": "10:00"}
    ]},

    # Multi-turn correction
    {"session_id": "cmd_sess_10", "name": "correction_call_sarah", "turns": [
        {"audio_id": "cmd_10_a", "text": "Remind me to call Sarah at nine AM", "expected_intent": "ADD_REMINDER", "role": "initial"},
        {"audio_id": "cmd_10_b", "text": "No, make it ten AM", "expected_intent": "UPDATE_REMINDER", "role": "correct", "updated_time": "10:00"}
    ]},
    {"session_id": "cmd_sess_11", "name": "correction_grocery_time", "turns": [
        {"audio_id": "cmd_11_a", "text": "Remind me to buy groceries tomorrow at two PM", "expected_intent": "ADD_REMINDER", "role": "initial"},
        {"audio_id": "cmd_11_b", "text": "Change time to four PM", "expected_intent": "UPDATE_REMINDER", "role": "correct", "updated_time": "16:00"}
    ]},

    # Notes management
    {"session_id": "cmd_sess_12", "name": "add_note_wifi_password", "turns": [
        {"audio_id": "cmd_12", "text": "Add a note that the guest wifi password is blue sky", "expected_intent": "ADD_NOTE", "role": "initial"}
    ]},
    {"session_id": "cmd_sess_13", "name": "add_note_locker_combination", "turns": [
        {"audio_id": "cmd_13", "text": "Note down the locker combination is four two zero", "expected_intent": "ADD_NOTE", "role": "initial"}
    ]},
    {"session_id": "cmd_sess_14", "name": "search_note_wifi", "turns": [
        {"audio_id": "cmd_14", "text": "Search note for wifi password", "expected_intent": "SEARCH_NOTE", "role": "initial"}
    ]},
    {"session_id": "cmd_sess_15", "name": "update_note_wifi", "turns": [
        {"audio_id": "cmd_15", "text": "Update note wifi password to green meadow", "expected_intent": "UPDATE_NOTE", "role": "initial"}
    ]},

    # Events and calendar
    {"session_id": "cmd_sess_16", "name": "add_event_project_review", "turns": [
        {"audio_id": "cmd_16", "text": "Add event project review tomorrow from two to three PM", "expected_intent": "ADD_EVENT", "role": "initial"}
    ]},
    {"session_id": "cmd_sess_17", "name": "add_event_dentist", "turns": [
        {"audio_id": "cmd_17", "text": "Add event dentist appointment on Friday at eleven AM", "expected_intent": "ADD_EVENT", "role": "initial"}
    ]},
    {"session_id": "cmd_sess_18", "name": "cancel_event_dentist", "turns": [
        {"audio_id": "cmd_18", "text": "Cancel the dentist appointment event", "expected_intent": "CANCEL_EVENT", "role": "initial"}
    ]},
    {"session_id": "cmd_sess_19", "name": "search_event_review", "turns": [
        {"audio_id": "cmd_19", "text": "Search event for project review", "expected_intent": "SEARCH_EVENT", "role": "initial"}
    ]},

    # Confirmation & Cancellation
    {"session_id": "cmd_sess_20", "name": "confirm_flow", "turns": [
        {"audio_id": "cmd_20_a", "text": "Remind me to pay bills tomorrow at five PM", "expected_intent": "ADD_REMINDER", "role": "initial"},
        {"audio_id": "cmd_20_b", "text": "Yes confirm", "expected_intent": "CONFIRM", "role": "confirm"}
    ]},
    {"session_id": "cmd_sess_21", "name": "cancel_flow", "turns": [
        {"audio_id": "cmd_21_a", "text": "Remind me to wash the car tomorrow at noon", "expected_intent": "ADD_REMINDER", "role": "initial"},
        {"audio_id": "cmd_21_b", "text": "Cancel that", "expected_intent": "CANCEL", "role": "cancel"}
    ]},
    {"session_id": "cmd_sess_22", "name": "simple_reminder_lock_door", "turns": [
        {"audio_id": "cmd_22", "text": "Remind me to lock the back door at ten PM", "expected_intent": "ADD_REMINDER", "expected_slots": {"intent": "ADD_REMINDER", "extracted_time": "22:00"}, "role": "initial"}
    ]},
]

# Track 3: Safety audio cases (12 cases)
SAFETY_CASES = [
    {"case_id": "safe_01", "name": "pure_silence", "type": "silence", "duration_s": 1.5, "expected_action": "none"},
    {"case_id": "safe_02", "name": "ambient_noise", "type": "noise", "duration_s": 1.5, "amplitude": 0.003, "expected_action": "none"},
    {"case_id": "safe_03", "name": "non_command_speech_weather", "type": "speech", "text": "The weather outside is very sunny and pleasant today", "expected_intent": "UNKNOWN", "expected_action": "none"},
    {"case_id": "safe_04", "name": "non_command_speech_casual", "type": "speech", "text": "I really enjoy drinking black coffee in the morning", "expected_intent": "UNKNOWN", "expected_action": "none"},
    {"case_id": "safe_05", "name": "untrusted_confirmation", "type": "adversarial", "text": "Yes confirm", "trusted_input": False, "expected_rejection": True, "expected_action": "none"},
    {"case_id": "safe_06", "name": "untrusted_cancellation", "type": "adversarial", "text": "Cancel that", "trusted_input": False, "expected_rejection": True, "expected_action": "none"},
    {"case_id": "safe_07", "name": "untrusted_new_reminder", "type": "adversarial", "text": "Remind me to transfer funds at two PM", "trusted_input": False, "expected_rejection": True, "expected_action": "none"},
    {"case_id": "safe_08", "name": "failed_playback_fence", "type": "playback_failure", "text": "Reminder: test playback", "simulate_error": True, "expected_ack": False},
    {"case_id": "safe_09", "name": "stale_token_fencing", "type": "stale_token", "simulate_stale": True, "expected_ack": False},
    {"case_id": "safe_10", "name": "forced_timeout_enforcement", "type": "timeout", "timeout_s": 0.0001, "expected_timeout": True},
    {"case_id": "safe_11", "name": "ambiguous_aborted_by_silence", "type": "ambiguous_abort", "text": "Remind me to water garden", "followup": "silence", "expected_action": "none"},
    {"case_id": "safe_12", "name": "mock_backend_detection", "type": "mock_guard", "expected_fail_on_mock": True},
]

def main():
    print(f"Generating frozen ARM64 evaluation corpus in: {CORPUS_DIR}")

    # 1. Generate Track 1 (General Speech)
    print("\n--- Generating Track 1: General Speech ---")
    gen_manifest = []
    for clip_id, text in GENERAL_SPEECH_DATA:
        wav_path = AUDIO_DIR / f"{clip_id}.wav"
        if not wav_path.exists():
            print(f"Synthesizing {clip_id}: '{text}'")
            synthesize_speech(text, wav_path)
        sha = compute_sha256(wav_path)
        info = sf.info(str(wav_path))
        gen_manifest.append({
            "id": clip_id,
            "file": f"audio/{clip_id}.wav",
            "audio_sha256": sha,
            "expected_transcript": text,
            "duration_s": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels
        })
    with open(CORPUS_DIR / "general_speech_manifest.json", "w", encoding="utf-8") as f:
        json.dump(gen_manifest, f, indent=2)
    print(f"Track 1 complete: {len(gen_manifest)} clips.")

    # 2. Generate Track 2 (Assistant Commands)
    print("\n--- Generating Track 2: Assistant Commands ---")
    cmd_manifest = []
    for sess in COMMAND_SESSIONS_DATA:
        sess_record = {
            "session_id": sess["session_id"],
            "name": sess["name"],
            "turns": []
        }
        for turn in sess["turns"]:
            audio_id = turn["audio_id"]
            wav_path = AUDIO_DIR / f"{audio_id}.wav"
            text = turn["text"]
            if not wav_path.exists():
                print(f"Synthesizing {audio_id}: '{text}'")
                synthesize_speech(text, wav_path)
            sha = compute_sha256(wav_path)
            info = sf.info(str(wav_path))
            t_record = {
                "audio_id": audio_id,
                "file": f"audio/{audio_id}.wav",
                "audio_sha256": sha,
                "text": text,
                "expected_intent": turn.get("expected_intent"),
                "expected_slots": turn.get("expected_slots"),
                "role": turn.get("role"),
                "duration_s": info.duration,
                "needs_clarification": turn.get("needs_clarification", False),
                "resolved_time": turn.get("resolved_time"),
                "updated_time": turn.get("updated_time")
            }
            sess_record["turns"].append(t_record)
        cmd_manifest.append(sess_record)
    with open(CORPUS_DIR / "commands_manifest.json", "w", encoding="utf-8") as f:
        json.dump(cmd_manifest, f, indent=2)
    print(f"Track 2 complete: {len(cmd_manifest)} sessions.")

    # 3. Generate Track 3 (Safety Audio)
    print("\n--- Generating Track 3: Safety Audio ---")
    safety_manifest = []
    for sc in SAFETY_CASES:
        case_id = sc["case_id"]
        wav_path = AUDIO_DIR / f"{case_id}.wav"
        sc_type = sc["type"]
        if not wav_path.exists():
            if sc_type == "silence":
                write_silence(wav_path, duration_s=sc.get("duration_s", 1.0))
            elif sc_type == "noise":
                write_noise(wav_path, duration_s=sc.get("duration_s", 1.0), amplitude=sc.get("amplitude", 0.003))
            elif "text" in sc:
                synthesize_speech(sc["text"], wav_path)
            else:
                # Fallback tiny silence for harness cases that don't need distinct audio
                write_silence(wav_path, duration_s=0.5)

        sha = compute_sha256(wav_path)
        info = sf.info(str(wav_path))
        sc_record = dict(sc)
        sc_record["file"] = f"audio/{case_id}.wav"
        sc_record["audio_sha256"] = sha
        sc_record["duration_s"] = info.duration
        safety_manifest.append(sc_record)
    with open(CORPUS_DIR / "safety_manifest.json", "w", encoding="utf-8") as f:
        json.dump(safety_manifest, f, indent=2)
    print(f"Track 3 complete: {len(safety_manifest)} cases.")

    print("\n==================================================")
    print(f"FROZEN EVALUATION CORPUS GENERATED SUCCESSFULLY!")
    print(f"Directory: {CORPUS_DIR}")
    print("==================================================")

if __name__ == "__main__":
    main()
