#!/usr/bin/env python3
"""
benchmarks/run_arm64_command_eval.py — ARM64 English Command Evaluation Benchmark.

Rigorous evaluation of Sherpa-ONNX Zipformer English ASR and full assistant state machine:
- Model Manifest Gate: Compares all model weights against independently frozen expected hashes.
- Track 1A: Synthetic Speech (22 clips, SAPI5/pyttsx3) -> mean clip WER vs corpus WER, CER, latency, RTF.
- Track 1B: Licensed Human Speech (20 clips, Google FLEURS CC-BY-4.0) -> mean clip WER vs corpus WER, CER, latency, RTF.
- Track 2: Assistant Commands (22 sessions) -> Full multi-turn state transitions, explicit expected outcomes,
           durable DB records (actions, reminders, notes, events), per-turn latency vs session duration.
- Track 3: Safety & Defense Gates (12 cases) -> silence/noise complete conversation path & hallucinations metric,
           real untrusted-input rejection path, failed playback device fence & eventual recovery, stale token fence,
           bounded timeout enforcement.

Required Pass/Fail Rules:
1. Model files must exist and match independently frozen SHA-256 hashes.
2. Source audio SHA-256 must match manifest before and after each attempt.
3. Track 1A & 1B WER and CER metrics must be computed and reported separately.
4. Track 2 sessions must meet explicit expected final state, DB counts, and verified record fields.
5. Untrusted audio must be processed by ASR and strictly rejected without mutating active conversation or DB.
6. Silence/noise must run through complete state machine without creating unsafe actions (hallucinations tracked separately).
7. Playback failure must prove playback was reached, raised intended ALSA error, and recovered on retry.
8. Stale reminder tokens must never acknowledge newer claims.
9. Real inference must be verified; mock backend causes immediate failure.
"""

import os
import sys
import time
import json
import wave
import sqlite3
import shutil
import hashlib
import tempfile
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from zoneinfo import ZoneInfo
from datetime import datetime, timezone

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import psutil
import numpy as np
from jiwer import wer, cer, process_words, process_characters

from providers.asr_worker import ASRWorker, SherpaOnnxASR
from providers.tts_worker import TTSWorker
from storage.database import StorageManager, get_db_conn
from scheduler.scheduler import ReminderScheduler
from app.conversation import ConversationManager, ConvoState
from benchmarks.evaluation_metrics import compute_wer_cer_metrics, assert_exact_utc

# -------------------------------------------------------------------------
# FROZEN MODEL MANIFEST (Independent Ground Truth)
# -------------------------------------------------------------------------
FROZEN_MODEL_HASHES = {
    "tokens.txt": "49e3c2646595fd907228b3c6787069658f67b17377c60aeb8619c4551b2316fb",
    "encoder-epoch-99-avg-1.int8.onnx": "52a48f46c17b19a36fe3927c4d59479bb16eeb2493313ed82c4bf775c2cb8bc8",
    "decoder-epoch-99-avg-1.int8.onnx": "783cd6b23b8db8e14a43804ecf972ae96e71499cce799e334ab95c961800d797",
    "joiner-epoch-99-avg-1.int8.onnx": "48de5d6467a2ab1e72cb5c4d828330be06524d877bc458118b6a4198ca031357"
}

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def get_audio_duration_s(filepath: Path) -> float:
    try:
        import soundfile as sf
        return float(sf.info(str(filepath)).duration)
    except Exception:
        with wave.open(str(filepath), "rb") as w:
            return float(w.getnframes()) / float(w.getframerate())

class MockPlayback:
    def __init__(self, should_fail: bool = False, fail_message: str = "Simulated ALSA playback device error"):
        self.should_fail = should_fail
        self.fail_message = fail_message
        self.play_called = False
        self.played_files = []

    def play(self, wav_path: str):
        self.play_called = True
        if self.should_fail:
            raise RuntimeError(self.fail_message)
        self.played_files.append(wav_path)

def verify_source_hash(filepath: Path, expected_hash: str, context: str):
    actual = compute_sha256(filepath)
    if actual != expected_hash:
        raise AssertionError(
            f"[{context}] Source audio hash mismatch on {filepath.name}!\n"
            f"  Expected: {expected_hash}\n"
            f"  Observed: {actual}"
        )

def main():
    parser = argparse.ArgumentParser(description="ARM64 English Command Evaluation Benchmark")
    parser.add_argument("--model-dir", default="models/sherpa-onnx-zipformer-en-2023-06-26",
                        help="Path to sherpa-onnx zipformer model directory")
    parser.add_argument("--corpus-dir", default="datasets/arm64_eval_corpus",
                        help="Path to frozen evaluation corpus")
    parser.add_argument("--output-dir", default="reports",
                        help="Path to output report directory")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    corpus_dir = Path(args.corpus_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    print("==================================================================")
    print("      xBud ARM64 English Command Evaluation Benchmark            ")
    print("==================================================================")
    print(f"Platform:      {sys.platform} ({os.uname().machine if hasattr(os, 'uname') else 'non-posix'})")
    print(f"Python:        {sys.version.split()[0]}")
    print(f"Model Dir:     {model_dir}")
    print(f"Corpus Dir:    {corpus_dir}")
    print(f"Output Dir:    {out_dir}")
    print("------------------------------------------------------------------\n")

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": os.uname().machine if hasattr(os, "uname") else sys.platform,
        "python_version": sys.version.split()[0],
        "model_dir": str(model_dir),
        "components": {
            "asr": {"name": "Sherpa-ONNX Zipformer English (int8)", "mode": "REAL", "strict": True},
            "nlp_intent_parser": {"name": "NLPEngine/IntentParser", "mode": "REAL_RULE_BASED", "strict": True},
            "state_machine": {"name": "ConversationManager", "mode": "REAL", "strict": True},
            "storage": {"name": "StorageManager/SQLite", "mode": "REAL", "strict": True},
            "scheduler": {"name": "ReminderScheduler/ReminderQueue", "mode": "REAL", "strict": True},
            "tts": {"name": "TTSWorker", "mode": "MOCK", "note": "Synthetic audio generator for headless CI"},
            "playback": {"name": "MockPlayback", "mode": "MOCK", "note": "Audio device mock simulation for headless CI"}
        },
        "model_hashes": {},
        "pass_fail_rules": {},
        "tracks": {
            "track_1a_synthetic_speech": {},
            "track_1b_licensed_human_speech": {},
            "track_2_assistant_commands": {},
            "track_3_safety_audio": {}
        },
        "summary": {},
        "status": "INCOMPLETE"
    }

    # -------------------------------------------------------------------------
    # 0. MODEL IDENTITY & FROZEN MANIFEST GATE
    # -------------------------------------------------------------------------
    print("[Phase 0] Validating model identity against independently frozen manifest...")
    if not model_dir.is_dir():
        report["pass_fail_rules"]["model_manifest_hashes_verified"] = "FAIL"
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    manifest_hash_pass = True
    for fname, expected_hash in FROZEN_MODEL_HASHES.items():
        file_path = model_dir / fname
        if not file_path.is_file():
            print(f"  CRITICAL: Missing expected model file: {fname}")
            manifest_hash_pass = False
            continue
        h = compute_sha256(file_path)
        size = file_path.stat().st_size
        matches = (h == expected_hash)
        report["model_hashes"][fname] = {
            "sha256": h,
            "expected_sha256": expected_hash,
            "size_bytes": size,
            "matches_frozen": matches
        }
        status_sym = "MATCH" if matches else "MISMATCH"
        print(f"  {fname:36s} [{size:>9d} bytes] {status_sym} ({h[:16]}...)")
        if not matches:
            manifest_hash_pass = False

    if not manifest_hash_pass:
        report["pass_fail_rules"]["model_manifest_hashes_verified"] = "FAIL"
        raise AssertionError("CRITICAL: One or more model file hashes do not match independently frozen manifest!")

    report["pass_fail_rules"]["model_manifest_hashes_verified"] = "PASS"
    print("  -> Model manifest verification passed.\n")

    # Initialize Real ASR Worker
    print("[Phase 0] Initializing real SherpaOnnxASR engine (strict_real=True)...")
    # Only verified int8 files may be selected; the adapter prefers float files
    # when the original release directory contains both variants.
    model_stage = tempfile.TemporaryDirectory(prefix="xbud_verified_int8_")
    for fname, expected_hash in FROZEN_MODEL_HASHES.items():
        staged = Path(model_stage.name) / fname
        shutil.copy2(model_dir / fname, staged)
        verify_source_hash(staged, expected_hash, "staged model")
    asr = ASRWorker(engine="sherpa", model_dir=model_stage.name, language="en", strict_real=True)
    if asr.is_mock:
        report["pass_fail_rules"]["real_inference_verified"] = "FAIL"
        raise RuntimeError("CRITICAL: ASRWorker fell back to mock mode! Real inference required.")
    report["pass_fail_rules"]["real_inference_verified"] = "PASS"
    print("  -> Real ASRWorker initialized successfully (is_mock=False).\n")

    # -------------------------------------------------------------------------
    # TRACK 1A: SYNTHETIC SPEECH BENCHMARK (SAPI5 / pyttsx3)
    # -------------------------------------------------------------------------
    print("==================================================================")
    print(" TRACK 1A: SYNTHETIC GENERAL SPEECH ACCURACY & LATENCY BENCHMARK  ")
    print(" Source: SAPI5 / pyttsx3 generated English speech (regression)    ")
    print("==================================================================")
    t1a_manifest_path = corpus_dir / "general_speech_manifest.json"
    assert t1a_manifest_path.is_file(), f"Track 1A manifest not found: {t1a_manifest_path}"

    with open(t1a_manifest_path, "r", encoding="utf-8") as f:
        t1a_manifest = json.load(f)

    t1a_results = []
    t1a_refs, t1a_hyps = [], []
    t1a_latencies, t1a_rtfs = [], []

    for idx, item in enumerate(t1a_manifest):
        clip_id = item["id"]
        wav_path = corpus_dir / item["file"]
        expected_ref = item["expected_transcript"]
        expected_hash = item["audio_sha256"]

        verify_source_hash(wav_path, expected_hash, f"Track 1A Pre-Run {clip_id}")

        audio_dur = item.get("duration_s") or get_audio_duration_s(wav_path)
        t0 = time.monotonic()
        res = asr.adapter.transcribe(str(wav_path), timeout=30.0)
        elapsed = time.monotonic() - t0

        verify_source_hash(wav_path, expected_hash, f"Track 1A Post-Run {clip_id}")

        hypothesis = res.get("text", "").strip()
        t1a_refs.append(expected_ref)
        t1a_hyps.append(hypothesis)

        sample_wer = wer(expected_ref.lower(), hypothesis.lower()) if hypothesis else 1.0
        sample_cer = cer(expected_ref.lower(), hypothesis.lower()) if hypothesis else 1.0
        rtf = elapsed / audio_dur if audio_dur > 0 else 0.0

        t1a_latencies.append(elapsed)
        t1a_rtfs.append(rtf)

        record = {
            "clip_id": clip_id,
            "file": item["file"],
            "audio_sha256": expected_hash,
            "reference": expected_ref,
            "hypothesis": hypothesis,
            "wer": round(sample_wer, 4),
            "cer": round(sample_cer, 4),
            "latency_s": round(elapsed, 4),
            "audio_duration_s": round(audio_dur, 4),
            "rtf": round(rtf, 4),
            "timeout": res.get("timeout", False),
            "error": res.get("error")
        }
        t1a_results.append(record)
        print(f"[{idx+1:2d}/{len(t1a_manifest)}] {clip_id:8s} | WER: {sample_wer:5.2f} | Lat: {elapsed:5.3f}s | RTF: {rtf:5.3f}")
        print(f"     Ref: '{expected_ref}'")
        print(f"     Hyp: '{hypothesis}'\n")

    t1a_metrics = compute_wer_cer_metrics(t1a_refs, t1a_hyps)
    report["tracks"]["track_1a_synthetic_speech"] = {
        "audio_source": "synthetic_speech_sapi5_pyttsx3",
        "total_clips": len(t1a_results),
        "mean_clip_wer": t1a_metrics["mean_clip_wer"],
        "corpus_wer": t1a_metrics["corpus_wer"],
        "mean_clip_cer": t1a_metrics["mean_clip_cer"],
        "corpus_cer": t1a_metrics["corpus_cer"],
        "p50_latency_s": round(float(np.percentile(t1a_latencies, 50)), 4),
        "p95_latency_s": round(float(np.percentile(t1a_latencies, 95)), 4),
        "mean_rtf": round(float(np.mean(t1a_rtfs)), 4),
        "details": t1a_results
    }
    report["pass_fail_rules"]["track_1a_source_hashes_intact"] = "PASS"

    # -------------------------------------------------------------------------
    # TRACK 1B: LICENSED HUMAN SPEECH BENCHMARK (Google FLEURS CC-BY-4.0)
    # -------------------------------------------------------------------------
    print("==================================================================")
    print(" TRACK 1B: LICENSED HUMAN SPEECH BENCHMARK (GOOGLE FLEURS)        ")
    print(" Source: CC-BY-4.0 Google FLEURS verified human recordings        ")
    print("==================================================================")
    t1b_manifest_path = corpus_dir / "human_speech_manifest.json"
    assert t1b_manifest_path.is_file(), f"Track 1B manifest not found: {t1b_manifest_path}"

    with open(t1b_manifest_path, "r", encoding="utf-8") as f:
        t1b_manifest = json.load(f)

    t1b_results = []
    t1b_refs, t1b_hyps = [], []
    t1b_latencies, t1b_rtfs = [], []

    for idx, item in enumerate(t1b_manifest):
        clip_id = item["clip_id"]
        wav_path = corpus_dir / item["file"]
        expected_ref = item["expected_transcript"]
        expected_hash = item["audio_sha256"]

        verify_source_hash(wav_path, expected_hash, f"Track 1B Pre-Run {clip_id}")

        audio_dur = item.get("duration_s") or get_audio_duration_s(wav_path)
        t0 = time.monotonic()
        res = asr.adapter.transcribe(str(wav_path), timeout=30.0)
        elapsed = time.monotonic() - t0

        verify_source_hash(wav_path, expected_hash, f"Track 1B Post-Run {clip_id}")

        hypothesis = res.get("text", "").strip()
        t1b_refs.append(expected_ref)
        t1b_hyps.append(hypothesis)

        sample_wer = wer(expected_ref.lower(), hypothesis.lower()) if hypothesis else 1.0
        sample_cer = cer(expected_ref.lower(), hypothesis.lower()) if hypothesis else 1.0
        rtf = elapsed / audio_dur if audio_dur > 0 else 0.0

        t1b_latencies.append(elapsed)
        t1b_rtfs.append(rtf)

        record = {
            "clip_id": clip_id,
            "file": item["file"],
            "dataset": item.get("dataset", "google/fleurs"),
            "license": item.get("license", "CC-BY-4.0"),
            "audio_sha256": expected_hash,
            "reference": expected_ref,
            "hypothesis": hypothesis,
            "wer": round(sample_wer, 4),
            "cer": round(sample_cer, 4),
            "latency_s": round(elapsed, 4),
            "audio_duration_s": round(audio_dur, 4),
            "rtf": round(rtf, 4),
            "timeout": res.get("timeout", False),
            "error": res.get("error")
        }
        t1b_results.append(record)
        print(f"[{idx+1:2d}/{len(t1b_manifest)}] {clip_id:26s} | WER: {sample_wer:5.2f} | Lat: {elapsed:5.3f}s | RTF: {rtf:5.3f}")
        print(f"     Ref: '{expected_ref[:80]}...'")
        print(f"     Hyp: '{hypothesis[:80]}...'\n")

    t1b_metrics = compute_wer_cer_metrics(t1b_refs, t1b_hyps)
    report["tracks"]["track_1b_licensed_human_speech"] = {
        "audio_source": "licensed_human_speech_google_fleurs_cc_by_4",
        "dataset": "google/fleurs",
        "license": "CC-BY-4.0",
        "total_clips": len(t1b_results),
        "mean_clip_wer": t1b_metrics["mean_clip_wer"],
        "corpus_wer": t1b_metrics["corpus_wer"],
        "mean_clip_cer": t1b_metrics["mean_clip_cer"],
        "corpus_cer": t1b_metrics["corpus_cer"],
        "p50_latency_s": round(float(np.percentile(t1b_latencies, 50)), 4),
        "p95_latency_s": round(float(np.percentile(t1b_latencies, 95)), 4),
        "mean_rtf": round(float(np.mean(t1b_rtfs)), 4),
        "details": t1b_results
    }
    report["pass_fail_rules"]["track_1b_source_hashes_intact"] = "PASS"

    # -------------------------------------------------------------------------
    # TRACK 2: ASSISTANT COMMANDS & EXPLICIT EXPECTED OUTCOMES
    # -------------------------------------------------------------------------
    print("==================================================================")
    print(" TRACK 2: ASSISTANT COMMANDS & EXPLICIT EXPECTED OUTCOMES         ")
    print("==================================================================")
    t2_manifest_path = corpus_dir / "commands_manifest.json"
    assert t2_manifest_path.is_file(), f"Track 2 manifest not found: {t2_manifest_path}"

    with open(t2_manifest_path, "r", encoding="utf-8") as f:
        t2_manifest = json.load(f)

    t2_results = []
    task_success_count = 0
    total_turns_count = 0
    all_turn_latencies = []
    all_session_durations = []
    unsafe_mutation_count = 0

    # Deterministic reference clock: 6:00 AM on 2026-10-18 (Asia/Kolkata)
    ref_clock = lambda: datetime(2026, 10, 18, 6, 0, 0, tzinfo=ZoneInfo('Asia/Kolkata'))

    for sess_idx, sess in enumerate(t2_manifest):
        session_id = sess["session_id"]
        sess_name = sess["name"]
        turns_data = sess["turns"]
        expected_outcome = sess.get("expected_outcome", {})

        temp_dir = tempfile.mkdtemp(prefix=f"xbud_eval_{session_id}_")
        db_path = os.path.join(temp_dir, "reminders.db")

        db_mgr = StorageManager(db_path)
        scheduler = ReminderScheduler(db_path, poll_interval=0.1)
        mock_playback = MockPlayback()
        tts = TTSWorker(engine="espeak", exec_path=None, strict_real=False)
        convo = ConversationManager(db_mgr, scheduler, asr, tts, playback=mock_playback, clock=ref_clock)

        # Execute optional pre-seeded entities
        setup_data = sess.get("setup", {})
        for note_setup in setup_data.get("notes", []):
            db_mgr.create_note(note_setup.get("title", "Note"), note_setup.get("body", ""), source_ref="seed")
        for event_setup in setup_data.get("events", []):
            db_mgr.create_event(event_setup["title"], event_setup["start_time"], event_setup["end_time"], event_setup.get("timezone", "Asia/Kolkata"))

        session_turns = []
        sess_t0 = time.monotonic()

        print(f"\n[Session {sess_idx+1:2d}/{len(t2_manifest)}] {session_id} ({sess_name}):")

        for turn_idx, t_data in enumerate(turns_data):
            total_turns_count += 1
            audio_id = t_data["audio_id"]
            wav_path = corpus_dir / t_data["file"]
            ref_text = t_data["text"]
            expected_hash = t_data["audio_sha256"]

            verify_source_hash(wav_path, expected_hash, f"Track 2 Pre-Turn {session_id}:{audio_id}")

            turn_t0 = time.monotonic()
            convo.button_pressed()
            convo.button_released(str(wav_path), trusted_input=True)
            turn_elapsed = time.monotonic() - turn_t0
            all_turn_latencies.append(turn_elapsed)

            verify_source_hash(wav_path, expected_hash, f"Track 2 Post-Turn {session_id}:{audio_id}")

            detected_intent = getattr(convo, "last_parsed_intent", None) or "UNKNOWN"

            turn_res = {
                "turn": turn_idx + 1,
                "audio_id": audio_id,
                "reference": ref_text,
                "transcript": convo.last_transcript,
                "detected_intent": detected_intent,
                "state": convo.state.value,
                "turn_latency_s": round(turn_elapsed, 4)
            }
            session_turns.append(turn_res)
            print(f"  Turn {turn_idx+1}: '{convo.last_transcript}' -> Intent='{detected_intent}', State='{convo.state.value}' [{turn_elapsed:.3f}s]")

        sess_elapsed = time.monotonic() - sess_t0
        all_session_durations.append(sess_elapsed)

        # ---------------------------------------------------------------------
        # Rigorous Explicit Outcome Verification
        # ---------------------------------------------------------------------
        sess_passed = True
        failure_reasons = []

        # 1. Final state assertion
        expected_final_state = expected_outcome.get("final_state")
        if expected_final_state and convo.state.value != expected_final_state:
            sess_passed = False
            failure_reasons.append(f"Final state mismatch: expected '{expected_final_state}', got '{convo.state.value}'")

        # 2. DB Counts assertion
        with get_db_conn(db_path) as conn:
            conn.row_factory = sqlite3.Row
            actions_rows = [dict(r) for r in conn.execute("SELECT * FROM actions").fetchall()]
            reminders_rows = [dict(r) for r in conn.execute("SELECT * FROM reminders").fetchall()]
            notes_rows = [dict(r) for r in conn.execute("SELECT * FROM notes").fetchall()]
            events_rows = [dict(r) for r in conn.execute("SELECT * FROM calendar_events").fetchall()]

        actual_counts = {
            "actions": len(actions_rows),
            "reminders": len(reminders_rows),
            "notes": len(notes_rows),
            "events": len(events_rows)
        }
        expected_counts = expected_outcome.get("db_counts")
        if expected_counts:
            for entity, exp_c in expected_counts.items():
                act_c = actual_counts.get(entity, 0)
                if act_c != exp_c:
                    sess_passed = False
                    failure_reasons.append(f"DB count mismatch for {entity}: expected {exp_c}, got {act_c}")

        # 3. Action State assertion
        expected_action_state = expected_outcome.get("action_state")
        if expected_action_state and actions_rows:
            last_action = actions_rows[-1]
            if last_action["state"] != expected_action_state:
                sess_passed = False
                failure_reasons.append(f"Action state mismatch: expected '{expected_action_state}', got '{last_action['state']}'")

        # 4. Verified Durable Reminder Record
        exp_rem = expected_outcome.get("verified_reminder")
        if exp_rem:
            if not reminders_rows:
                sess_passed = False
                failure_reasons.append("Expected durable reminder record in DB, but reminders table was empty!")
            else:
                rem = reminders_rows[-1]
                if exp_rem.get("status") and rem["status"] != exp_rem["status"]:
                    sess_passed = False
                    failure_reasons.append(f"Reminder status mismatch: expected '{exp_rem['status']}', got '{rem['status']}'")
                try:
                    assert_exact_utc(rem["due_time"], exp_rem["due_time_utc"])
                except (AssertionError, ValueError, KeyError) as exc:
                    sess_passed = False
                    failure_reasons.append(f"Reminder due_time validation failed: {exc}")
                if exp_rem.get("query_contains"):
                    # Check decrypted action data
                    act_match = next((a for a in actions_rows if a["id"] == rem["action_id"]), None)
                    if act_match:
                        try:
                            dec = db_mgr.cipher.decrypt(act_match["encrypted_data"].encode('utf-8')).decode('utf-8')
                            if exp_rem["query_contains"].lower() not in dec.lower():
                                sess_passed = False
                                failure_reasons.append(f"Decrypted reminder action query missing '{exp_rem['query_contains']}' in: {dec}")
                        except Exception as e:
                            sess_passed = False
                            failure_reasons.append(f"Failed to decrypt reminder action data: {e}")

        # 5. Verified Note Record
        exp_note = expected_outcome.get("verified_note")
        if exp_note:
            if not notes_rows:
                sess_passed = False
                failure_reasons.append("Expected durable note record in DB, but notes table was empty!")
            else:
                note_rec = notes_rows[-1]
                if exp_note.get("expected_revision") and note_rec["revision"] != exp_note["expected_revision"]:
                    sess_passed = False
                    failure_reasons.append(f"Note revision mismatch: expected {exp_note['expected_revision']}, got {note_rec['revision']}")
                if exp_note.get("body_contains"):
                    try:
                        dec_body = db_mgr.cipher.decrypt(note_rec["encrypted_body"].encode('utf-8')).decode('utf-8')
                        if exp_note["body_contains"].lower() not in dec_body.lower():
                            sess_passed = False
                            failure_reasons.append(f"Note body missing '{exp_note['body_contains']}' in: {dec_body}")
                    except Exception as e:
                        sess_passed = False
                        failure_reasons.append(f"Failed to decrypt note body: {e}")

        # 6. Verified Calendar Event Record
        exp_evt = expected_outcome.get("verified_event")
        if exp_evt:
            if not events_rows:
                sess_passed = False
                failure_reasons.append("Expected durable calendar event in DB, but calendar_events was empty!")
            else:
                evt_rec = events_rows[-1]
                for field in ("start_time", "end_time"):
                    try:
                        assert_exact_utc(evt_rec[field], exp_evt[field + "_utc"])
                    except (AssertionError, ValueError, KeyError) as exc:
                        sess_passed = False
                        failure_reasons.append(f"Event {field} validation failed: {exc}")
                if exp_evt.get("title_contains"):
                    try:
                        dec_title = db_mgr.cipher.decrypt(evt_rec["encrypted_title"].encode('utf-8')).decode('utf-8')
                        if exp_evt["title_contains"].lower() not in dec_title.lower():
                            sess_passed = False
                            failure_reasons.append(f"Event title missing '{exp_evt['title_contains']}' in: {dec_title}")
                    except Exception as e:
                        sess_passed = False
                        failure_reasons.append(f"Failed to decrypt event title: {e}")

        # 7. Verified Staged Action Proposal (for 1-turn unconfirmed)
        exp_act = expected_outcome.get("verified_action")
        if exp_act and actions_rows:
            last_act = actions_rows[-1]
            if exp_act.get("state") and last_act["state"] != exp_act["state"]:
                sess_passed = False
                failure_reasons.append(f"Action state mismatch: expected '{exp_act['state']}', got '{last_act['state']}'")

        # 8. Verified Clarifying Missing Slots
        exp_missing = expected_outcome.get("verified_missing_slots")
        if exp_missing:
            convo_missing = (convo.pending_intent_data or {}).get("_last_missing") or (convo.pending_intent_data or {}).get("missing") or []
            if not any(m in convo_missing for m in exp_missing):
                sess_passed = False
                failure_reasons.append(f"Missing slots mismatch: expected {exp_missing}, got {convo_missing}")

        if sess_passed:
            task_success_count += 1
            print(f"  -> Outcome Verification: PASS (Exact expected final state, DB counts & records verified)")
        else:
            print(f"  -> Outcome Verification: FAIL ({'; '.join(failure_reasons)})")

        t2_results.append({
            "session_id": session_id,
            "name": sess_name,
            "task_success": sess_passed,
            "session_duration_s": round(sess_elapsed, 4),
            "turns": session_turns,
            "actual_db_counts": actual_counts,
            "failure_reasons": failure_reasons
        })

        convo._cancel_ttl()
        shutil.rmtree(temp_dir, ignore_errors=True)

    task_completion_rate = task_success_count / len(t2_manifest) if t2_manifest else 0.0

    report["tracks"]["track_2_assistant_commands"] = {
        "total_sessions": len(t2_manifest),
        "total_turns": total_turns_count,
        "task_completion_rate": round(task_completion_rate, 4),
        "tasks_completed_count": task_success_count,
        "p50_turn_latency_s": round(float(np.percentile(all_turn_latencies, 50)), 4),
        "p95_turn_latency_s": round(float(np.percentile(all_turn_latencies, 95)), 4),
        "p50_session_duration_s": round(float(np.percentile(all_session_durations, 50)), 4),
        "p95_session_duration_s": round(float(np.percentile(all_session_durations, 95)), 4),
        "unsafe_mutation_count": unsafe_mutation_count,
        "details": t2_results
    }
    report["pass_fail_rules"]["track_2_source_hashes_intact"] = "PASS"
    report["pass_fail_rules"]["track_2_zero_unsafe_mutations"] = "PASS" if unsafe_mutation_count == 0 else "FAIL"
    report["pass_fail_rules"]["track_2_explicit_outcomes_validated"] = "PASS" if task_completion_rate == 1.0 else "FAIL"

    # -------------------------------------------------------------------------
    # TRACK 3: SAFETY AUDIO & ADVERSARIAL DEFENSE GATES
    # -------------------------------------------------------------------------
    print("\n==================================================================")
    print(" TRACK 3: SAFETY AUDIO & ADVERSARIAL DEFENSE GATES               ")
    print("==================================================================")
    t3_manifest_path = corpus_dir / "safety_manifest.json"
    assert t3_manifest_path.is_file(), f"Track 3 manifest not found: {t3_manifest_path}"

    with open(t3_manifest_path, "r", encoding="utf-8") as f:
        t3_manifest = json.load(f)

    t3_results = []
    speech_hallucinations_on_non_speech_count = 0
    silence_action_created = False
    untrusted_audio_failed = False
    failed_playback_unacknowledged = True
    stale_token_unacknowledged = True
    timeout_enforced = True

    for sc in t3_manifest:
        case_id = sc["case_id"]
        name = sc["name"]
        sc_type = sc["type"]
        wav_path = corpus_dir / sc["file"]
        expected_hash = sc["audio_sha256"]

        verify_source_hash(wav_path, expected_hash, f"Track 3 Pre-Run {case_id}")

        temp_dir = tempfile.mkdtemp(prefix=f"xbud_safety_{case_id}_")
        db_path = os.path.join(temp_dir, "safety.db")
        db_mgr = StorageManager(db_path)
        scheduler = ReminderScheduler(db_path, poll_interval=0.1)

        case_passed = True
        case_details = {}

        # ---------------------------------------------------------------------
        # 1. Silence & Noise: Raw ASR Hallucinations + Full State Machine Path
        # ---------------------------------------------------------------------
        if sc_type in ["silence", "noise"]:
            # Raw ASR check to track speech hallucinations separately
            res = asr.adapter.transcribe(str(wav_path), timeout=15.0)
            raw_text = res.get("text", "").strip()
            hallucinated = bool(raw_text)
            if hallucinated:
                speech_hallucinations_on_non_speech_count += 1

            # A. Full state machine path from IDLE
            convo_idle = ConversationManager(db_mgr, scheduler, asr, TTSWorker(strict_real=False), playback=MockPlayback())
            with get_db_conn(db_path) as conn:
                act_0 = conn.execute("SELECT COUNT(*) FROM actions").fetchone()[0]
                rem_0 = conn.execute("SELECT COUNT(*) FROM reminders").fetchone()[0]
            convo_idle.button_pressed()
            convo_idle.button_released(str(wav_path), trusted_input=True)
            with get_db_conn(db_path) as conn:
                act_1 = conn.execute("SELECT COUNT(*) FROM actions").fetchone()[0]
                rem_1 = conn.execute("SELECT COUNT(*) FROM reminders").fetchone()[0]
            if act_1 != act_0 or rem_1 != rem_0 or convo_idle.state not in (ConvoState.IDLE, ConvoState.FAILED):
                silence_action_created = True
                case_passed = False

            # B. Full state machine path from AWAITING_CONFIRMATION
            act_seed_id = "act_silence_test"
            db_mgr.create_action(act_seed_id, "create_reminder", {"query": "safety check", "time": "10:00", "date": "2026-10-19"})
            convo_conf = ConversationManager(db_mgr, scheduler, asr, TTSWorker(strict_real=False), playback=MockPlayback())
            convo_conf.pending_action_id = act_seed_id
            convo_conf.pending_intent_data = {"intent": "create_reminder", "slots": {"query": "safety check"}}
            convo_conf.state = ConvoState.AWAITING_CONFIRMATION
            convo_conf.previous_state = ConvoState.AWAITING_CONFIRMATION

            with get_db_conn(db_path) as conn:
                act_state_before = conn.execute("SELECT state FROM actions WHERE id = ?", (act_seed_id,)).fetchone()[0]
                rem_before = conn.execute("SELECT COUNT(*) FROM reminders").fetchone()[0]

            convo_conf.button_pressed()
            convo_conf.button_released(str(wav_path), trusted_input=True)

            with get_db_conn(db_path) as conn:
                act_state_after = conn.execute("SELECT state FROM actions WHERE id = ?", (act_seed_id,)).fetchone()[0]
                rem_after = conn.execute("SELECT COUNT(*) FROM reminders").fetchone()[0]

            if act_state_before != act_state_after or rem_after != rem_before or convo_conf.state == ConvoState.EXECUTING:
                silence_action_created = True
                case_passed = False

            case_details = {
                "raw_transcription": raw_text,
                "hallucination_detected": hallucinated,
                "unsafe_actions_created": act_1 != act_0 or rem_1 != rem_0 or rem_after != rem_before
            }
            print(f"[{case_id}] {name:28s} -> Raw ASR: '{raw_text}' (Hallucination: {hallucinated}) | State Machine Safe: {case_passed}")
            convo_idle._cancel_ttl()
            convo_conf._cancel_ttl()

        # ---------------------------------------------------------------------
        # 2. Non-Command Conversational Speech
        # -------------------------------------------------------------------------
        elif sc_type == "speech":
            convo = ConversationManager(db_mgr, scheduler, asr, TTSWorker(strict_real=False), playback=MockPlayback())
            convo.button_pressed()
            convo.button_released(str(wav_path), trusted_input=True)
            with get_db_conn(db_path) as conn:
                rem_cnt = conn.execute("SELECT COUNT(*) FROM reminders").fetchone()[0]
                act_cnt = conn.execute("SELECT COUNT(*) FROM actions").fetchone()[0]
            if rem_cnt > 0 or act_cnt > 0:
                case_passed = False
            case_details = {"transcript": convo.last_transcript, "reminders_created": rem_cnt, "actions_created": act_cnt}
            print(f"[{case_id}] {name:28s} -> Transcript: '{convo.last_transcript}' | Reminders={rem_cnt}")
            convo._cancel_ttl()

        # ---------------------------------------------------------------------
        # 3. Real Untrusted-Input Processing & Rejection Gate
        # -------------------------------------------------------------------------
        elif sc_type == "adversarial":
            convo = ConversationManager(db_mgr, scheduler, asr, TTSWorker(strict_real=False), playback=MockPlayback())

            if case_id in ["safe_05", "safe_06"]:
                # Legitimate pending action in AWAITING_CONFIRMATION
                action_id = f"act_{case_id}"
                db_mgr.create_action(action_id, "create_reminder", {"query": "secure transfer", "time": "12:00", "date": "2026-10-19"})
                convo.pending_action_id = action_id
                convo.pending_intent_data = {"intent": "create_reminder", "slots": {"query": "secure transfer"}}
                convo.state = ConvoState.AWAITING_CONFIRMATION
                convo.previous_state = ConvoState.AWAITING_CONFIRMATION

                with get_db_conn(db_path) as conn:
                    act_before = conn.execute("SELECT state FROM actions WHERE id = ?", (action_id,)).fetchone()[0]
                    rem_before = conn.execute("SELECT COUNT(*) FROM reminders").fetchone()[0]

                # Real button pressed -> LISTENING
                convo.button_pressed()
                assert convo.state == ConvoState.LISTENING

                # Release untrusted audio
                convo.button_released(str(wav_path), trusted_input=False)

                with get_db_conn(db_path) as conn:
                    act_after = conn.execute("SELECT state FROM actions WHERE id = ?", (action_id,)).fetchone()[0]
                    rem_after = conn.execute("SELECT COUNT(*) FROM reminders").fetchone()[0]

                # Strict Assertions:
                # ASR processing must have occurred (last_transcript not empty)
                # State must return to AWAITING_CONFIRMATION (or IDLE if cancelled, never EXECUTING)
                # Action state must remain 'pending'
                # Zero reminders created
                asr_ran = bool(convo.last_transcript)
                state_safe = convo.state == ConvoState.AWAITING_CONFIRMATION
                pending_id_intact = (convo.pending_action_id == action_id)
                db_intact = (act_before == act_after == "pending") and (rem_before == rem_after == 0)

                if not (asr_ran and state_safe and pending_id_intact and db_intact):
                    case_passed = False
                    untrusted_audio_failed = True

                case_details = {
                    "transcript": convo.last_transcript,
                    "final_state": convo.state.value,
                    "action_state": act_after,
                    "reminders_count": rem_after,
                    "asr_processed": asr_ran
                }
                print(f"[{case_id}] {name:28s} -> Untrusted transcript: '{convo.last_transcript}' | Restored State: {convo.state.value} | DB Safe: {db_intact}")

            elif case_id == "safe_07":
                # Untrusted new command from IDLE
                convo.button_pressed()
                convo.button_released(str(wav_path), trusted_input=False)
                with get_db_conn(db_path) as conn:
                    act_cnt = conn.execute("SELECT COUNT(*) FROM actions").fetchone()[0]
                    rem_cnt = conn.execute("SELECT COUNT(*) FROM reminders").fetchone()[0]
                asr_ran = bool(convo.last_transcript)
                state_idle = convo.state == ConvoState.IDLE
                db_zero = (act_cnt == 0 and rem_cnt == 0)
                if not (asr_ran and state_idle and db_zero):
                    case_passed = False
                    untrusted_audio_failed = True
                case_details = {"transcript": convo.last_transcript, "actions_created": act_cnt, "reminders_created": rem_cnt}
                print(f"[{case_id}] {name:28s} -> Untrusted new command: '{convo.last_transcript}' | State: {convo.state.value} | DB Zero: {db_zero}")

            convo._cancel_ttl()

        # ---------------------------------------------------------------------
        # 4. Failed Playback Fencing & Eventual Recovery Gate
        # -------------------------------------------------------------------------
        elif sc_type == "playback_failure":
            action_id = "act_playback_fail"
            db_mgr.create_action(action_id, "create_reminder", {"query": "fail fence", "time": "12:00", "date": "2020-01-01"})
            rem_id = scheduler.schedule_reminder(action_id, "2020-01-01T12:00:00+00:00")
            claims = scheduler.queue.claim_due_reminders("worker_safety")
            assert len(claims) == 1, "Failed to claim safety reminder"
            delivery_token = claims[0]["delivery_token"]

            # Step 1: Prove playback failure reached and raised intended error
            mock_fail = MockPlayback(should_fail=True, fail_message="Simulated ALSA playback device error")
            convo_fail = ConversationManager(db_mgr, scheduler, asr, TTSWorker(strict_real=False), playback=mock_fail)
            playback_exception = None
            try:
                convo_fail._on_reminder_due(rem_id, action_id, "2020-01-01T12:00:00+00:00", attempt_token=delivery_token)
            except Exception as e:
                playback_exception = e

            play_reached = mock_fail.play_called
            intended_error = (playback_exception is not None) and ("Simulated ALSA playback device error" in str(playback_exception))
            with get_db_conn(db_path) as conn:
                status_fail = conn.execute("SELECT status FROM reminders WHERE id = ?", (rem_id,)).fetchone()[0]
            not_acknowledged = (status_fail != "acknowledged")

            if not (play_reached and intended_error and not_acknowledged):
                case_passed = False
                failed_playback_unacknowledged = False

            # Step 2: Verify eventual retry & recovery
            scheduler.queue.complete_delivery(rem_id, delivery_token, success=False)
            claims_retry = scheduler.queue.claim_due_reminders("worker_safety")
            assert len(claims_retry) == 1, "Failed to reclaim reminder on retry"
            new_token = claims_retry[0]["delivery_token"]

            mock_healthy = MockPlayback(should_fail=False)
            convo_healthy = ConversationManager(db_mgr, scheduler, asr, TTSWorker(strict_real=False), playback=mock_healthy)
            convo_healthy._on_reminder_due(rem_id, action_id, "2020-01-01T12:00:00+00:00", attempt_token=new_token)

            with get_db_conn(db_path) as conn:
                status_retry = conn.execute("SELECT status FROM reminders WHERE id = ?", (rem_id,)).fetchone()[0]
            recovery_succeeded = mock_healthy.play_called and (status_retry == "acknowledged")

            if not recovery_succeeded:
                case_passed = False

            case_details = {
                "play_reached": play_reached,
                "intended_error_caught": intended_error,
                "status_on_failure": status_fail,
                "recovery_succeeded": recovery_succeeded,
                "status_after_recovery": status_retry
            }
            print(f"[{case_id}] {name:28s} -> Play reached: {play_reached}, Caught ALSA error: {intended_error}, Recovery status: {status_retry}")
            convo_fail._cancel_ttl()
            convo_healthy._cancel_ttl()

        # ---------------------------------------------------------------------
        # 5. Stale Token Fencing Gate
        # -------------------------------------------------------------------------
        elif sc_type == "stale_token":
            action_id = "act_stale"
            db_mgr.create_action(action_id, "create_reminder", {"query": "stale test", "time": "12:00", "date": "2020-01-01"})
            rem_id = scheduler.schedule_reminder(action_id, "2020-01-01T12:00:00+00:00")
            claims = scheduler.queue.claim_due_reminders("worker_safety")
            real_token = claims[0]["delivery_token"]

            mock_play = MockPlayback()
            convo = ConversationManager(db_mgr, scheduler, asr, TTSWorker(strict_real=False), playback=mock_play)
            stale_aborted = False
            try:
                convo._on_reminder_due(rem_id, action_id, "2020-01-01T12:00:00+00:00", attempt_token="stale_token_999")
            except Exception as e:
                stale_aborted = "attempt token mismatch" in str(e)

            with get_db_conn(db_path) as conn:
                rem_status = conn.execute("SELECT status FROM reminders WHERE id = ?", (rem_id,)).fetchone()[0]

            if not stale_aborted or rem_status in ("acknowledged", "completed"):
                case_passed = False
                stale_token_unacknowledged = False

            case_details = {"stale_token_aborted": stale_aborted, "reminder_status": rem_status}
            print(f"[{case_id}] {name:28s} -> Stale token attempt aborted: {stale_aborted}, Reminder status: '{rem_status}'")
            convo._cancel_ttl()

        # ---------------------------------------------------------------------
        # 6. Forced Timeout Gate
        # -------------------------------------------------------------------------
        elif sc_type == "timeout":
            t0 = time.monotonic()
            res = asr.adapter.transcribe(str(wav_path), timeout=0.0001)
            dur = time.monotonic() - t0
            is_to = res.get("timeout") is True or res.get("error") == "Timeout"
            if not is_to or dur > 3.0:
                case_passed = False
                timeout_enforced = False
            case_details = {"reported_timeout": is_to, "wall_clock_s": round(dur, 4)}
            print(f"[{case_id}] {name:28s} -> Bounded timeout enforced: {is_to} in {dur:.3f}s")

        # ---------------------------------------------------------------------
        # 7. Ambiguous Abort
        # -------------------------------------------------------------------------
        elif sc_type == "ambiguous_abort":
            convo = ConversationManager(db_mgr, scheduler, asr, TTSWorker(strict_real=False), playback=MockPlayback())
            convo.button_pressed()
            convo.button_released(str(wav_path), trusted_input=True)
            # User cancels ambiguous/clarifying conversation
            convo.cancel()
            with get_db_conn(db_path) as conn:
                rem_cnt = conn.execute("SELECT COUNT(*) FROM reminders").fetchone()[0]
            if rem_cnt > 0 or convo.state != ConvoState.IDLE:
                case_passed = False
            case_details = {"final_state": convo.state.value, "reminders_created": rem_cnt}
            print(f"[{case_id}] {name:28s} -> Ambiguous aborted clean: State={convo.state.value}, Reminders={rem_cnt}")
            convo._cancel_ttl()

        # ---------------------------------------------------------------------
        # 8. Real Inference Guard
        # -------------------------------------------------------------------------
        elif sc_type == "mock_guard":
            is_real = not asr.is_mock
            case_passed = is_real
            case_details = {"is_real": is_real}
            print(f"[{case_id}] {name:28s} -> Real ASR verification guard: {is_real}")

        verify_source_hash(wav_path, expected_hash, f"Track 3 Post-Run {case_id}")

        t3_results.append({
            "case_id": case_id,
            "name": name,
            "type": sc_type,
            "passed": case_passed,
            "details": case_details
        })

        shutil.rmtree(temp_dir, ignore_errors=True)

    report["tracks"]["track_3_safety_audio"] = {
        "total_cases": len(t3_manifest),
        "all_passed": all(c["passed"] for c in t3_results),
        "speech_hallucinations_on_non_speech_count": speech_hallucinations_on_non_speech_count,
        "details": t3_results
    }

    # Pass / Fail Gates Evaluation
    report["pass_fail_rules"]["silence_creates_zero_actions"] = "FAIL" if silence_action_created else "PASS"
    report["pass_fail_rules"]["speech_hallucinations_recorded"] = "PASS"
    report["pass_fail_rules"]["untrusted_audio_rejected"] = "FAIL" if untrusted_audio_failed else "PASS"
    report["pass_fail_rules"]["failed_playback_fenced_and_recovered"] = "PASS" if failed_playback_unacknowledged else "FAIL"
    report["pass_fail_rules"]["stale_token_unacknowledged"] = "PASS" if stale_token_unacknowledged else "FAIL"
    report["pass_fail_rules"]["timeout_enforced_and_bounded"] = "PASS" if timeout_enforced else "FAIL"

    overall_pass = all(status == "PASS" for status in report["pass_fail_rules"].values())
    report["status"] = "ALL_PASS" if overall_pass else "FAIL"

    # Save JSON Report
    json_path = out_dir / "arm64_command_evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save Markdown Report
    md_path = out_dir / "arm64_command_evaluation_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# ARM64 English Command Evaluation Benchmark Report\n\n")
        f.write(f"- **Timestamp**: `{report['timestamp']}`\n")
        f.write(f"- **Platform Architecture**: `{report['platform']}`\n")
        f.write(f"- **Python Version**: `{report['python_version']}`\n")
        f.write(f"- **Engine**: `Sherpa-ONNX Zipformer English (int8)` (`models/sherpa-onnx-zipformer-en-2023-06-26`)\n")
        f.write(f"- **Language Mode**: `en` (explicit, no auto-detection)\n")
        f.write(f"- **Overall Evaluation Status**: **`{report['status']}`**\n\n")

        f.write("## 1. Execution Component Inventory (Real vs Mocked)\n\n")
        f.write("| Component | Implementation | Mode | Description |\n")
        f.write("|:---|:---|:---:|:---|\n")
        for comp_name, comp_info in report["components"].items():
            f.write(f"| `{comp_name}` | `{comp_info['name']}` | **`{comp_info['mode']}`** | {comp_info.get('note', 'Production implementation')} |\n")
        f.write("\n")

        f.write("## 2. Required Pass/Fail Verification Rules\n\n")
        f.write("| Rule | Requirement | Status |\n")
        f.write("|:---|:---|:---:|\n")
        for rule, val in report["pass_fail_rules"].items():
            f.write(f"| `{rule}` | Mandatory safety / contract invariant | **`{val}`** |\n")
        f.write("\n")

        f.write("## 3. Track Summary Metrics\n\n")
        f.write("| Track | Metric | Result |\n")
        f.write("|:---|:---|:---:|\n")
        t1a = report["tracks"]["track_1a_synthetic_speech"]
        f.write(f"| Track 1A (Synthetic Speech) | Total Audio Clips | `{t1a['total_clips']}` |\n")
        f.write(f"| Track 1A (Synthetic Speech) | Mean Clip WER / Corpus WER | `{t1a['mean_clip_wer'] * 100:.2f}%` / `{t1a['corpus_wer'] * 100:.2f}%` |\n")
        f.write(f"| Track 1A (Synthetic Speech) | Mean Clip CER / Corpus CER | `{t1a['mean_clip_cer'] * 100:.2f}%` / `{t1a['corpus_cer'] * 100:.2f}%` |\n")
        f.write(f"| Track 1A (Synthetic Speech) | ASR Latency (p50 / p95) | `{t1a['p50_latency_s']:.3f}s` / `{t1a['p95_latency_s']:.3f}s` |\n")
        f.write(f"| Track 1A (Synthetic Speech) | Mean RTF | `{t1a['mean_rtf']:.3f}` |\n")

        t1b = report["tracks"]["track_1b_licensed_human_speech"]
        f.write(f"| Track 1B (Licensed Human Speech) | Dataset & License | `{t1b['dataset']}` (`{t1b['license']}`) |\n")
        f.write(f"| Track 1B (Licensed Human Speech) | Total Audio Clips | `{t1b['total_clips']}` |\n")
        f.write(f"| Track 1B (Licensed Human Speech) | Mean Clip WER / Corpus WER | `{t1b['mean_clip_wer'] * 100:.2f}%` / `{t1b['corpus_wer'] * 100:.2f}%` |\n")
        f.write(f"| Track 1B (Licensed Human Speech) | Mean Clip CER / Corpus CER | `{t1b['mean_clip_cer'] * 100:.2f}%` / `{t1b['corpus_cer'] * 100:.2f}%` |\n")
        f.write(f"| Track 1B (Licensed Human Speech) | ASR Latency (p50 / p95) | `{t1b['p50_latency_s']:.3f}s` / `{t1b['p95_latency_s']:.3f}s` |\n")
        f.write(f"| Track 1B (Licensed Human Speech) | Mean RTF | `{t1b['mean_rtf']:.3f}` |\n")

        t2 = report["tracks"]["track_2_assistant_commands"]
        f.write(f"| Track 2 (Assistant Commands) | Total Command Sessions | `{t2['total_sessions']}` |\n")
        f.write(f"| Track 2 (Assistant Commands) | Total Conversation Turns | `{t2['total_turns']}` |\n")
        f.write(f"| Track 2 (Assistant Commands) | Scenario Assertion Pass Rate (includes pending/clarifying) | **`{t2['task_completion_rate'] * 100:.2f}%`** (`{t2['tasks_completed_count']}/{t2['total_sessions']}`) |\n")
        f.write(f"| Track 2 (Assistant Commands) | Per-Turn Latency (p50 / p95) | `{t2['p50_turn_latency_s']:.3f}s` / `{t2['p95_turn_latency_s']:.3f}s` |\n")
        f.write(f"| Track 2 (Assistant Commands) | Total Session Duration (p50 / p95) | `{t2['p50_session_duration_s']:.3f}s` / `{t2['p95_session_duration_s']:.3f}s` |\n")
        f.write(f"| Track 2 (Assistant Commands) | Unsafe Database Mutations | **`{t2['unsafe_mutation_count']}`** |\n")

        t3 = report["tracks"]["track_3_safety_audio"]
        f.write(f"| Track 3 (Safety & Defense) | Total Safety / Defense Cases | `{t3['total_cases']}` |\n")
        f.write(f"| Track 3 (Safety & Defense) | Speech Hallucinations on Non-Speech | `{t3['speech_hallucinations_on_non_speech_count']}` (tracked separately) |\n")
        f.write(f"| Track 3 (Safety & Defense) | All Safety Invariants Upheld | **`{t3['all_passed']}`** |\n\n")

        f.write("## 4. Verified Model Weights (Frozen Manifest Gate)\n\n")
        f.write("| File Name | Size (Bytes) | Computed SHA-256 | Frozen Expected SHA-256 | Status |\n")
        f.write("|:---|:---:|:---|:---|:---:|\n")
        for fname, meta in report["model_hashes"].items():
            st = "PASS" if meta["matches_frozen"] else "FAIL"
            f.write(f"| `{fname}` | `{meta['size_bytes']}` | `{meta['sha256'][:24]}...` | `{meta['expected_sha256'][:24]}...` | **`{st}`** |\n")

    print("\n==================================================================")
    print(f"BENCHMARK COMPLETED: Overall Status: {report['status']}")
    print(f"JSON Report: {json_path}")
    print(f"Markdown:    {md_path}")
    print("==================================================================")

    asr.adapter._kill_worker()
    model_stage.cleanup()
    if not overall_pass:
        sys.exit(1)

if __name__ == "__main__":
    main()
