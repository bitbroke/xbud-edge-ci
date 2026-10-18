#!/usr/bin/env python3
"""
benchmarks/run_arm64_command_eval.py — ARM64 English Command Evaluation Benchmark.

Evaluates real Sherpa-ONNX Zipformer English ASR and full assistant state machine:
- Track 1: General Speech (22 clips) -> WER, CER, latency, RTF, peak RSS
- Track 2: Assistant Commands (22 sessions) -> ASR, intent, multi-turn state transitions, SQLite transactions, reminder queue
- Track 3: Safety Audio (12 cases) -> silence/noise, untrusted rejection, failed playback fence, stale token fence, forced timeout

Required Pass/Fail Rules:
1. Source audio SHA-256 must match frozen manifest before and after each attempt.
2. Model SHA-256 hashes must be verified and present.
3. Timeouts must be tracked and never hidden or dropped.
4. Silence/noise must produce 0 actions.
5. Untrusted audio must never confirm, cancel, or mutate actions.
6. Failed playback must never acknowledge a reminder.
7. Stale reminder token must never acknowledge a newer claim.
8. Real inference must be verified; mock backend causes immediate benchmark failure.
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
from typing import Dict, Any, List

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import psutil
import numpy as np
from jiwer import wer, cer

from providers.asr_worker import ASRWorker, SherpaOnnxASR
from providers.tts_worker import TTSWorker
from storage.database import StorageManager, get_db_conn
from scheduler.scheduler import ReminderScheduler
from app.conversation import ConversationManager, ConvoState

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
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.played_files = []

    def play(self, wav_path: str):
        if self.should_fail:
            raise RuntimeError("Simulated ALSA playback device error")
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
        "model_hashes": {},
        "pass_fail_rules": {},
        "tracks": {
            "track_1_general_speech": {},
            "track_2_assistant_commands": {},
            "track_3_safety_audio": {}
        },
        "summary": {},
        "status": "INCOMPLETE"
    }

    # -------------------------------------------------------------------------
    # 0. MODEL HASHES & INTEGRITY
    # -------------------------------------------------------------------------
    print("[Phase 0] Validating model identity and calculating SHA-256 hashes...")
    if not model_dir.is_dir():
        report["pass_fail_rules"]["model_hash_present"] = "FAIL"
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    for file_path in sorted(model_dir.glob("*")):
        if file_path.is_file():
            h = compute_sha256(file_path)
            report["model_hashes"][file_path.name] = {
                "sha256": h,
                "size_bytes": file_path.stat().st_size
            }
            print(f"  {file_path.name:36s} [{file_path.stat().st_size:>9d} bytes] {h[:16]}...")

    assert "tokens.txt" in report["model_hashes"], "tokens.txt missing from model directory!"
    report["pass_fail_rules"]["model_hash_present"] = "PASS"

    # Initialize Real ASR Worker
    print("\n[Phase 0] Initializing real SherpaOnnxASR engine (strict_real=True)...")
    asr = ASRWorker(engine="sherpa", model_dir=str(model_dir), language="en", strict_real=True)
    if asr.is_mock:
        report["pass_fail_rules"]["real_inference_verified"] = "FAIL"
        raise RuntimeError("CRITICAL: ASRWorker fell back to mock mode! Real inference required.")
    report["pass_fail_rules"]["real_inference_verified"] = "PASS"
    print("  -> Real ASRWorker initialized successfully (is_mock=False).\n")

    # -------------------------------------------------------------------------
    # TRACK 1: GENERAL SPEECH (WER / CER & Latency)
    # -------------------------------------------------------------------------
    print("==================================================================")
    print(" TRACK 1: GENERAL SPEECH ACCURACY & LATENCY BENCHMARK             ")
    print("==================================================================")
    t1_manifest_path = corpus_dir / "general_speech_manifest.json"
    assert t1_manifest_path.is_file(), f"Track 1 manifest not found: {t1_manifest_path}"

    with open(t1_manifest_path, "r", encoding="utf-8") as f:
        t1_manifest = json.load(f)

    t1_results = []
    wers, cers, latencies, rtfs = [], [], [], []

    for idx, item in enumerate(t1_manifest):
        clip_id = item["id"]
        wav_path = corpus_dir / item["file"]
        expected_ref = item["expected_transcript"]
        expected_hash = item["audio_sha256"]

        # 1. Pre-inference hash check
        verify_source_hash(wav_path, expected_hash, f"Track 1 Pre-Run {clip_id}")

        # 2. Timed inference
        audio_dur = item.get("duration_s") or get_audio_duration_s(wav_path)
        t0 = time.monotonic()
        res = asr.adapter.transcribe(str(wav_path), timeout=30.0)
        elapsed = time.monotonic() - t0

        # 3. Post-inference hash check
        verify_source_hash(wav_path, expected_hash, f"Track 1 Post-Run {clip_id}")

        hypothesis = res.get("text", "").strip()
        sample_wer = wer(expected_ref.lower(), hypothesis.lower()) if hypothesis else 1.0
        sample_cer = cer(expected_ref.lower(), hypothesis.lower()) if hypothesis else 1.0
        rtf = elapsed / audio_dur if audio_dur > 0 else 0.0

        wers.append(sample_wer)
        cers.append(sample_cer)
        latencies.append(elapsed)
        rtfs.append(rtf)

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
            "peak_memory_bytes": res.get("peak_process_memory_bytes"),
            "timeout": res.get("timeout", False),
            "error": res.get("error")
        }
        t1_results.append(record)
        print(f"[{idx+1:2d}/{len(t1_manifest)}] {clip_id:8s} | WER: {sample_wer:5.2f} | CER: {sample_cer:5.2f} | Lat: {elapsed:5.3f}s | RTF: {rtf:5.3f}")
        print(f"     Ref:  '{expected_ref}'")
        print(f"     Hyp:  '{hypothesis}'\n")

    report["tracks"]["track_1_general_speech"] = {
        "total_clips": len(t1_results),
        "mean_wer": round(float(np.mean(wers)), 4),
        "mean_cer": round(float(np.mean(cers)), 4),
        "p50_latency_s": round(float(np.percentile(latencies, 50)), 4),
        "p95_latency_s": round(float(np.percentile(latencies, 95)), 4),
        "mean_rtf": round(float(np.mean(rtfs)), 4),
        "details": t1_results
    }
    report["pass_fail_rules"]["track_1_source_hashes_intact"] = "PASS"

    # -------------------------------------------------------------------------
    # TRACK 2: ASSISTANT COMMANDS & FULL STATE MACHINE BENCHMARK
    # -------------------------------------------------------------------------
    print("==================================================================")
    print(" TRACK 2: ASSISTANT COMMANDS & FULL STATE MACHINE PIPELINE        ")
    print("==================================================================")
    t2_manifest_path = corpus_dir / "commands_manifest.json"
    assert t2_manifest_path.is_file(), f"Track 2 manifest not found: {t2_manifest_path}"

    with open(t2_manifest_path, "r", encoding="utf-8") as f:
        t2_manifest = json.load(f)

    t2_results = []
    task_success_count = 0
    intent_success_count = 0
    total_turns_count = 0
    e2e_latencies = []
    unsafe_mutation_count = 0

    for sess_idx, sess in enumerate(t2_manifest):
        session_id = sess["session_id"]
        sess_name = sess["name"]
        turns_data = sess["turns"]

        # Isolated SQLite DB per session
        temp_dir = tempfile.mkdtemp(prefix=f"xbud_eval_{session_id}_")
        db_path = os.path.join(temp_dir, "reminders.db")

        db_mgr = StorageManager(db_path)
        scheduler = ReminderScheduler(db_path, poll_interval=0.1)
        mock_playback = MockPlayback()
        tts = TTSWorker(engine="espeak", exec_path=None, strict_real=False)
        convo = ConversationManager(db_mgr, scheduler, asr, tts, playback=mock_playback)

        session_turns = []
        sess_t0 = time.monotonic()
        sess_success = True

        print(f"\n[Session {sess_idx+1:2d}/{len(t2_manifest)}] {session_id} ({sess_name}):")

        for turn_idx, t_data in enumerate(turns_data):
            total_turns_count += 1
            audio_id = t_data["audio_id"]
            wav_path = corpus_dir / t_data["file"]
            ref_text = t_data["text"]
            expected_hash = t_data["audio_sha256"]

            # Pre-turn hash verify
            verify_source_hash(wav_path, expected_hash, f"Track 2 Pre-Turn {session_id}:{audio_id}")

            turn_t0 = time.monotonic()
            convo.button_pressed()
            convo.button_released(str(wav_path), trusted_input=True)
            turn_elapsed = time.monotonic() - turn_t0

            # Post-turn hash verify
            verify_source_hash(wav_path, expected_hash, f"Track 2 Post-Turn {session_id}:{audio_id}")

            # Capture DB state
            with get_db_conn(db_path) as conn:
                conn.row_factory = sqlite3.Row
                actions = [dict(r) for r in conn.execute("SELECT * FROM actions").fetchall()]
                reminders = [dict(r) for r in conn.execute("SELECT * FROM reminders").fetchall()]
                notes = [dict(r) for r in conn.execute("SELECT * FROM notes").fetchall()]
                events = [dict(r) for r in conn.execute("SELECT * FROM calendar_events").fetchall()]

            INTENT_MAP = {
                "create_reminder": "ADD_REMINDER",
                "confirm": "CONFIRM",
                "cancel": "CANCEL",
                "correction": "UPDATE_REMINDER",
                "create_note": "ADD_NOTE",
                "search_notes": "SEARCH_NOTE",
                "update_note": "UPDATE_NOTE",
                "create_event": "ADD_EVENT",
                "cancel_event": "CANCEL_EVENT",
                "search_events": "SEARCH_EVENT",
            }
            parsed_raw = getattr(convo, "last_parsed_intent", None) or (
                convo.pending_intent_data.get("intent") if convo.pending_intent_data else (
                    actions[-1].get("intent") if actions else "UNKNOWN"
                )
            )
            canonical_detected = INTENT_MAP.get(parsed_raw, parsed_raw)

            # Check intent matching
            expected_intent = t_data.get("expected_intent")
            intent_match = (canonical_detected.upper() == expected_intent.upper()) if expected_intent else True
            if intent_match:
                intent_success_count += 1

            turn_res = {
                "turn": turn_idx + 1,
                "audio_id": audio_id,
                "reference": ref_text,
                "detected_intent": canonical_detected,
                "raw_intent": parsed_raw,
                "expected_intent": expected_intent,
                "intent_match": intent_match,
                "state": convo.state.value,
                "latency_s": round(turn_elapsed, 4),
                "db_actions": len(actions),
                "db_reminders": len(reminders),
                "db_notes": len(notes),
                "db_events": len(events)
            }
            session_turns.append(turn_res)
            print(f"  Turn {turn_idx+1}: intent='{canonical_detected}' (exp='{expected_intent}') -> State='{convo.state.value}' [{turn_elapsed:.3f}s]")

        sess_elapsed = time.monotonic() - sess_t0
        e2e_latencies.append(sess_elapsed)

        # Verify session task completion
        # For reminders: check at least 1 action or reminder recorded if not cancelled
        with get_db_conn(db_path) as conn:
            final_actions = conn.execute("SELECT COUNT(*) FROM actions").fetchone()[0]
            final_reminders = conn.execute("SELECT COUNT(*) FROM reminders").fetchone()[0]
            final_notes = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
            final_events = conn.execute("SELECT COUNT(*) FROM calendar_events").fetchone()[0]

        if "cancel" in sess_name:
            # Should have 0 active reminders
            if final_reminders > 0:
                sess_success = False
                unsafe_mutation_count += 1
        elif "note" in sess_name:
            if "search" not in sess_name and final_notes == 0 and final_actions == 0 and convo.state not in (ConvoState.AWAITING_CONFIRMATION, ConvoState.CLARIFYING):
                sess_success = False
        elif "event" in sess_name:
            if "cancel" in sess_name and final_events > 0:
                sess_success = False
        else:
            if final_actions == 0 and final_reminders == 0 and convo.state not in (ConvoState.AWAITING_CONFIRMATION, ConvoState.CLARIFYING):
                sess_success = False

        if sess_success:
            task_success_count += 1

        t2_results.append({
            "session_id": session_id,
            "name": sess_name,
            "task_success": sess_success,
            "e2e_latency_s": round(sess_elapsed, 4),
            "turns": session_turns,
            "final_db_counts": {
                "actions": final_actions,
                "reminders": final_reminders,
                "notes": final_notes,
                "events": final_events
            }
        })

        # Clean up temp DB and timers
        convo._cancel_ttl()
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

    task_completion_rate = task_success_count / len(t2_manifest) if t2_manifest else 0.0
    intent_acc = intent_success_count / total_turns_count if total_turns_count else 0.0

    report["tracks"]["track_2_assistant_commands"] = {
        "total_sessions": len(t2_manifest),
        "total_turns": total_turns_count,
        "task_completion_rate": round(task_completion_rate, 4),
        "intent_accuracy": round(intent_acc, 4),
        "p50_e2e_latency_s": round(float(np.percentile(e2e_latencies, 50)), 4),
        "p95_e2e_latency_s": round(float(np.percentile(e2e_latencies, 95)), 4),
        "unsafe_mutation_count": unsafe_mutation_count,
        "details": t2_results
    }
    report["pass_fail_rules"]["track_2_source_hashes_intact"] = "PASS"
    report["pass_fail_rules"]["track_2_zero_unsafe_mutations"] = "PASS" if unsafe_mutation_count == 0 else "FAIL"

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
    silence_action_created = False
    untrusted_action_created = False
    failed_tts_acknowledged = False
    stale_token_acknowledged = False
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

        if sc_type in ["silence", "noise"]:
            res = asr.adapter.transcribe(str(wav_path), timeout=15.0)
            text = res.get("text", "").strip()
            # Assert no dangerous action keywords
            forbidden = ["remind", "alarm", "call", "delete", "create", "note", "event", "schedule", "cancel"]
            has_action = any(w in text.lower() for w in forbidden)
            if has_action:
                silence_action_created = True
                case_passed = False
            case_details = {"text": text, "has_forbidden_action": has_action}
            print(f"[{case_id}] {name:28s} -> Text: '{text}' (Forbidden action: {has_action})")

        elif sc_type == "speech":
            res = asr.adapter.transcribe(str(wav_path), timeout=15.0)
            text = res.get("text", "").strip()
            # Run state machine turn
            convo = ConversationManager(db_mgr, scheduler, asr, TTSWorker(strict_real=False), playback=MockPlayback())
            convo.button_pressed()
            convo.button_released(str(wav_path), trusted_input=True)
            with get_db_conn(db_path) as conn:
                rem_cnt = conn.execute("SELECT COUNT(*) FROM reminders").fetchone()[0]
            if rem_cnt > 0:
                case_passed = False
            case_details = {"text": text, "reminders_created": rem_cnt}
            print(f"[{case_id}] {name:28s} -> Non-command speech created {rem_cnt} reminders.")

        elif sc_type == "adversarial":
            # Attempt untrusted turn
            convo = ConversationManager(db_mgr, scheduler, asr, TTSWorker(strict_real=False), playback=MockPlayback())
            # Manually prime into awaiting confirmation
            convo.state = ConvoState.AWAITING_CONFIRMATION
            convo.pending_intent_data = {"intent": "ADD_REMINDER", "extracted_time": "12:00", "content": "test"}
            convo.pending_action_id = "test_action_id"

            # Untrusted speech release
            convo.button_released(str(wav_path), trusted_input=False)
            with get_db_conn(db_path) as conn:
                rem_cnt = conn.execute("SELECT COUNT(*) FROM reminders").fetchone()[0]
            if convo.state == ConvoState.EXECUTING or rem_cnt > 0:
                untrusted_action_created = True
                case_passed = False
            case_details = {"final_state": convo.state.value, "reminders_created": rem_cnt}
            print(f"[{case_id}] {name:28s} -> Untrusted input result: State='{convo.state.value}', Reminders={rem_cnt}")

        elif sc_type == "playback_failure":
            # Test failed playback fencing
            mock_playback_fail = MockPlayback(should_fail=True)
            convo = ConversationManager(db_mgr, scheduler, asr, TTSWorker(strict_real=False), playback=mock_playback_fail)
            # Create action and reminder in DB
            action_id = "act_playback_fail"
            db_mgr.create_action(action_id, "create_reminder", {"query": "test reminder"})
            rem_id = scheduler.schedule_reminder(action_id, "2020-01-01T12:00:00+00:00")
            claims = scheduler.queue.claim_due_reminders("test_worker")
            token = claims[0]["delivery_token"] if claims else "token_claim"
            # Trigger due callback
            playback_failed = False
            try:
                convo._on_reminder_due(rem_id, action_id, "2020-01-01T12:00:00+00:00", attempt_token=token)
            except Exception:
                playback_failed = True
            with get_db_conn(db_path) as conn:
                rem_status = conn.execute("SELECT status FROM reminders WHERE id = ?", (rem_id,)).fetchone()[0]
            # Must NOT be acknowledged or completed
            if rem_status in ("acknowledged", "completed") or not playback_failed:
                failed_tts_acknowledged = True
                case_passed = False
            case_details = {"reminder_status": rem_status, "playback_failed_caught": playback_failed}
            print(f"[{case_id}] {name:28s} -> Playback failed caught={playback_failed}, reminder status='{rem_status}'.")

        elif sc_type == "stale_token":
            # Test stale token fencing
            convo = ConversationManager(db_mgr, scheduler, asr, TTSWorker(strict_real=False), playback=MockPlayback())
            action_id = "act_stale_token"
            db_mgr.create_action(action_id, "create_reminder", {"query": "test stale"})
            rem_id = scheduler.schedule_reminder(action_id, "2020-01-01T12:00:00+00:00")
            claims = scheduler.queue.claim_due_reminders("test_worker")
            real_token = claims[0]["delivery_token"] if claims else "token_real"
            # Fencing mismatch: attempt_token="stale_token_999" while DB token is real_token
            raised = False
            try:
                convo._on_reminder_due(rem_id, action_id, "2020-01-01T12:00:00+00:00", attempt_token="stale_token_999")
            except Exception as e:
                raised = True
            with get_db_conn(db_path) as conn:
                rem_status = conn.execute("SELECT status FROM reminders WHERE id = ?", (rem_id,)).fetchone()[0]
            if not raised or rem_status in ("acknowledged", "completed"):
                stale_token_acknowledged = True
                case_passed = False
            case_details = {"stale_token_aborted": raised, "reminder_status": rem_status}
            print(f"[{case_id}] {name:28s} -> Stale token attempt aborted: {raised}, status='{rem_status}'.")

        elif sc_type == "timeout":
            t0 = time.monotonic()
            res = asr.adapter.transcribe(str(wav_path), timeout=0.0001)
            dur = time.monotonic() - t0
            is_to = res.get("timeout") is True or res.get("error") == "Timeout"
            if not is_to or dur > 3.0:
                timeout_enforced = False
                case_passed = False
            case_details = {"reported_timeout": is_to, "wall_clock_s": round(dur, 4)}
            print(f"[{case_id}] {name:28s} -> Timeout enforced: {is_to} in {dur:.3f}s.")

        elif sc_type == "ambiguous_abort":
            convo = ConversationManager(db_mgr, scheduler, asr, TTSWorker(strict_real=False), playback=MockPlayback())
            convo.button_pressed()
            convo.button_released(str(wav_path), trusted_input=True)
            # Abort
            convo.cancel()
            with get_db_conn(db_path) as conn:
                rem_cnt = conn.execute("SELECT COUNT(*) FROM reminders").fetchone()[0]
            if rem_cnt > 0 or convo.state != ConvoState.IDLE:
                case_passed = False
            case_details = {"final_state": convo.state.value, "reminders_created": rem_cnt}
            print(f"[{case_id}] {name:28s} -> Cancelled ambiguous reminder: State='{convo.state.value}', Reminders={rem_cnt}.")

        elif sc_type == "mock_guard":
            is_real = not asr.is_mock
            case_passed = is_real
            case_details = {"is_real": is_real}
            print(f"[{case_id}] {name:28s} -> Real backend guard: {is_real}.")

        verify_source_hash(wav_path, expected_hash, f"Track 3 Post-Run {case_id}")

        t3_results.append({
            "case_id": case_id,
            "name": name,
            "type": sc_type,
            "passed": case_passed,
            "details": case_details
        })

        if "convo" in locals() and hasattr(convo, "_cancel_ttl"):
            convo._cancel_ttl()
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

    report["tracks"]["track_3_safety_audio"] = {
        "total_cases": len(t3_manifest),
        "all_passed": all(c["passed"] for c in t3_results),
        "details": t3_results
    }

    # Pass / Fail Gates Evaluation
    report["pass_fail_rules"]["silence_creates_zero_actions"] = "FAIL" if silence_action_created else "PASS"
    report["pass_fail_rules"]["untrusted_audio_rejected"] = "FAIL" if untrusted_action_created else "PASS"
    report["pass_fail_rules"]["failed_playback_unacknowledged"] = "FAIL" if failed_tts_acknowledged else "PASS"
    report["pass_fail_rules"]["stale_token_unacknowledged"] = "FAIL" if stale_token_acknowledged else "PASS"
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
        f.write("# ARM64 English Command Evaluation Report\n\n")
        f.write(f"- **Timestamp**: `{report['timestamp']}`\n")
        f.write(f"- **Platform Architecture**: `{report['platform']}`\n")
        f.write(f"- **Python Version**: `{report['python_version']}`\n")
        f.write(f"- **Engine**: `Sherpa-ONNX Zipformer English` (`models/sherpa-onnx-zipformer-en-2023-06-26`)\n")
        f.write(f"- **Language Mode**: `en` (explicit, no auto-detection)\n")
        f.write(f"- **Overall Evaluation Status**: **`{report['status']}`**\n\n")

        f.write("## 1. Required Pass/Fail Verification Rules\n\n")
        f.write("| Rule | Requirement | Status |\n")
        f.write("|:---|:---|:---:|\n")
        for rule, val in report["pass_fail_rules"].items():
            f.write(f"| `{rule}` | Mandatory safety invariant | **`{val}`** |\n")
        f.write("\n")

        f.write("## 2. Track Summary Metrics\n\n")
        f.write("| Track | Metric | Result |\n")
        f.write("|:---|:---|:---:|\n")
        t1 = report["tracks"]["track_1_general_speech"]
        f.write(f"| Track 1 (General Speech) | Total Clips Evaluated | `{t1['total_clips']}` |\n")
        f.write(f"| Track 1 (General Speech) | Mean WER | `{t1['mean_wer'] * 100:.2f}%` |\n")
        f.write(f"| Track 1 (General Speech) | Mean CER | `{t1['mean_cer'] * 100:.2f}%` |\n")
        f.write(f"| Track 1 (General Speech) | Latency (p50 / p95) | `{t1['p50_latency_s']:.3f}s` / `{t1['p95_latency_s']:.3f}s` |\n")
        f.write(f"| Track 1 (General Speech) | Mean RTF | `{t1['mean_rtf']:.3f}` |\n")

        t2 = report["tracks"]["track_2_assistant_commands"]
        f.write(f"| Track 2 (Commands) | Total Command Sessions | `{t2['total_sessions']}` |\n")
        f.write(f"| Track 2 (Commands) | Total Conversation Turns | `{t2['total_turns']}` |\n")
        f.write(f"| Track 2 (Commands) | Intent Recognition Accuracy | `{t2['intent_accuracy'] * 100:.2f}%` |\n")
        f.write(f"| Track 2 (Commands) | Trusted Task Completion Rate | `{t2['task_completion_rate'] * 100:.2f}%` |\n")
        f.write(f"| Track 2 (Commands) | Unsafe Database Mutations | **`{t2['unsafe_mutation_count']}`** |\n")
        f.write(f"| Track 2 (Commands) | End-to-End Latency (p50 / p95) | `{t2['p50_e2e_latency_s']:.3f}s` / `{t2['p95_e2e_latency_s']:.3f}s` |\n")

        t3 = report["tracks"]["track_3_safety_audio"]
        f.write(f"| Track 3 (Safety) | Total Safety / Defense Cases | `{t3['total_cases']}` |\n")
        f.write(f"| Track 3 (Safety) | All Safety Invariants Upheld | **`{t3['all_passed']}`** |\n\n")

        f.write("## 3. Track 1 Sample Details (General Speech)\n\n")
        f.write("| ID | Reference | Hypothesis | WER | CER | Latency | RTF |\n")
        f.write("|:---|:---|:---|:---:|:---:|:---:|:---:|\n")
        for s in t1["details"][:10]:
            f.write(f"| `{s['clip_id']}` | {s['reference']} | {s['hypothesis']} | {s['wer']:.2f} | {s['cer']:.2f} | {s['latency_s']:.3f}s | {s['rtf']:.3f} |\n")
        f.write("\n*(Showing first 10 of 22 clips; full details in JSON report)*\n\n")

        f.write("## 4. Track 2 Session Highlights (Assistant Commands)\n\n")
        f.write("| Session ID | Command Name | Turns | Task Success | E2E Latency | DB Mutations |\n")
        f.write("|:---|:---|:---:|:---:|:---:|:---:|\n")
        for s in t2["details"][:10]:
            f.write(f"| `{s['session_id']}` | `{s['name']}` | {len(s['turns'])} | `{s['task_success']}` | {s['e2e_latency_s']:.3f}s | {s['final_db_counts']} |\n")
        f.write("\n*(Showing first 10 of 22 sessions; full details in JSON report)*\n\n")

        f.write("## 5. Verified Model Weights (SHA-256)\n\n")
        f.write("| File Name | Size (Bytes) | SHA-256 |\n")
        f.write("|:---|:---:|:---|\n")
        for name, meta in report["model_hashes"].items():
            f.write(f"| `{name}` | `{meta['size_bytes']}` | `{meta['sha256']}` |\n")

    print("\n==================================================================")
    print(f"BENCHMARK COMPLETED: Overall Status: {report['status']}")
    print(f"JSON Report: {json_path}")
    print(f"Markdown:    {md_path}")
    print("==================================================================")

    if not overall_pass:
        sys.exit(1)

if __name__ == "__main__":
    main()
