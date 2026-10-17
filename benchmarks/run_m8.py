"""
benchmarks/run_m8.py — M8 Reproducible Real-World Benchmark Harness.

Three tracks:
  A – ASR accuracy (whisper.cpp, one inference per clip, unique sidecar dirs)
  B – Background-content safety (imported audio must commit zero actions)
  C – Assistant command sessions (separate annotated command manifest)

Outputs go to reports/<run_id>/ so prior runs are never overwritten.
Source-file SHA-256 is verified before and after each track.
"""

import os, json, time, argparse, hashlib, shutil, threading, tempfile, platform
from datetime import datetime

import psutil

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from providers.asr_worker import ASRWorker
from providers.tts_worker import TTSWorker
from scheduler.scheduler import ReminderScheduler
from storage.database import StorageManager, get_db_conn
try:
    from app.conversation import ConversationManager
except ImportError:
    ConversationManager = None
try:
    from jiwer import wer, cer
except ImportError:
    raise ImportError("jiwer package is required for calculating WER/CER in benchmarks")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def sha256_file(path: str) -> str:
    import hashlib
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest().lower()


def md5_file(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _poll_peak_memory(stop_event: threading.Event, result: list):
    """Background thread: sample child-process RSS every 50 ms."""
    peak = 0.0
    proc = psutil.Process(os.getpid())
    while not stop_event.is_set():
        try:
            total = proc.memory_info().rss
            for child in proc.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            peak = max(peak, total / (1024 * 1024))
        except Exception:
            pass
        stop_event.wait(0.05)
    result.append(peak)


def timed_transcribe(asr: ASRWorker, wav_path: str, timeout: float = 45.0):
    """Run ASR with a background memory-poller; return (result, latency, peak_mb)."""
    stop = threading.Event()
    peak_result = []
    t = threading.Thread(target=_poll_peak_memory, args=(stop, peak_result), daemon=True)
    t.start()
    t0 = time.monotonic()
    try:
        res = asr.transcribe(wav_path, timeout=timeout)
    except Exception as e:
        res = {"error": f"inference_exception: {e}"}
    finally:
        latency = time.monotonic() - t0
        stop.set()
        t.join()
        peak_mb = peak_result[0] if peak_result else 0.0
    return res, latency, peak_mb


def verify_unchanged(path_to_hash: dict[str, str], stage: str):
    """Abort if any file hash has changed since the baseline was taken."""
    for path, expected in path_to_hash.items():
        if not os.path.exists(path):
            raise RuntimeError(f"[{stage}] Source file missing: {path}")
        got = sha256_file(path)
        if got != expected:
            raise RuntimeError(
                f"[{stage}] SHA-256 mismatch on {path}\n"
                f"  expected: {expected}\n  got:      {got}"
            )


def validate_manifest_item(item: dict, schema: dict = None) -> list[str]:
    import jsonschema
    if schema:
        try:
            # We assume schema here is the schema for the array items
            jsonschema.validate(instance=item, schema=schema["items"])
            return []
        except jsonschema.exceptions.ValidationError as e:
            return [e.message]
    else:
        required = ["id", "file", "audio_sha256", "language", "expected_transcript", "review_status"]
        missing = [f for f in required if f not in item]
        if "duration" not in item and ("start_time" not in item or "end_time" not in item):
            missing.append("start_time/end_time or duration")
        return missing


def is_eligible(item: dict) -> bool:
    """Shared eligibility function for Track A."""
    return item.get("review_status") in ("REVIEWED", "SOURCE_VALIDATED")

# ---------------------------------------------------------------------------
# Track A — ASR only
# ---------------------------------------------------------------------------

def run_track_a(manifest, dataset_dir: str, run_dir: str,
                model_path: str, asr_bin: str, source_hashes: dict, language_mode: str = "manifest"):
    print("\n--- TRACK A: ASR ONLY ---")
    results = []
    
    schema = None
    schema_path = os.path.join(dataset_dir, "manifest_schema.json")
    if os.path.exists(schema_path):
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

    for item in manifest:
        clip_id = item.get("id", item.get("clip_id", "?"))

        # Schema check
        missing = validate_manifest_item(item, schema=schema)
        if missing:
            results.append({"track": "A", "clip_id": clip_id,
                            "error": f"manifest_schema_error: missing {missing}"})
            continue

        # Skip un-reviewed
        if not is_eligible(item):
            results.append({"track": "A", "clip_id": clip_id, "skipped": "NEEDS_HUMAN_REVIEW"})
            continue

        wav_path = os.path.join(dataset_dir, item["file"])
        if not os.path.exists(wav_path):
            results.append({"track": "A", "clip_id": clip_id, "error": "audio_missing"})
            continue

        # Verify before inference
        verify_unchanged({wav_path: source_hashes[wav_path]}, f"track_A pre {clip_id}")

        # Unique sidecar work directory per clip prevents stale JSON contamination
        work_dir = os.path.join(run_dir, "sidecar", clip_id)
        os.makedirs(work_dir, exist_ok=True)
        temp_wav = os.path.join(work_dir, f"{clip_id}.wav")
        shutil.copy2(wav_path, temp_wav)

        # Language configuration
        if language_mode == "manifest":
            lang = item.get("language", "auto")
        else:
            lang = language_mode
            
        if lang == "hinglish":
            lang = "auto"  # whisper.cpp flag for multilingual
            
        asr = ASRWorker(cli_path=asr_bin, model_path=model_path,
                        language=lang, strict_real=True)

        try:
            res, latency, peak_mb = timed_transcribe(asr, temp_wav, timeout=45.0)
            
            res_clean = {k: v for k, v in res.items() if not k.startswith("raw_")} if isinstance(res, dict) else res
            with open(os.path.join(work_dir, "diagnostics.json"), "w", encoding="utf-8") as f:
                json.dump(res_clean, f, indent=2, ensure_ascii=False)
                
            if isinstance(res, dict):
                if res.get("raw_stdout") is not None:
                    with open(os.path.join(work_dir, "stdout.bin"), "wb") as f:
                        f.write(res["raw_stdout"])
                if res.get("raw_stderr") is not None:
                    with open(os.path.join(work_dir, "stderr.bin"), "wb") as f:
                        f.write(res["raw_stderr"])
                if res.get("raw_json") is not None:
                    with open(os.path.join(work_dir, "sidecar.bin"), "wb") as f:
                        f.write(res["raw_json"])
        finally:
            try:
                os.remove(temp_wav)
            except Exception:
                pass

        # Verify after inference
        verify_unchanged({wav_path: source_hashes[wav_path]}, f"track_A post {clip_id}")

        transcript = res.get("text", "") if isinstance(res, dict) else str(res)
        asr_error = res.get("error") if isinstance(res, dict) else None
        expected = item.get("expected_transcript", "")

        # Empty reference: report hallucinations/min, not WER
        if not expected.strip():
            words_hallucinated = len(transcript.split()) if transcript.strip() else 0
            duration_val = item.get("duration", item.get("end_time", 1) - item.get("start_time", 0))
            duration_s = max(duration_val, 1)
            hpm = words_hallucinated / (duration_s / 60)
            results.append({
                "track": "A", "clip_id": clip_id,
                "empty_reference": True,
                "hallucinations_per_min": hpm,
                "transcript": transcript,
                "latency": latency, "peak_mb": peak_mb,
                "rtf": latency / max(duration_s, 1)
            })
            continue

        # Empty / error output → 100 % error, stays in denominator
        if not transcript.strip() or asr_error:
            sample_wer, sample_cer = 1.0, 1.0
        else:
            sample_wer = wer(expected.lower(), transcript.lower())
            sample_cer = cer(expected.lower(), transcript.lower())

        duration_val = item.get("duration", item.get("end_time", 1) - item.get("start_time", 0))
        duration_s = max(duration_val, 1)
        results.append({
            "track": "A", "clip_id": clip_id,
            "language": lang, "genre": item.get("genre"),
            "cli_args": res.get("cli_args") if isinstance(res, dict) else None,
            "overlap": item.get("overlap", False),
            "expected": expected, "transcript": transcript,
            "wer": sample_wer, "cer": sample_cer,
            "latency": latency, "peak_mb": peak_mb,
            "rtf": latency / duration_s,
            "error": asr_error
        })

    return results


# ---------------------------------------------------------------------------
# Track B — Background-content safety
# ---------------------------------------------------------------------------

def run_track_b(manifest, dataset_dir: str, run_dir: str,
                model_path: str, asr_bin: str, source_hashes: dict, language_mode: str = "manifest"):
    """Untrusted audio through the full pipeline; zero authorised writes expected."""
    print("\n--- TRACK B: CONTENT SAFETY ---")
    results = []

    # Each Track B run gets its own isolated DB
    td = os.path.join(run_dir, "track_b_db")
    os.makedirs(td, exist_ok=True)
    db_path = os.path.join(td, "safety.db")
    db = StorageManager(db_path)
    scheduler = ReminderScheduler(db_path, poll_interval=1.0)
    scheduler.start()

    asr = ASRWorker(cli_path=asr_bin, model_path=model_path, language="auto", strict_real=True)
    tts = TTSWorker(engine="sapi5", strict_real=True)
    convo = ConversationManager(db, scheduler, asr, tts)

    try:
        for item in manifest:
            clip_id = item.get("id", item.get("clip_id", "?"))
            if item.get("review_status") != "REVIEWED":
                results.append({"track": "B", "clip_id": clip_id, "skipped": "NEEDS_HUMAN_REVIEW"})
                continue

            if language_mode == "manifest":
                lang = item.get("language", "auto")
            else:
                lang = language_mode
                
            if lang == "hinglish":
                lang = "auto"

            # Set language strictly per clip
            asr.language = lang

            wav_path = os.path.join(dataset_dir, item["file"])
            if not os.path.exists(wav_path):
                results.append({"track": "B", "clip_id": clip_id, "error": "audio_missing"})
                continue

            verify_unchanged({wav_path: source_hashes[wav_path]}, f"track_B pre {clip_id}")

            # Use a throw-away copy; is_temp=False because THIS is untrusted content
            temp_wav = os.path.join(td, f"temp_rec_{clip_id}.wav")
            shutil.copy2(wav_path, temp_wav)

            # Count baseline confirmed actions (state != 'pending') before
            with get_db_conn(db_path) as conn:
                before = conn.execute(
                    "SELECT COUNT(*) FROM actions WHERE state NOT IN ('pending')"
                ).fetchone()[0]

            convo.button_pressed()
            # is_temp=True so the throw-away copy is cleaned up
            # trusted_input=False ensures sequence attacks (e.g. "yes") don't authorize actions
            convo.button_released(temp_wav, is_temp=True, trusted_input=False)

            with get_db_conn(db_path) as conn:
                after = conn.execute(
                    "SELECT COUNT(*) FROM actions WHERE state NOT IN ('pending')"
                ).fetchone()[0]
                new_events = conn.execute(
                    "SELECT COUNT(*) FROM calendar_events"
                ).fetchone()[0]
                new_reminders = conn.execute(
                    "SELECT COUNT(*) FROM reminders"
                ).fetchone()[0]

            verify_unchanged({wav_path: source_hashes[wav_path]}, f"track_B post {clip_id}")

            violation = (after > before) or (new_events > 0) or (new_reminders > 0)
            results.append({
                "track": "B", "clip_id": clip_id,
                "language": lang,
                "safety_violation": violation,
                "confirmed_actions_delta": after - before,
                "total_events": new_events,
                "total_reminders": new_reminders
            })
    finally:
        scheduler.stop()

    return results


# ---------------------------------------------------------------------------
# Track C — Assistant command sessions
# ---------------------------------------------------------------------------

def validate_command_audio(c_manifest, dataset_dir):
    if not c_manifest:
        return [{"track": "C", "status": "NOT TESTED", "reason": "Empty command manifest"}]
    from audio.recording_validation import inspect_recording
    invalid = []
    human_dataset = os.path.basename(os.path.abspath(dataset_dir)) == "commands_human_v1"
    for session in c_manifest:
        if not session.get('commands'):
            invalid.append({'track': 'C', 'status': 'NOT TESTED', 'session_id': session.get('session_id'),
                            'reason': 'Session contains no commands'})
        for cmd in session.get("commands", []):
            path = os.path.join(dataset_dir, cmd["file"])
            errors = inspect_recording(path)["errors"]
            if human_dataset and (cmd.get("recording_type") != "human" or
                                  cmd.get("review_status") != "REVIEWED" or
                                  session.get("consent_recorded") is not True):
                errors.append("Unverified human recording/review/consent")
            if os.path.isfile(path) and cmd.get("audio_sha256") and sha256_file(path) != cmd["audio_sha256"]:
                errors.append("Audio hash mismatch")
            if errors:
                invalid.append({"track": "C", "status": "NOT TESTED", "session_id": session.get("session_id"),
                                "file": cmd["file"], "error": "; ".join(errors)})
    return invalid
    


def run_track_c(command_manifest_path: str | None, run_dir: str,
                model_path: str, asr_bin: str, mock_asr: bool = False, language_mode: str = "manifest",
                timeout: float = 45.0, reference_time: str | None = None):
    print("\n--- TRACK C: ASSISTANT COMMANDS ---")
    if not command_manifest_path or not os.path.exists(command_manifest_path):
        return [{"track": "C", "status": "NOT TESTED",
                 "reason": "No command manifest provided (--commands flag)"}]
                 
    results = []
    with open(command_manifest_path, 'r', encoding='utf-8') as f:
        c_manifest = json.load(f)
        
    dataset_dir = os.path.dirname(command_manifest_path)

    invalid = validate_command_audio(c_manifest, dataset_dir)
    if invalid:
        return invalid

    if timeout <= 0 or not __import__('math').isfinite(timeout):
        raise ValueError('Track C timeout must be finite and positive')
    frozen = datetime.fromisoformat(reference_time) if reference_time else None
    if frozen is not None and frozen.tzinfo is None:
        raise ValueError('Track C reference_time must include timezone offset')
    provenance = {'measurement_type': 'software_test' if mock_asr else 'real_asr_with_simulated_output',
                  'model_sha256': None if mock_asr else sha256_file(model_path),
                  'executable_sha256': None if mock_asr else sha256_file(asr_bin),
                  'manifest_sha256': sha256_file(command_manifest_path),
                  'reference_time': reference_time, 'timeout_s': timeout,
                  'audible_delivery_tested': False}

    for session in c_manifest:
        session_id = session.get("session_id", "?")
        
        if language_mode == "manifest":
            session_lang = session.get("language", "auto")
        else:
            session_lang = language_mode
            
        if session_lang == "hinglish":
            session_lang = "auto"
            
        # Each session gets its own DB
        sd = os.path.join(run_dir, "track_c_db", session_id)
        os.makedirs(sd, exist_ok=False)
        db_path = os.path.join(sd, "session.db")
        db = StorageManager(db_path)
        
        scheduler = ReminderScheduler(db_path, poll_interval=1.0)
        
        class RecordingTTS:
            def __init__(self):
                self.last_spoken = None
            def synthesize(self, text, path):
                self.last_spoken = text
                return True
                
        class RecordingASR:
            def __init__(self, real_asr, is_mock=False):
                self.real = real_asr
                self.is_mock = is_mock
                self.last_transcript = None
                self.mock_transcript = None
                self.last_result = None
            def transcribe(self, wav_path):
                if self.is_mock:
                    res = {'text': self.mock_transcript, 'error': None, 'language': session_lang,
                           'duration': None, 'peak_process_memory_bytes': None, 'software_test': True}
                else:
                    try:
                        res = self.real.transcribe(wav_path, timeout=timeout)
                    except Exception as exc:
                        res = {'text': '', 'error': f'inference_exception: {exc}', 'language': session_lang}
                self.last_transcript = res.get('text', '')
                self.last_result = res
                return res

        real_asr = None if mock_asr else ASRWorker(cli_path=asr_bin, model_path=model_path, language=session_lang, strict_real=True)
        asr_wrapper = RecordingASR(real_asr, is_mock=mock_asr)
        tts_wrapper = RecordingTTS()
        
        class MockPlayback:
            def play(self, path): pass
            
        convo = ConversationManager(db, scheduler, asr_wrapper, tts_wrapper, playback=MockPlayback(),
                                    clock=(lambda: frozen) if frozen else None)
        
        asr_success = True
        intent_success = True
        task_success = True
        errors = []
        turns = []
        
        for idx, cmd in enumerate(session.get("commands", [])):
            wav_file = os.path.join(dataset_dir, cmd["file"])
            if not os.path.exists(wav_file):
                task_success = False
                errors.append(f"Missing audio {cmd['file']}")
                break
                
            trusted = cmd.get("trusted_input", cmd.get("trusted", True))
            tts_wrapper.last_spoken = None
            asr_wrapper.last_transcript = None
            asr_wrapper.last_result = None
            asr_wrapper.mock_transcript = cmd.get("expected_transcript", "")
            
            orig_hash = sha256_file(wav_file)
            turn_dir = os.path.join(sd, f'turn_{idx + 1:03d}')
            os.makedirs(turn_dir)
            work_wav = os.path.join(turn_dir, 'input.wav')
            shutil.copy2(wav_file, work_wav)
            convo.button_pressed()
            pipeline_error = None
            try:
                convo.button_released(work_wav, is_temp=False, trusted_input=trusted)
            except Exception as exc:
                pipeline_error = repr(exc)
                errors.append(f'Step {idx}: Pipeline exception: {exc}')
                task_success = False
            new_hash = sha256_file(wav_file)

            res = asr_wrapper.last_result or {'error': 'ASR was not invoked', 'text': ''}
            if res.get('error'):
                asr_success = False
                errors.append(f"Step {idx}: ASR error: {res['error']}")
            diagnostics = {k: v for k, v in res.items() if not k.startswith('raw_')}
            for field in ('raw_stdout', 'raw_stderr', 'raw_json'):
                raw = res.get(field)
                if raw is not None:
                    with open(os.path.join(turn_dir, field + '.bin'), 'wb') as stream:
                        stream.write(raw if isinstance(raw, bytes) else str(raw).encode('utf-8'))
            import soundfile as sf
            audio_duration = sf.info(work_wav).duration
            turn = {'turn': idx + 1, 'reference': cmd.get('expected_transcript'),
                    'hypothesis': res.get('text', ''), 'actual_language_argument': res.get('language', session_lang),
                    'error': res.get('error'), 'pipeline_error': pipeline_error,
                    'timeout': res.get('timeout', res.get('error') == 'Timeout'),
                    'asr_elapsed_s': None if mock_asr else res.get('duration'),
                    'peak_process_memory_bytes': None if mock_asr else res.get('peak_process_memory_bytes'),
                    'audio_duration_s': audio_duration, 'source_sha256': orig_hash,
                    'source_file': wav_file, 'trusted_input': trusted,
                    'state': convo.state.value, 'spoken_response': tts_wrapper.last_spoken,
                    'diagnostics_directory': os.path.abspath(turn_dir), 'diagnostics': diagnostics}
            turn['rtf'] = turn['asr_elapsed_s'] / audio_duration if turn['asr_elapsed_s'] is not None and audio_duration else None
            if cmd.get('expected_transcript') is not None:
                turn['wer'] = None if mock_asr else wer(cmd['expected_transcript'].lower(), turn['hypothesis'].lower())
                turn['cer'] = None if mock_asr else cer(cmd['expected_transcript'].lower(), turn['hypothesis'].lower())
            with get_db_conn(db_path) as conn:
                action_ids = [r[0] for r in conn.execute('SELECT id FROM actions ORDER BY rowid')]
                conn.row_factory = __import__('sqlite3').Row
                turn['stored_reminders'] = [dict(r) for r in conn.execute('SELECT * FROM reminders ORDER BY rowid')]
            turn['stored_actions'] = [db.get_action(aid) for aid in action_ids]
            expected_contents = cmd.get('expected_action_data')
            if expected_contents is not None:
                confirmed = [a for a in turn['stored_actions'] if a['state'] == 'confirmed']
                if not confirmed or any(confirmed[-1]['data'].get(k) != v for k, v in expected_contents.items()):
                    errors.append(f'Step {idx}: Stored action contents mismatch')
                    task_success = False
            expected_reminder = cmd.get('expected_reminder')
            if expected_reminder is not None and (not turn['stored_reminders'] or any(
                    turn['stored_reminders'][-1].get(k) != v for k, v in expected_reminder.items())):
                errors.append(f'Step {idx}: Stored reminder contents mismatch')
                task_success = False
            turns.append(turn)
            with open(os.path.join(turn_dir, 'diagnostics.json'), 'w', encoding='utf-8') as stream:
                json.dump(turn, stream, ensure_ascii=False, indent=2)
            
            if orig_hash != new_hash:
                errors.append(f"Step {idx}: Source WAV hash modified during execution!")
                task_success = False
            
            # Step assertions
            if cmd.get("expected_transcript"):
                # We do a loose check because ASR is real and might make small mistakes
                # But for strict C1 correctness on synthetic TTS audio, we expect near-perfect match
                # Wait, the user wants "expected_transcript". We'll just assert it's somewhat close or logged.
                # Actually, the user says "expected transcript". Let's check exactly, ignoring case/punct.
                import re
                def clean(t): return re.sub(r'[^\w\s]', '', (t or '').lower()).strip()
                if clean(asr_wrapper.last_transcript) != clean(cmd["expected_transcript"]):
                    errors.append(f"Step {idx}: Transcript mismatch. Expected '{cmd['expected_transcript']}', got '{asr_wrapper.last_transcript}'")
                    asr_success = False
            
            if cmd.get("expected_state") and convo.state.value != cmd["expected_state"]:
                errors.append(f"Step {idx}: State mismatch. Expected '{cmd['expected_state']}', got '{convo.state.value}'")
                intent_success = False
                
            if cmd.get("expected_spoken_response") is not None:
                import re
                expected_regex = cmd["expected_spoken_response"]
                if expected_regex == "":
                    if tts_wrapper.last_spoken is not None:
                        errors.append(f"Step {idx}: Spoken mismatch. Expected nothing, got '{tts_wrapper.last_spoken}'")
                        intent_success = False
                elif not tts_wrapper.last_spoken or not re.search(expected_regex, tts_wrapper.last_spoken):
                    errors.append(f"Step {idx}: Spoken mismatch. Expected '{expected_regex}', got '{tts_wrapper.last_spoken}'")
                    intent_success = False
                    
            if cmd.get("expected_db"):
                with get_db_conn(db_path) as conn:
                    actions = conn.execute("SELECT COUNT(*) FROM actions WHERE state = 'confirmed'").fetchone()[0]
                    reminders = conn.execute("SELECT COUNT(*) FROM reminders").fetchone()[0]
                    events = conn.execute("SELECT COUNT(*) FROM calendar_events").fetchone()[0]
                    notes = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
                exp_db = cmd["expected_db"]
                if actions != exp_db.get("actions", actions): errors.append(f"Step {idx}: Actions mismatch"); task_success = False
                if reminders != exp_db.get("reminders", reminders): errors.append(f"Step {idx}: Reminders mismatch"); task_success = False
                if events != exp_db.get("events", events): errors.append(f"Step {idx}: Events mismatch"); task_success = False
                if notes != exp_db.get("notes", notes): errors.append(f"Step {idx}: Notes mismatch"); task_success = False
        
        # Final state must be checked even after recognition/intent failure: writes may have occurred.
        expected = session.get("expected_final_state", {})
        with get_db_conn(db_path) as conn:
            actions = conn.execute("SELECT COUNT(*) FROM actions WHERE state = 'confirmed'").fetchone()[0]
            reminders = conn.execute("SELECT COUNT(*) FROM reminders").fetchone()[0]
            events = conn.execute("SELECT COUNT(*) FROM calendar_events").fetchone()[0]
            notes = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
            
        if "actions_confirmed" in expected and actions != expected["actions_confirmed"]:
            task_success = False
            errors.append(f"Expected {expected['actions_confirmed']} confirmed actions, got {actions}")
        if "reminders_count" in expected and reminders != expected["reminders_count"]:
            task_success = False
            errors.append(f"Expected {expected['reminders_count']} reminders, got {reminders}")
        if "events_count" in expected and events != expected["events_count"]:
            task_success = False
            errors.append(f"Expected {expected['events_count']} events, got {events}")
        if "notes_count" in expected and notes != expected["notes_count"]:
            task_success = False
            errors.append(f"Expected {expected['notes_count']} notes, got {notes}")
            
        with get_db_conn(db_path) as conn:
            actions = conn.execute("SELECT COUNT(*) FROM actions WHERE state = 'confirmed'").fetchone()[0]
            reminders = conn.execute("SELECT COUNT(*) FROM reminders").fetchone()[0]

        results.append({
            "track": "C", 
            "session_id": session_id, 
            "language": session_lang,
            "provenance": provenance,
            "turns": turns,
            "asr_success": asr_success,
            "intent_success": intent_success,
            "task_success": task_success,
            "success": task_success and intent_success and asr_success, 
            "error": " | ".join(errors) if errors else None,
            "actions_confirmed": actions,
            "reminders_count": reminders,
            "events_count": events,
            "notes_count": notes,
            "inference_errors": sum(bool(t['error']) for t in turns),
            "inference_error_rate": None if mock_asr else sum(bool(t['error']) for t in turns) / len(turns),
            "human_command_accuracy_measured": not mock_asr and all(
                c.get('recording_type') == 'human' for c in session.get('commands', []))
        })
        
    return results


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate(results: list[dict]):
    import numpy as np  # already in .venv (required by jiwer)

    a_results = [r for r in results if r.get("track") == "A" and
                 "wer" in r and "cer" in r]
    if not a_results:
        return {}

    by_lang = {}
    by_genre = {}
    for r in a_results:
        for bucket, key in [(by_lang, "language"), (by_genre, "genre")]:
            k = r.get(key, "unknown") or "unknown"
            bucket.setdefault(k, []).append(r)

    def stats(recs):
        ref_words = sum(len(r["expected"].split()) for r in recs)
        sub_del_ins = sum(r["wer"] * len(r["expected"].split()) for r in recs)
        corpus_wer = sub_del_ins / ref_words if ref_words else None
        timeout_count = sum(1 for r in recs if r.get("error") == "Timeout")
        timeout_rate = timeout_count / len(recs) if recs else 0.0
        return {
            "n": len(recs),
            "corpus_wer": corpus_wer,
            "mean_clip_wer": float(np.mean([r["wer"] for r in recs])),
            "mean_cer": float(np.mean([r["cer"] for r in recs])),
            "p50_latency": float(np.percentile([r["latency"] for r in recs], 50)),
            "p95_latency": float(np.percentile([r["latency"] for r in recs], 95)),
            "mean_rtf": float(np.mean([r["rtf"] for r in recs])),
            "peak_mb_max": float(max(r["peak_mb"] for r in recs)),
            "timeout_rate": float(timeout_rate),
        }

    return {
        "by_language": {k: stats(v) for k, v in by_lang.items()},
        "by_genre":    {k: stats(v) for k, v in by_genre.items()},
        "overall":     stats(a_results),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="M8 Real-World Benchmark Harness")
    parser.add_argument("--dataset",  default="datasets/youtube_stress_v1",
                        help="Directory containing manifest.json and audio files")
    parser.add_argument("--commands", default=None,
                        help="Optional path to assistant command manifest (Track C)")
    parser.add_argument("--model", type=str, default="models/ggml-small.bin", help="Path to ggml model")
    parser.add_argument("--whisper-cli", dest="asr_bin", type=str, default="bin_extract/whisper-latest/Release/whisper-cli.exe")
    parser.add_argument("--mock-asr", action="store_true", help="Inject expected transcripts directly instead of running Whisper")
    parser.add_argument("--verify-only", action="store_true", help="Only perform manifest and SHA-256 hash verification without starting ASR")
    parser.add_argument("--skip-track-a", action="store_true", help="Skip Track A ASR benchmark")
    parser.add_argument("--language-mode", type=str, choices=["manifest", "auto", "en", "hi"], default="manifest", help="Language configuration mode")
    parser.add_argument('--command-timeout', type=float, default=45.0, help='Finite per-command ASR timeout in seconds')
    parser.add_argument('--reference-time', default=None, help='Freeze Track C parser clock at timezone-aware ISO timestamp')
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset)
    model_path  = os.path.abspath(args.model)
    asr_bin     = os.path.abspath(args.asr_bin)

    manifest_path = os.path.join(dataset_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"ERROR: manifest not found at {manifest_path}")
        return 1

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    if not manifest:
        print("Manifest is empty. Nothing to evaluate.")
        return 1

    # Unique run directory — never overwrites prior runs (UUID for total exclusivity)
    import uuid
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    run_dir = os.path.abspath(os.path.join("reports", f"run_{run_id}"))
    os.makedirs(run_dir, exist_ok=True)
    results_file = os.path.join(run_dir, "m8_results.jsonl")

    # Snapshot source-file hashes before ANY inference
    source_hashes: dict[str, str] = {}
    
    # Load schema for validation
    schema = None
    schema_path = os.path.join(dataset_dir, "manifest_schema.json")
    if os.path.exists(schema_path):
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
            
    for item in manifest:
        if not is_eligible(item):
            continue
            
        if args.verify_only:
            missing = validate_manifest_item(item, schema=schema)
            if missing:
                print(f"ERROR: Schema validation failed for item {item.get('id', '?')}: {missing}")
                return 1
                
        p = os.path.join(dataset_dir, item.get("file", ""))
        if os.path.exists(p):
            actual_hash = sha256_file(p).lower()
            expected_hash = item.get("audio_sha256", "").lower()
            if expected_hash and actual_hash != expected_hash:
                print(f"ERROR: Hash mismatch for {p}. Expected {expected_hash}, got {actual_hash}")
                return 1
            source_hashes[p] = actual_hash
            
    if args.commands:
        with open(args.commands, encoding="utf-8") as stream:
            command_errors = validate_command_audio(json.load(stream), os.path.dirname(args.commands))
        if command_errors:
            print("ERROR: Command recording preflight failed: " + json.dumps(command_errors, ensure_ascii=True))
            return 1

    if args.verify_only:
        print("Verify-only mode: manifest and hashes validated successfully.")
        return 0

    # Write run metadata
    metadata = {
        "type": "metadata",
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "hardware": platform.processor(),
        "os": f"{platform.system()} {platform.version()}",
        "python": platform.python_version(),
        "model_filename": os.path.basename(model_path),
        "model_path": model_path,
        "model_sha256": sha256_file(model_path),
        "model_md5": md5_file(model_path).upper() if md5_file(model_path) else None,
        "model_size": os.path.getsize(model_path),
        "asr_bin_sha256": sha256_file(asr_bin),
        "cli_path": asr_bin,
        "language_mode": args.language_mode,
        "platform": f"{platform.system()} {platform.machine()}",
        "dataset": dataset_dir,
        "dataset_items": len(manifest),
        "timeout_policy": "45.0s per clip",
    }
    with open(results_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(metadata) + "\n")

    all_results: list[dict] = []

    a_results = run_track_a(manifest, dataset_dir, run_dir, model_path, asr_bin, source_hashes, args.language_mode)
    all_results.extend(a_results)

    b_results = run_track_b(manifest, dataset_dir, run_dir, model_path, asr_bin, source_hashes, args.language_mode)
    all_results.extend(b_results)

    c_results = run_track_c(args.commands, run_dir, model_path, asr_bin, args.mock_asr, args.language_mode,
                            timeout=args.command_timeout, reference_time=args.reference_time)
    all_results.extend(c_results)

    # Write per-sample results
    for r in all_results:
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(r) + "\n")

    # Aggregate and write summary
    summary = _aggregate(all_results)
    summary_record = {"type": "summary", **summary}
    with open(results_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary_record) + "\n")

    # Final provenance check
    verify_unchanged(source_hashes, "final provenance check")

    # Human-readable summary
    print(f"\n{'='*60}")
    print(f"Run ID: {run_id}")
    print(f"Results: {results_file}")
    if summary.get("overall"):
        ov = summary["overall"]
        print(f"\nOverall (Track A):")
        print(f"  Clips evaluated : {ov['n']}")
        print(f"  Corpus WER      : {ov['corpus_wer']:.4f}" if ov['corpus_wer'] is not None else "  Corpus WER      : N/A")
        print(f"  Mean clip WER   : {ov['mean_clip_wer']:.4f}")
        print(f"  Mean CER        : {ov['mean_cer']:.4f}")
        print(f"  p50 latency     : {ov['p50_latency']:.2f}s")
        print(f"  p95 latency     : {ov['p95_latency']:.2f}s")
        print(f"  Mean RTF        : {ov['mean_rtf']:.4f}x")
        print(f"  Peak mem (max)  : {ov['peak_mb_max']:.1f} MB")
        print(f"  Timeout rate    : {ov['timeout_rate']:.2%}")

    b_violations = sum(1 for r in b_results if r.get("safety_violation"))
    print(f"\nTrack B safety violations : {b_violations} / {len(b_results)}")

    c_failures = 0
    if c_results:
        print(f"\nTrack C Assistant Commands:")
        asr_passes = sum(1 for r in c_results if r.get("asr_success"))
        intent_passes = sum(1 for r in c_results if r.get("intent_success"))
        task_passes = sum(1 for r in c_results if r.get("task_success"))
        
        for r in c_results:
            if r.get("status") == "NOT TESTED":
                print(f"  Track C: {r.get('reason')}")
                continue
                
            status = "PASS" if r.get("success") else "FAIL"
            if status == "FAIL":
                c_failures += 1
                err_text = str(r.get('error')).encode('ascii', 'backslashreplace').decode('ascii')
                print(f"  {r.get('session_id')}: {status} - {err_text}")
            else:
                print(f"  {r.get('session_id')}: {status}")
        
        valid_c_results = [r for r in c_results if r.get("status") != "NOT TESTED"]
        if valid_c_results:
            print(f"\n  {'Software transcript checks' if args.mock_asr else 'Transcript exact-match sessions'}: {asr_passes} / {len(valid_c_results)}")
            print(f"  Intent Correctness: {intent_passes} / {len(valid_c_results)}")
            print(f"  Task Success      : {task_passes} / {len(valid_c_results)}")
            print(f"  Total End-to-End Failures: {c_failures} / {len(valid_c_results)}")
    if b_violations > 0 or c_failures > 0:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
