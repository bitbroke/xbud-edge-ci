import time
import logging
from enum import Enum
import uuid

from providers.asr_worker import ASRWorker
from providers.tts_worker import TTSWorker
from storage.database import StorageManager
from scheduler.scheduler import ReminderScheduler
from audio.models import CaptureState
from nlp.slm_intent_parser import NLPEngine
from app.telemetry import TelemetryManager, get_session_id
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
import threading
tts_active_flag = threading.Event()

class ConvoState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    CLARIFYING = "clarifying"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    EXECUTING = "executing"
    SPEAKING = "speaking"
    FAILED = "failed"

class ConversationManager:
    def __init__(self, db_manager: StorageManager, scheduler: ReminderScheduler, asr: ASRWorker, tts: TTSWorker, playback=None, clock=None):
        self.db = db_manager
        self.scheduler = scheduler
        self.asr = asr
        self.tts = tts
        self.playback = playback
        self.clock = clock
        self.state = ConvoState.IDLE
        self.previous_state = ConvoState.IDLE
        
        self.pending_action_id = None
        self.pending_intent_data = None
        self.active_session_id = None
        self.telemetry = TelemetryManager.get()
        
        self.ttl_timer = None
        
        # Instantiate the heavy SLM model once to avoid cold-start delays on every parse
        self.nlp_engine = NLPEngine(clock=self.clock)
        self.last_transcript = ""
        self.last_parsed_intent = None
        logger.info("ConversationManager initialized.")

    def _start_ttl(self):
        self._cancel_ttl()
        def timeout():
            logger.info("TTL expired: Conversation abandoned. Resetting state.")
            try:
                if self.state in [ConvoState.CLARIFYING, ConvoState.AWAITING_CONFIRMATION]:
                    self.cancel()
            except Exception:
                pass
        import threading
        self.ttl_timer = threading.Timer(30.0, timeout)
        # A pending conversation must never keep the application alive during
        # shutdown.  The normal cancellation paths still cancel this timer.
        self.ttl_timer.daemon = True
        self.ttl_timer.start()

    def _cancel_ttl(self):
        if self.ttl_timer:
            self.ttl_timer.cancel()
            self.ttl_timer = None

    def _on_reminder_due(self, reminder_id, action_id, due_time, attempt_token: int = None):
        old_state = self.state
        self.state = ConvoState.SPEAKING
        tts_active_flag.set()
        
        try:
            action = self.db.get_action(action_id)
            if not action or action['state'] == 'cancelled':
                logger.info("Action cancelled, dropping reminder.")
                self.scheduler.cancel_reminder(reminder_id)
                return

            intent = action['intent']
            if intent in ('create_event', 'update_event'):
                event_id = action_id if intent == 'create_event' else action['data'].get('target_id')
                event = self.db.get_event(event_id)
                if not event:
                    logger.info("Event cancelled, dropping reminder.")
                    self.scheduler.cancel_reminder(reminder_id)
                    return
                
                action_rev = 1 if intent == 'create_event' else action['data'].get('expected_revision', 0) + 1
                if event['revision'] != action_rev:
                    logger.info("Event revision changed, dropping stale reminder.")
                    self.scheduler.cancel_reminder(reminder_id)
                    return

                text = f"Reminder for event: {action['data'].get('query', 'Event')}"
            else:
                text = f"Reminder: {action['data'].get('query') or action['data'].get('time', '')}"
                
            wav_path = f"rem_{reminder_id}.wav"
            success = self.tts.synthesize(text, wav_path)
            
            if success:
                try:
                    import os
                    from storage.database import get_db_conn
                    with get_db_conn(self.db.db_path) as conn:
                        rem = conn.execute("SELECT status, delivery_id FROM reminders WHERE id = ?", (reminder_id,)).fetchone()
                    
                    status_ok = rem and rem[0] == 'processing'
                    token_ok = rem and rem[1] == attempt_token
                    if not status_ok or not token_ok:
                        logger.warning(
                            f"Reminder {reminder_id} fencing failed "
                            f"(status={rem[0] if rem else 'missing'}, "
                            f"db_token={rem[1] if rem else 'missing'}, token={attempt_token}). "
                            "Aborting playback."
                        )
                        try:
                            os.remove(wav_path)
                        except Exception:
                            pass
                        raise RuntimeError(f"Late delivery aborted (attempt token mismatch or wrong status)")

                    # Renew lease for the duration of playback to prevent race conditions
                    try:
                        self.scheduler.renew_reminder_lease(reminder_id, attempt_token, lease_seconds=60)
                    except RuntimeError as e:
                        logger.warning(f"Lease renewal failed: {e}. Aborting playback.")
                        raise RuntimeError("Late delivery aborted (lease renewal failed)")

                    import threading
                    stop_hb = threading.Event()
                    def _heartbeat():
                        while not stop_hb.wait(30):
                            try:
                                self.scheduler.renew_reminder_lease(reminder_id, attempt_token, lease_seconds=60)
                            except Exception:
                                break
                    hb_t = threading.Thread(target=_heartbeat, daemon=True)
                    hb_t.start()

                    try:
                        if self.playback:
                            self.playback.play(wav_path)
                        else:
                            import winsound
                            winsound.PlaySound(wav_path, winsound.SND_FILENAME)
                    finally:
                        stop_hb.set()
                    self.scheduler.acknowledge_reminder(reminder_id, delivery_token=attempt_token)
                    try:
                        os.remove(wav_path)
                    except Exception:
                        pass
                except Exception as e:
                    logger.error(f"Failed to play reminder audio: {e}")
                    raise RuntimeError(f"Playback failed: {e}")
                finally:
                    # Failed playback is retryable and must not accumulate synthesized WAVs.
                    import os
                    try:
                        os.remove(wav_path)
                    except FileNotFoundError:
                        pass
            else:
                logger.error(f"TTS synthesis failed for reminder {reminder_id}. Delivery aborted.")
                raise RuntimeError("TTS synthesis failed")
                
        finally:
            tts_active_flag.clear()
            self.state = old_state

    def button_pressed(self):
        self._cancel_ttl()
        if self.state == ConvoState.SPEAKING:
            self.state = ConvoState.IDLE
            return
            
        if self.state in [ConvoState.IDLE, ConvoState.AWAITING_CONFIRMATION, ConvoState.CLARIFYING]:
            if self.state == ConvoState.IDLE:
                self.active_session_id = get_session_id()
                self.telemetry.log_event(self.active_session_id, "session_start", {})
                
            self.previous_state = self.state
            self.state = ConvoState.LISTENING

    def button_released(self, wav_path: str, is_temp: bool = False, trusted_input: bool = True):
        if self.state != ConvoState.LISTENING:
            return
            
        self.state = ConvoState.TRANSCRIBING
        
        from audio.recording_validation import inspect_recording
        import os as _os
        if _os.path.isfile(wav_path):
            inspection = inspect_recording(wav_path)
            if inspection.get('errors'):
                self.telemetry.record_metric("asr_vad_rejected", 1)
                self._handle_failure(f"VAD rejected: {inspection['errors'][0]}")
                if is_temp:
                    try:
                        _os.remove(wav_path)
                    except Exception:
                        pass
                return
        
        result = self.asr.transcribe(wav_path)
        
        # ... Wait, actually ASR happens here. Let's record latency.
        
        if result.get("error"):
            self.telemetry.record_metric("asr_failures", 1)
            self.telemetry.log_event(self.active_session_id, "asr_error", {"error": result.get("error")})
            logger.error(f"ASR Error: {result.get('error')}")
            self.state = self.previous_state
            if is_temp:
                try:
                    import os
                    os.remove(wav_path)
                except Exception:
                    pass
            return
            
        if is_temp:
            try:
                import os
                os.remove(wav_path)
            except Exception:
                pass
            
        if result['error']:
            self._handle_failure(f"ASR Error: {result['error']}")
            return
            
        text = result.get("text", "")
        self.last_transcript = text
        self.telemetry.log_event(self.active_session_id, "transcript_received", {"transcript": text})
        
        if self.pending_action_id and self.previous_state == ConvoState.AWAITING_CONFIRMATION:
            self._process_confirmation(text, trusted_input)
        elif self.pending_action_id and self.previous_state == ConvoState.CLARIFYING:
            self._process_clarification(text, trusted_input)
        else:
            self.telemetry.record_metric("total_intents", 1)
            self._process_new_intent(text, trusted_input)

    def _get_parser(self):
        return self.nlp_engine

    def _resolve_ambiguity_for_intent(self, intent: str, slots: dict) -> list:
        missing = []
        if intent == "create_note":
            if not slots.get("body"): missing.append("body")
        elif intent == "update_note":
            if not slots.get("body"): missing.append("body")
            if not slots.get("target_id"):
                query = slots.get("query") or ""
                notes = self.db.search_notes(query)
                if len(notes) == 1:
                    slots["target_id"] = notes[0]["id"]
                    slots["expected_revision"] = notes[0]["revision"]
                elif len(notes) > 1:
                    missing.append("ambiguous_note")
                else:
                    missing.append("note_not_found")
        elif intent == "create_event":
            if not slots.get("query"): missing.append("title")
            if not slots.get("time"): missing.append("time")
            if not slots.get("end_time"): missing.append("end_time")
        elif intent in ["update_event", "cancel_event"]:
            if not slots.get("target_id"):
                events = self.db.list_events_by_range("1970-01-01T00:00:00Z", "2099-01-01T00:00:00Z")
                query = slots.get('query', "").lower()
                matches = [e for e in events if query in e['title'].lower() or (e['title'].lower() in query and len(e['title']) > 3)]
                if len(matches) == 1:
                    slots["target_id"] = matches[0]["id"]
                    slots["expected_revision"] = matches[0]["revision"]
                elif len(matches) > 1:
                    missing.append("ambiguous_event")
                else:
                    missing.append("event_not_found")
            if intent == "update_event":
                if not slots.get("time"): missing.append("time")
        return missing

    def _speak_clarification(self, missing: list, ambiguous: list):
        if "ambiguous_note" in missing:
            self._speak("Which note? Multiple matched.")
        elif "note_not_found" in missing:
            self._speak("I couldn't find that note.")
        elif "ambiguous_event" in missing:
            self._speak("Which event? Multiple matched.")
        elif "event_not_found" in missing:
            self._speak("I couldn't find that event.")
        elif "title" in missing:
            self._speak("What is the title?")
        elif "body" in missing:
            self._speak("What should the note say?")
        elif "time" in missing:
            self._speak("At what time?")
        elif "end_time" in missing:
            self._speak("Until what time?")
        elif "am_pm" in ambiguous:
            self._speak("AM or PM?")
        elif "past_time" in ambiguous:
            self._speak("That time is in the past. Did you mean tomorrow?")
        else:
            self._speak("Please provide more details.")

    def _process_new_intent(self, text: str, trusted_input: bool = True):
        if not trusted_input:
            logger.warning("Untrusted input detected for new intent. Discarding without mutating active conversation.")
            self.state = ConvoState.IDLE
            return

        if not text:
            self._handle_failure("Silence detected. Nothing to do.")
            return
            
        parser = self._get_parser()
        result = parser.parse(text)
        self.last_parsed_intent = result["intent"]
        
        if result["intent"] == "cancel":
            self._speak("Nothing to cancel.")
            self.state = ConvoState.IDLE
            return
            
        if result["intent"] == "unknown":
            self._handle_failure("Unrelated speech or unsupported command.")
            return
            
        # Immediate intent handlers (Read-only)
        if result["intent"] == "search_notes":
            notes = self.db.search_notes(result["slots"].get("query", ""))
            if notes:
                self._speak(f"Found {len(notes)} notes.")
            else:
                self._speak("No notes found.")
            self.state = ConvoState.IDLE
            return
            
        if result["intent"] == "search_events":
            date_slot = result["slots"].get("date")
            tz = ZoneInfo('Asia/Kolkata')
            start_dt = datetime.strptime(date_slot, "%Y-%m-%d").replace(tzinfo=tz)
            end_dt = start_dt + timedelta(days=1)
            
            events = self.db.list_events_by_range(start_dt.astimezone(timezone.utc).isoformat(), end_dt.astimezone(timezone.utc).isoformat())
            if events:
                self._speak(f"You have {len(events)} events.")
            else:
                self._speak("Your calendar is clear.")
            self.state = ConvoState.IDLE
            return

        self.pending_action_id = f"act_{uuid.uuid4().hex[:8]}"
        self.pending_intent_data = result
        
        missing = result["missing"]
        missing.extend(self._resolve_ambiguity_for_intent(result["intent"], result["slots"]))
        
        if missing or result["ambiguous"]:
            self.pending_intent_data["_last_missing"] = missing
            self.state = ConvoState.CLARIFYING
            self._start_ttl()
            self._speak_clarification(missing, result["ambiguous"])
            return
            
        self._finalize_intent_for_confirmation()

    def _process_clarification(self, text: str, trusted_input: bool = True):
        if not trusted_input:
            logger.warning("Untrusted input detected in CLARIFYING state. Discarding without mutating active conversation.")
            self.state = self.previous_state
            return

        parser = self._get_parser()
        clarif_res = parser.parse(text)
        self.last_parsed_intent = clarif_res["intent"]
        if clarif_res["intent"] == "cancel":
            self.db.cancel_action(self.pending_action_id)
            self.pending_action_id = None
            self.pending_intent_data = None
            self.state = ConvoState.IDLE
            self._speak("Cancelled.")
            return

        old_missing = self.pending_intent_data.get("_last_missing", [])
        old_slots = self.pending_intent_data["slots"]

        if self.pending_intent_data.get("_correction_pending"):
            # Resolve against already-corrected slots, never by reparsing the
            # original request (which could silently resurrect its old time).
            compact = "".join(text.lower().replace(".", "").split())
            reply = text
            if compact in ("am", "pm") and old_slots.get("time"):
                hour = int(old_slots["time"].split(":")[0]) % 12 or 12
                reply = f"{hour} {compact}"
            correction = "change to " + reply
            if parser.parse(correction)["intent"] == "correction":
                self._process_confirmation(correction, trusted_input=True)
            else:
                self.state = ConvoState.CLARIFYING
                self._start_ttl()
                self._speak_clarification(
                    self.pending_intent_data.get("missing", []),
                    self.pending_intent_data.get("ambiguous", []),
                )
            return
        
        # If we were disambiguating a note or event, try to narrow using the clarification text directly
        if "ambiguous_note" in old_missing:
            query = old_slots.get("query", "")
            # Use clarification text to further filter
            notes = self.db.search_notes(query)
            clarification_lower = text.lower()
            narrowed = [n for n in notes if clarification_lower in n['title'].lower() or n['title'].lower() in clarification_lower]
            if len(narrowed) == 1:
                old_slots["target_id"] = narrowed[0]["id"]
                old_slots["expected_revision"] = narrowed[0]["revision"]
                self.pending_intent_data["_last_missing"] = []
                self._finalize_intent_for_confirmation()
                return
            elif len(narrowed) == 0:
                # Try searching the clarification text itself as query
                notes2 = self.db.search_notes(text)
                if len(notes2) == 1:
                    old_slots["target_id"] = notes2[0]["id"]
                    old_slots["expected_revision"] = notes2[0]["revision"]
                    self.pending_intent_data["_last_missing"] = []
                    self._finalize_intent_for_confirmation()
                    return
            # Still ambiguous or none found
            self.state = ConvoState.CLARIFYING
            self._start_ttl()
            self._speak("Which note? Please be more specific.")
            return
            
        if "ambiguous_event" in old_missing:
            events = self.db.list_events_by_range("1970-01-01T00:00:00Z", "2099-01-01T00:00:00Z")
            clarification_lower = text.lower()
            narrowed = [e for e in events if clarification_lower in e['title'].lower() or e['title'].lower() in clarification_lower]
            if len(narrowed) == 1:
                old_slots["target_id"] = narrowed[0]["id"]
                old_slots["expected_revision"] = narrowed[0]["revision"]
                self.pending_intent_data["_last_missing"] = []
                self._finalize_intent_for_confirmation()
                return
            self.state = ConvoState.CLARIFYING
            self._start_ttl()
            self._speak("Which event? Please be more specific.")
            return

        # Default: merge text and re-parse
        merged_text = self.pending_intent_data["original"] + " " + text
        result = parser.parse(merged_text)
        
        # Preserve target_id and expected_revision if we already resolved them
        if "target_id" in old_slots:
            result["slots"]["target_id"] = old_slots["target_id"]
            result["slots"]["expected_revision"] = old_slots["expected_revision"]

        self.pending_intent_data = result
        
        missing = result["missing"]
        missing.extend(self._resolve_ambiguity_for_intent(result["intent"], result["slots"]))
        
        if missing or result["ambiguous"]:
            self.pending_intent_data["_last_missing"] = missing
            self.state = ConvoState.CLARIFYING
            self._start_ttl()
            self._speak_clarification(missing, result["ambiguous"])
            return
            
        self._finalize_intent_for_confirmation()

    def _finalize_intent_for_confirmation(self):
        intent = self.pending_intent_data["intent"]
        slots = self.pending_intent_data["slots"]
        self.db.create_action(self.pending_action_id, intent, slots)
        self.state = ConvoState.AWAITING_CONFIRMATION
        self._start_ttl()
        
        if intent == "create_reminder":
            task = f" to {slots['query']}" if slots.get('query') else ""
            self._speak(f"Confirm reminder{task} for {slots.get('date')} at {slots.get('time')}?")
        elif intent == "create_note":
            self._speak("Confirm saving this note?")
        elif intent == "update_note":
            self._speak("Confirm updating the note?")
        elif intent == "create_event":
            self._speak(f"Confirm event from {slots.get('time')} to {slots.get('end_time')}?")
        elif intent == "update_event":
            self._speak(f"Confirm moving event to {slots.get('time')}?")
        elif intent == "cancel_event":
            self._speak(f"Confirm cancelling the event?")
        elif intent == "correction":
            self._speak(f"Confirm correction?")

    def _invalidate_event_reminders(self, cursor, event_id: str):
        # Cancel all pending reminders whose action_id matches the event_id.
        # Since we use action_id as event_id for create_event, and store target_id
        # in the action data for update_event, we check both.
        cursor.execute("SELECT id, action_id FROM reminders WHERE status IN ('pending', 'snoozed', 'processing')")
        rows = cursor.fetchall()  # Fetch all before modifying
        for rem_id, act_id in rows:
            # Check direct match first (create_event uses action_id as event_id)
            if act_id == event_id:
                cursor.execute("UPDATE reminders SET status = 'cancelled' WHERE id = ?", (rem_id,))
                continue
            # Check if action's target_id matches (for update_event actions)
            cursor.execute("SELECT encrypted_data FROM actions WHERE id = ?", (act_id,))
            act_row = cursor.fetchone()
            if act_row:
                try:
                    decrypted = self.db.cipher.decrypt(act_row[0].encode('utf-8')).decode('utf-8')
                    import json
                    act_data = json.loads(decrypted)
                    if act_data.get('target_id') == event_id:
                        cursor.execute("UPDATE reminders SET status = 'cancelled' WHERE id = ?", (rem_id,))
                except Exception:
                    pass

    def _process_confirmation(self, text: str, trusted_input: bool = True):
        if not trusted_input:
            logger.warning("Untrusted input detected in AWAITING_CONFIRMATION state. Discarding without mutating active conversation.")
            self.state = self.previous_state
            return

        parser = self._get_parser()
        result = parser.parse(text)
        self.last_parsed_intent = result["intent"]
        
        if result["intent"] == "confirm":
            self.state = ConvoState.EXECUTING
            
            def effect_callback(cursor):
                tz = ZoneInfo('Asia/Kolkata')
                slots = self.pending_intent_data["slots"]
                intent = self.pending_intent_data["intent"]
                
                if intent == "create_reminder":
                    local_dt = datetime.strptime(f"{slots['date']} {slots['time']}", "%Y-%m-%d %H:%M").replace(tzinfo=tz)
                    from scheduler.reminder_queue import ReminderQueue
                    queue = ReminderQueue(self.db.db_path)
                    queue.create_with_cursor(cursor, local_dt, 'Asia/Kolkata', self.pending_action_id, 'deliver')
                
                elif intent == "create_note":
                    title = "Note"
                    body = slots.get('body') or ""
                    import uuid
                    note_id = f"note_{uuid.uuid4().hex[:8]}"
                    enc_title = self.db.cipher.encrypt(title.encode('utf-8')).decode('utf-8')
                    enc_body = self.db.cipher.encrypt(body.encode('utf-8')).decode('utf-8')
                    cursor.execute("INSERT INTO notes (id, encrypted_title, encrypted_body, source_ref) VALUES (?, ?, ?, ?)", (note_id, enc_title, enc_body, self.pending_action_id))
                    
                elif intent == "update_note":
                    target_id = slots.get('target_id')
                    expected_rev = slots.get('expected_revision')
                    new_body = slots.get('body') or ""
                    # Read old title from same cursor to avoid opening a new connection
                    cursor.execute("SELECT encrypted_title, revision FROM notes WHERE id = ?", (target_id,))
                    note_row = cursor.fetchone()
                    if note_row:
                        enc_title = note_row[0]  # Keep existing encrypted title
                        enc_body = self.db.cipher.encrypt(new_body.encode('utf-8')).decode('utf-8')
                        cursor.execute("UPDATE notes SET encrypted_title = ?, encrypted_body = ?, revision = revision + 1, updated_at = current_timestamp WHERE id = ? AND revision = ?",
                                     (enc_title, enc_body, target_id, expected_rev))
                        if cursor.rowcount == 0:
                            raise ValueError("Concurrency conflict: revision mismatch")
                        
                elif intent == "create_event":
                    title = slots.get('query') or 'Event'
                    start_dt = datetime.strptime(f"{slots['date']} {slots['time']}", "%Y-%m-%d %H:%M").replace(tzinfo=tz)
                    end_dt = datetime.strptime(f"{slots['date']} {slots['end_time']}", "%Y-%m-%d %H:%M").replace(tzinfo=tz)
                    if end_dt <= start_dt:
                        raise ValueError("Event end time must be after start time.")
                    start_iso = start_dt.astimezone(timezone.utc).isoformat()
                    end_iso = end_dt.astimezone(timezone.utc).isoformat()
                    
                    cursor.execute("SELECT COUNT(*) FROM calendar_events WHERE start_time < ? AND end_time > ?", (end_iso, start_iso))
                    if cursor.fetchone()[0] > 0:
                        raise ValueError("Conflict detected.")
                    
                    event_id = self.pending_action_id  # Use action ID as event ID to easily track reminders
                    enc_title = self.db.cipher.encrypt(title.encode('utf-8')).decode('utf-8')
                    cursor.execute("INSERT INTO calendar_events (id, encrypted_title, start_time, end_time, display_timezone) VALUES (?, ?, ?, ?, ?)", (event_id, enc_title, start_iso, end_iso, 'Asia/Kolkata'))
                    
                    rem_dt = start_dt - timedelta(minutes=10)
                    from scheduler.reminder_queue import ReminderQueue
                    queue = ReminderQueue(self.db.db_path)
                    queue.create_with_cursor(cursor, rem_dt, 'Asia/Kolkata', self.pending_action_id, 'deliver')
                    
                elif intent == "update_event":
                    target_id = slots.get('target_id')
                    expected_rev = slots.get('expected_revision')
                    # Read event data using the same cursor to avoid opening a new connection
                    cursor.execute("SELECT encrypted_title, start_time, end_time, revision FROM calendar_events WHERE id = ?", (target_id,))
                    evt_row = cursor.fetchone()
                    if evt_row:
                        old_enc_title = evt_row[0]
                        old_start = evt_row[1]
                        old_end = evt_row[2]
                        
                        date_str = slots.get('date')
                        if date_str:
                            start_dt = datetime.strptime(f"{date_str} {slots['time']}", "%Y-%m-%d %H:%M").replace(tzinfo=tz)
                        else:
                            old_start_dt = datetime.fromisoformat(old_start).astimezone(tz)
                            time_part = datetime.strptime(slots['time'], "%H:%M").time()
                            start_dt = old_start_dt.replace(hour=time_part.hour, minute=time_part.minute)

                        end_str = slots.get('end_time')
                        if end_str:
                            end_date_str = date_str if date_str else start_dt.strftime("%Y-%m-%d")
                            end_dt = datetime.strptime(f"{end_date_str} {end_str}", "%Y-%m-%d %H:%M").replace(tzinfo=tz)
                        else:
                            old_dur = datetime.fromisoformat(old_end) - datetime.fromisoformat(old_start)
                            end_dt = start_dt + old_dur
                            
                        if end_dt <= start_dt:
                            raise ValueError("Event end time must be after start time.")
                            
                        start_iso = start_dt.astimezone(timezone.utc).isoformat()
                        end_iso = end_dt.astimezone(timezone.utc).isoformat()
                        
                        cursor.execute("SELECT COUNT(*) FROM calendar_events WHERE start_time < ? AND end_time > ? AND id != ?", (end_iso, start_iso, target_id))
                        if cursor.fetchone()[0] > 0:
                            raise ValueError("Conflict detected.")
                            
                        cursor.execute("UPDATE calendar_events SET encrypted_title = ?, start_time = ?, end_time = ?, display_timezone = ?, revision = revision + 1, updated_at = current_timestamp WHERE id = ? AND revision = ?",
                                     (old_enc_title, start_iso, end_iso, 'Asia/Kolkata', target_id, expected_rev))
                        if cursor.rowcount == 0:
                            raise ValueError("Concurrency conflict: revision mismatch")
                        
                        self._invalidate_event_reminders(cursor, target_id)
                        
                        rem_dt = start_dt - timedelta(minutes=10)
                        from scheduler.reminder_queue import ReminderQueue
                        queue = ReminderQueue(self.db.db_path)
                        queue.create_with_cursor(cursor, rem_dt, 'Asia/Kolkata', self.pending_action_id, 'deliver')
                        
                elif intent == "cancel_event":
                    target_id = slots.get('target_id')
                    expected_rev = slots.get('expected_revision')
                    cursor.execute("DELETE FROM calendar_events WHERE id = ? AND revision = ?", (target_id, expected_rev))
                    if cursor.rowcount == 0:
                        raise ValueError("Concurrency conflict: revision mismatch")
                    self._invalidate_event_reminders(cursor, target_id)
                    
            try:
                self.db.confirm_action(self.pending_action_id, effect_callback)
                self._speak("Confirmed.")
                self.state = ConvoState.IDLE
                self.pending_action_id = None
                self.pending_intent_data = None
            except Exception as e:
                self._handle_failure(str(e))
                
        elif result["intent"] == "cancel":
            self.db.cancel_action(self.pending_action_id)
            self.pending_action_id = None
            self.pending_intent_data = None
            self._speak("Cancelled.")
            self.state = ConvoState.IDLE
            
        elif result["intent"] == "correction":
            for k, v in result["slots"].items():
                if v:
                    self.pending_intent_data["slots"][k] = v
            new_result = self.pending_intent_data

            # A correction is a partial update, not a new request. Validate the
            # merged instant instead of the parser's default/current date.
            slots = new_result["slots"]
            missing = [field for field in ("date", "time") if not slots.get(field)]
            new_result["missing"] = missing
            unresolved_meridiem = (
                "am_pm" in new_result.get("ambiguous", [])
                and not result["slots"].get("time")
            )
            new_result["ambiguous"] = [
                value for value in result.get("ambiguous", []) if value != "past_time"
            ]
            if unresolved_meridiem and "am_pm" not in new_result["ambiguous"]:
                new_result["ambiguous"].append("am_pm")
            if slots.get("date") and slots.get("time") and not new_result["ambiguous"]:
                tz = ZoneInfo("Asia/Kolkata")
                now = self.clock() if self.clock else datetime.now(tz)
                if now.tzinfo is None:
                    now = now.replace(tzinfo=tz)
                try:
                    merged_dt = datetime.strptime(
                        f"{slots['date']} {slots['time']}", "%Y-%m-%d %H:%M"
                    ).replace(tzinfo=tz)
                    if merged_dt <= now:
                        new_result["ambiguous"].append("past_time")
                except ValueError:
                    new_result["ambiguous"].append("invalid_time")
            missing.extend(self._resolve_ambiguity_for_intent(new_result["intent"], new_result["slots"]))
            
            if missing or new_result["ambiguous"]:
                new_result["_correction_pending"] = True
                self.state = ConvoState.CLARIFYING
                self._start_ttl()
                self._speak_clarification(missing, new_result["ambiguous"])
                return
                
            new_result.pop("_correction_pending", None)
            old_action_id = self.pending_action_id
            self.pending_action_id = f"act_{uuid.uuid4().hex[:8]}"
            if old_action_id:
                self.db.cancel_action(old_action_id)
                
            self._finalize_intent_for_confirmation()
            
        else:
            self.state = self.previous_state
            self._start_ttl()
            self._speak("Please say yes or no.")

    def cancel(self):
        self._cancel_ttl()
        if self.pending_action_id:
            self.db.cancel_action(self.pending_action_id)
            self.pending_action_id = None
            self.pending_intent_data = None
        
        self.state = ConvoState.IDLE

    def _speak(self, text: str):
        old_state = self.state
        self.state = ConvoState.SPEAKING
        tts_active_flag.set()
        try:
            wav_path = "temp_response.wav"
            success = self.tts.synthesize(text, wav_path)
            if success:
                try:
                    import os
                    if self.playback:
                        self.playback.play(wav_path)
                    else:
                        import winsound
                        winsound.PlaySound(wav_path, winsound.SND_FILENAME)
                    try:
                        os.remove(wav_path)
                    except Exception:
                        pass
                except Exception as e:
                    logger.error(f"Failed to play audio: {e}")
                    success = False
            else:
                logger.error("TTS synthesis failed, unable to speak response.")
                success = False

            if not success:
                logger.warning("TTS failed, degrading to text/chime only.")
                try:
                    import winsound
                    winsound.MessageBeep()
                    time.sleep(0.5)
                except Exception:
                    time.sleep(0.5)
                    
        finally:
            tts_active_flag.clear()
            if self.state == ConvoState.SPEAKING:
                self.state = old_state

    def _handle_failure(self, msg: str):
        logger.error(msg)
        self.state = ConvoState.FAILED
        self._speak("I didn't catch that.")
        self.state = ConvoState.IDLE
