import time
import queue
import logging
import threading
from typing import Any
from dataclasses import dataclass
try:
    from .conversation import ConversationManager, ConvoState
except ImportError:
    ConversationManager = None
    ConvoState = None

logger = logging.getLogger(__name__)

@dataclass
class TriggerDownEvent:
    pass

@dataclass
class TriggerUpEvent:
    pass

@dataclass
class AudioCapturedEvent:
    wav_path: str

@dataclass
class ReminderDueEvent:
    reminder_id: str
    action_id: str
    due_time: str
    attempt_token: int
    completion_event: threading.Event
    outcome: list

class EventCoordinator:
    def __init__(self, convo: ConversationManager, audio_recorder: Any):
        self.convo = convo
        self.audio_recorder = audio_recorder
        self.event_queue = queue.Queue()
        self.running = False
        self.thread = None
        self._button_pressed = False

    def enqueue(self, event):
        if not self.running:
            return
        self.event_queue.put(event)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self, timeout=60.0):
        self.running = False
        # Drain queue to resolve pending callbacks
        while not self.event_queue.empty():
            try:
                ev = self.event_queue.get_nowait()
                if isinstance(ev, ReminderDueEvent):
                    ev.outcome.append(RuntimeError("Coordinator shutting down"))
                    ev.completion_event.set()
            except queue.Empty:
                break
        
        if self.thread:
            self.thread.join(timeout=timeout)
            if self.thread.is_alive():
                raise RuntimeError("Coordinator did not finish active work before shutdown timeout")

    def _loop(self):
        while self.running:
            try:
                event = self.event_queue.get(timeout=0.1)
                self._process_event(event)
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"Coordinator error: {e}", exc_info=True)

    def _process_event(self, event):
        if isinstance(event, TriggerDownEvent):
            if self.convo.state == ConvoState.SPEAKING:
                # Interrupt speech
                self.convo.button_pressed()
                return

            if self.convo.state in [ConvoState.IDLE, ConvoState.AWAITING_CONFIRMATION, ConvoState.CLARIFYING]:
                if not self._button_pressed:
                    self._button_pressed = True
                    self.convo.button_pressed()
                    self.audio_recorder.start_recording()

        elif isinstance(event, TriggerUpEvent):
            if self._button_pressed:
                self._button_pressed = False
                wav_path = self.audio_recorder.stop_recording()
                if wav_path:
                    # Valid recording
                    self.convo.button_released(wav_path, is_temp=True)
                else:
                    # Empty or aborted recording
                    self.convo.state = self.convo.previous_state

        elif isinstance(event, ReminderDueEvent):
            # Process reminder through the state machine
            try:
                self.convo._on_reminder_due(event.reminder_id, event.action_id, event.due_time, event.attempt_token)
                event.outcome.append(True)
            except Exception as e:
                event.outcome.append(e)
            finally:
                event.completion_event.set()

    def reminder_callback(self, rem_id, action_id, due_time, attempt_token):
        if not self.running:
            raise RuntimeError("Coordinator shutting down")
            
        completion_event = threading.Event()
        outcome = []
        self.enqueue(ReminderDueEvent(rem_id, action_id, due_time, attempt_token, completion_event, outcome))
        # Bound the wait in case of deadlock or shutdown
        completed = completion_event.wait(timeout=30.0)
        if not completed:
            raise RuntimeError("Reminder playback timed out")
            
        if outcome and isinstance(outcome[0], Exception):
            raise outcome[0]
