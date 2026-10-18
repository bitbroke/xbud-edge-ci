import json
from datetime import datetime
import logging
import re

logger = logging.getLogger(__name__)

import os

# Strict model loading is an explicit deployment choice.  Defaulting this to
# true made ordinary unit tests (and development machines without llama.cpp)
# fail before the deterministic parser fallback could be exercised.
STRICT_HARDWARE_AUDIT = os.getenv("STRICT_HARDWARE_AUDIT", "0") == "1"

class NLPEngine:
    def __init__(self, model_path: str = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf", clock=None, use_mock: bool = False):
        self.clock = clock
        self.use_mock = False if STRICT_HARDWARE_AUDIT else use_mock
        self.legacy_parser = None
        
        self.intent_schema = {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": ["ADD_REMINDER", "ADD_EVENT", "ADD_NOTE", "UPDATE_NOTE", "SEARCH_NOTE", "SEARCH_EVENT", "UPDATE_EVENT", "CANCEL_EVENT", "UNKNOWN"]
                },
                "extracted_time": {
                    "type": "string",
                    "description": "Time in 24-hour HH:MM format, or null"
                },
                "extracted_end_time": {
                    "type": "string",
                    "description": "End time in 24-hour HH:MM format for events, or null"
                },
                "extracted_date": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format, or null"
                },
                "content": {
                    "type": "string",
                    "description": "English translation of the core task, event title, note query, or note body"
                }
            },
            "required": ["intent", "content"]
        }
        
        if not self.use_mock:
            try:
                from llama_cpp import Llama
                self.llm = Llama(
                    model_path=model_path,
                    n_ctx=256,
                    n_threads=4,
                    n_gpu_layers=0,
                    chat_format="chatml",
                    verbose=False
                )
            except Exception as e:
                if STRICT_HARDWARE_AUDIT:
                    raise RuntimeError(f"HARDWARE AUDIT FAIL: Llama model failed to load. {e}")
                logger.warning(f"llama_cpp not available, falling back to legacy regex IntentParser: {e}")
                self.use_mock = True
                
        if self.use_mock:
            from nlp.intent_parser import IntentParser
            self.legacy_parser = IntentParser(reference_time=self.clock() if self.clock else None)

    def _get_now(self):
        return self.clock() if self.clock else datetime.now()

    def _fast_check(self, text: str) -> dict:
        orig = text.lower()
        is_negation = any(word in text.split() for word in ["no", "not", "nahi", "na", "radd", "chhod", "wrong"])
        is_confirmation = any(word in text.split() for word in ["yes", "haan", "ha", "okay", "ok", "thik", "sure", "correct", "right"])
        
        if is_confirmation and len(text.split()) <= 3:
            return {"intent": "confirm"}
            
        if is_negation and len(text.split()) <= 3:
            return {"intent": "cancel"}
            
        if "cancel" in text.split() and len(text.split()) <= 2:
            return {"intent": "cancel"}
            
        return None

    def parse_command(self, transcript: str) -> dict:
        now = self._get_now().strftime("%Y-%m-%d %H:%M")
        system_prompt = f"""You are an embedded intent extractor. 
        Current system time: {now}. 
        The user may speak in English, Hindi, or Hinglish. 
        Translate the core content to English and map the input to the required schema.
        Extract dates and times accurately based on the current time.
        IMPORTANT: Output ONLY a raw, unformatted JSON object. Do not use Markdown wrappers like ```json.
        Example output: {{"intent": "ADD_REMINDER", "extracted_time": "15:00", "content": "buy milk"}}"""
        
        try:
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": transcript}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=128
            )
            raw_content = response["choices"][0]["message"]["content"]
            print(f"RAW LLAMA OUTPUT: {repr(raw_content)}")
            
            if not raw_content.strip():
                logger.error("Llama output was completely empty.")
                return {"intent": "unknown", "slots": {}}
                
            return json.loads(raw_content)
        except Exception as e:
            logger.error(f"Llama inference failed: {e}")
            if 'raw_content' in locals():
                logger.error(f"Failed to parse content: {repr(raw_content)}")
            raise e

    def parse(self, text: str, context: dict = None) -> dict:
        if self.use_mock and self.legacy_parser:
            # Re-sync reference time just in case
            if self.clock:
                self.legacy_parser.reference_time = self.clock()
            return self.legacy_parser.parse(text, context)

        # Fast path for confirmations and cancellations
        fast_res = self._fast_check(text)
        if fast_res:
            return {
                "intent": fast_res["intent"],
                "slots": {"time": None, "end_time": None, "date": None, "query": None, "body": None},
                "missing": [],
                "ambiguous": [],
                "original": text
            }
            
        try:
            intent_payload = self.parse_command(text)
        except Exception:
            if self.legacy_parser:
                return self.legacy_parser.parse(text, context)
            else:
                return {"intent": "unknown", "slots": {}, "missing": [], "ambiguous": [], "original": text}
        
        # Map SLM output to internal legacy representation
        intent_map = {
            "ADD_REMINDER": "create_reminder",
            "ADD_EVENT": "create_event",
            "ADD_NOTE": "create_note",
            "UPDATE_NOTE": "update_note",
            "SEARCH_NOTE": "search_notes",
            "SEARCH_EVENT": "search_events",
            "UPDATE_EVENT": "update_event",
            "CANCEL_EVENT": "cancel_event",
            "UNKNOWN": "unknown"
        }
        
        legacy_intent = intent_map.get(intent_payload.get("intent"), "unknown")
        
        slots = {
            "time": intent_payload.get("extracted_time"),
            "end_time": intent_payload.get("extracted_end_time"),
            "date": intent_payload.get("extracted_date"),
            "query": None,
            "body": None
        }
        
        content = intent_payload.get("content")
        if legacy_intent in ["create_reminder", "create_event", "search_events", "update_event", "cancel_event", "search_notes"]:
            slots["query"] = content
        elif legacy_intent in ["create_note", "update_note"]:
            slots["body"] = content
            
        # Detect missing slots based on intent
        missing = []
        if legacy_intent == "create_reminder":
            if not slots["time"]: missing.append("time")
            if not slots["date"]: slots["date"] = self._get_now().strftime("%Y-%m-%d")
        elif legacy_intent == "create_event":
            if not slots["query"]: missing.append("title")
            if not slots["time"]: missing.append("time")
            if not slots["end_time"]: missing.append("end_time")
        elif legacy_intent == "create_note":
            if not slots["body"]: missing.append("body")
            
        return {
            "intent": legacy_intent,
            "slots": slots,
            "missing": missing,
            "ambiguous": [],
            "original": text
        }
