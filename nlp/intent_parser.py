import re
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class IntentParser:
    def __init__(self, reference_time: datetime = None):
        self.reference_time = reference_time
        
    def _get_now(self):
        return self.reference_time if self.reference_time else datetime.now()

    def parse(self, text: str, context: dict = None) -> dict:
        original_text = text
        
        # Whisper and Sherpa commonly emit dotted or spaced meridiems
        text = re.sub(r'\b([ap])\s*\.\s*m\s*\.?', r'\1m', text, flags=re.I)
        text = re.sub(r'\b([ap])\s+m\b', r'\1m', text, flags=re.I)
        text = re.sub(r'\bto\s+morrow\b', 'tomorrow', text, flags=re.I)
        # Remove punctuation to help with attached am/pm like "9am."
        text = re.sub(r'[^\w\s]', ' ', text).lower().strip()
        
        # Separate am/pm from numbers to fix matching (whisper sometimes outputs '9m' or '9am')
        text = re.sub(r'(?<=\d)(am|pm|a\.m\.|p\.m\.|a|p|m)\b', lambda m: ' am' if m.group(1) in ['a', 'm', 'am', 'a.m.'] else ' pm', text)
        
        # Word to number mapping
        num_map = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
            "ek": 1, "do": 2, "teen": 3, "chaar": 4, "paanch": 5, "chhah": 6, "saat": 7, "aath": 8, "nau": 9, "das": 10, "gyarah": 11, "barah": 12,
            "१": 1, "२": 2, "३": 3, "४": 4, "५": 5, "६": 6, "७": 7, "८": 8, "९": 9, "१०": 10, "११": 11, "१२": 12,
            "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11, "12": 12
        }

        is_negation = any(word in text.split() for word in ["no", "not", "nahi", "na", "radd", "chhod", "wrong"])
        is_confirmation = any(word in text.split() for word in ["yes", "haan", "ha", "okay", "ok", "thik", "sure", "correct", "right"])
        
        # Time matches
        time_pattern = r'\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|ek|do|teen|chaar|paanch|chhah|saat|aath|nau|das|gyarah|barah|1[0-2]|[1-9]|१[०-२]|[१-९])\b'
        time_matches = list(re.finditer(time_pattern, text))
        
        times = []
        ambiguous = []
        missing = []
        
        global_am_pm = None
        words = text.split()
        if any(w in words for w in ["am", "a.m", "subah", "morning"]):
            global_am_pm = "am"
        elif any(w in words for w in ["pm", "p.m", "shaam", "raat", "evening", "night", "dopahar", "afternoon"]):
            global_am_pm = "pm"

        for match in time_matches:
            time_val = num_map.get(match.group(1))
            if global_am_pm == "pm" and time_val < 12:
                time_val += 12
            elif global_am_pm == "am" and time_val == 12:
                time_val = 0
            times.append(f"{time_val:02d}:00")

        time_slot = times[0] if times else None
        end_time_slot = times[1] if len(times) > 1 else None

        # Date
        date_slot = None
        now = self._get_now()
        target_date = now.date()
        
        if "tomorrow" in text.split() or "kal" in text.split():
            target_date = now.date() + timedelta(days=1)
            date_slot = target_date.strftime("%Y-%m-%d")
        elif "day after" in text or "parso" in text.split():
            target_date = now.date() + timedelta(days=2)
            date_slot = target_date.strftime("%Y-%m-%d")
        elif "today" in text.split() or "aaj" in text.split():
            target_date = now.date()
            date_slot = target_date.strftime("%Y-%m-%d")

        intent = "unknown"
        query_slot = None
        body_slot = None
        
        orig_lower = original_text.lower()
        if "save a note:" in orig_lower or "save note:" in orig_lower or "create a note:" in orig_lower:
            intent = "create_note"
            parts = original_text.split(":", 1)
            body_slot = parts[1].strip() if len(parts) > 1 else ""
            if not body_slot:
                missing.append("body")
        elif "find my notes about" in text or "search notes" in text:
            intent = "search_notes"
            if "about" in text:
                parts = original_text.lower().split("about")
                query_idx = original_text.lower().find("about") + 5
                query_slot = original_text[query_idx:].strip()
            if not query_slot:
                missing.append("query")
        elif "change that note to say" in text or "update note" in text:
            intent = "update_note"
            idx = original_text.lower().find("to say")
            about_idx = original_text.lower().find("about")
            if about_idx != -1:
                if idx != -1:
                    body_slot = original_text[idx + 6:about_idx].strip()
                query_slot = original_text[about_idx + 5:].strip()
            else:
                if idx != -1:
                    body_slot = original_text[idx + 6:].strip()
            if not body_slot:
                missing.append("body")
        elif "schedule a" in text or "schedule an" in text:
            intent = "create_event"
            start_idx = text.find("schedule a") + 10
            if "schedule an" in text:
                start_idx = text.find("schedule an") + 11
            end_idx = text.find("tomorrow")
            if end_idx == -1: end_idx = text.find("today")
            if end_idx == -1: end_idx = text.find("kal")
            if end_idx == -1: end_idx = text.find("from")
            if end_idx == -1: end_idx = len(text)
            query_slot = original_text[start_idx:end_idx].strip()
            if not query_slot: missing.append("title")
            if not time_slot: missing.append("time")
            if not end_time_slot: missing.append("end_time")
        elif "what is on my calendar" in text or "what's on my calendar" in text:
            intent = "search_events"
            if not date_slot: missing.append("date")
        elif "move the" in text and "to" in text:
            intent = "update_event"
            start_idx = text.find("move the") + 8
            end_idx = text.find("to")
            query_slot = original_text[start_idx:end_idx].strip()
            if not query_slot: missing.append("title")
            if not time_slot: missing.append("time")
        elif ("cancel the" in text and "meeting" in text) or ("cancel the" in text and "demo" in text) or ("cancel the" in text and "event" in text) or ("cancel event" in text):
            intent = "cancel_event"
            start_idx = text.find("cancel the")
            if start_idx != -1:
                query_slot = original_text[start_idx + 10:].strip()
            else:
                query_slot = original_text[12:].strip()
        elif is_negation and (time_slot or date_slot):
            intent = "correction"
        elif "cancel" in text.split() and not ("note" in text or "event" in text or "demo" in text or "meeting" in text):
            intent = "cancel"
        elif is_confirmation and not time_slot and not date_slot and not body_slot:
            intent = "confirm"
        elif any(word in text.split() for word in ["remind", "yaad", "alarm", "set"]) or time_slot or date_slot or context:
            if context and context.get("state") == "CLARIFYING":
                intent = "clarification"
            else:
                intent = "create_reminder"

        if intent in ["create_reminder", "correction", "clarification"]:
            if intent == "create_reminder":
                task_match = re.search(
                    r'\bremind\s+me\s+to\s+(.+?)(?=\s+(?:tomorrow|today|at\s+|on\s+)|[.!?]*$)',
                    original_text, re.I,
                )
                if task_match:
                    query_slot = task_match.group(1).strip().rstrip('.!?')
            if times and not global_am_pm and "am_pm" not in ambiguous:
                ambiguous.append("am_pm")
            if not time_slot and "time" not in missing:
                missing.append("time")
            if not date_slot:
                date_slot = target_date.strftime("%Y-%m-%d") # default today
            if time_slot and date_slot and not ambiguous:
                try:
                    dt = datetime.strptime(f"{date_slot} {time_slot}", "%Y-%m-%d %H:%M")
                    if now.tzinfo:
                        dt = dt.replace(tzinfo=now.tzinfo)
                    if dt < now:
                        ambiguous.append("past_time")
                except ValueError:
                    pass
        elif intent in ["create_event", "update_event", "search_events"]:
            if times and not global_am_pm and intent in ["create_event", "update_event"] and "am_pm" not in ambiguous:
                ambiguous.append("am_pm")
            if not date_slot and intent != "update_event":
                date_slot = target_date.strftime("%Y-%m-%d")

        return {
            "intent": intent,
            "slots": {
                "time": time_slot,
                "end_time": end_time_slot,
                "date": date_slot,
                "query": query_slot,
                "body": body_slot
            },
            "missing": missing,
            "ambiguous": ambiguous,
            "original": original_text
        }
