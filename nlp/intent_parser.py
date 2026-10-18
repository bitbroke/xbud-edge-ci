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
        text = re.sub(r"\bma['’]?am\b", "maam", text, flags=re.I)
        text = re.sub(r"\bi\s*['’]?\s*m\b", "im", text, flags=re.I)
        text = re.sub(r'\bup\s+date\b', 'update', text, flags=re.I)
        text = re.sub(r'\bwife\s+i\b', 'wifi', text, flags=re.I)
        text = re.sub(r'\bpass\s+word\b', 'password', text, flags=re.I)
        text = re.sub(r'\b([ap])\s*\.\s*m\s*\.?', r'\1m', text, flags=re.I)
        text = re.sub(r'\b([ap])\s+m\b', r'\1m', text, flags=re.I)
        text = re.sub(r'\bto\s+morrow\b', 'tomorrow', text, flags=re.I)
        # Remove punctuation to help with attached am/pm like "9am."
        text = re.sub(r'[^\w\s]', ' ', text).lower().strip()
        
        # Separate am/pm from numbers to fix matching (whisper sometimes outputs '9m' or '9am')
        text = re.sub(r'(?<=\d)(am|pm|a\.m\.|p\.m\.|a|p|m)\b', lambda m: ' am' if m.group(1) in ['a', 'm', 'am', 'a.m.'] else ' pm', text)
        
        # Word to number mapping
        num_map = {
            "noon": 12, "midday": 12,
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
            "ek": 1, "do": 2, "teen": 3, "chaar": 4, "paanch": 5, "chhah": 6, "saat": 7, "aath": 8, "nau": 9, "das": 10, "gyarah": 11, "barah": 12,
            "१": 1, "२": 2, "३": 3, "४": 4, "५": 5, "६": 6, "७": 7, "८": 8, "९": 9, "१०": 10, "११": 11, "१२": 12,
            "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11, "12": 12
        }

        is_negation = any(word in text.split() for word in ["no", "not", "nahi", "na", "radd", "chhod", "wrong"])
        is_confirmation = any(word in text.split() for word in ["yes", "haan", "ha", "okay", "ok", "thik", "sure", "correct", "right"])
        
        # Time matches
        time_pattern = r'\b(noon|midday|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|ek|do|teen|chaar|paanch|chhah|saat|aath|nau|das|gyarah|barah|1[0-2]|[1-9]|१[०-२]|[१-९])\b'
        time_matches = list(re.finditer(time_pattern, text))
        
        times = []
        ambiguous = []
        missing = []
        
        global_am_pm = None
        words = text.split()
        if any(w in words for w in ["noon", "midday"]) or re.search(r'\b(?:at\s+)?(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+pm\b', text) or any(w in words for w in ["pm", "p.m", "shaam", "raat", "evening", "night", "dopahar", "afternoon"]):
            global_am_pm = "pm"
        elif re.search(r'\b(?:at\s+)?(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+am\b', text) or any(w in words for w in ["am", "a.m", "subah", "morning"]):
            global_am_pm = "am"

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
        note_match = re.search(r'\b(?:save|create|add|take)\s+(?:a\s+)?note(?:\s*:\s*|\s+that\s+|\s+about\s+|\s+to\s+|\s+)(.+)$', original_text, re.I)
        if not note_match:
            note_match = re.search(r'\bnote\s+down\s+(.+)$', original_text, re.I)

        if note_match:
            intent = "create_note"
            body_slot = note_match.group(1).strip()
            if not body_slot:
                missing.append("body")
        elif any(phrase in text for phrase in ["find my notes", "search notes", "search note", "find note"]):
            intent = "search_notes"
            if "about" in text:
                query_idx = text.find("about") + 5
                query_slot = text[query_idx:].strip()
            elif "for" in text:
                query_idx = text.find("for") + 3
                query_slot = text[query_idx:].strip()
            else:
                m_search = re.search(r'\b(?:search|find)\s+(?:my\s+)?notes?\s+(.+)$', text, re.I)
                query_slot = m_search.group(1).strip() if m_search else ""
            if query_slot:
                query_slot = re.sub(r'^(?:a|an|the)\s+', '', query_slot, flags=re.I).strip()
                query_slot = re.sub(r'\s+(?:say|please)$', '', query_slot, flags=re.I).strip()
            if not query_slot:
                missing.append("query")
        elif "change that note to say" in text or "update note" in text:
            intent = "update_note"
            m_up = re.search(r'\bupdate\s+note\s+(.+?)\s+to\s+(.+)$', text, re.I)
            if m_up:
                query_slot = m_up.group(1).strip()
                body_slot = m_up.group(2).strip()
            else:
                idx = original_text.lower().find("to say")
                if idx == -1: idx = original_text.lower().find("to")
                about_idx = original_text.lower().find("about")
                if about_idx != -1:
                    if idx != -1:
                        body_slot = original_text[idx + 6:about_idx].strip()
                    query_slot = original_text[about_idx + 5:].strip()
                elif idx != -1:
                    body_slot = original_text[idx + 2:].strip()
            if query_slot:
                query_slot = re.sub(r'^(?:a|an|the)\s+', '', query_slot, flags=re.I).strip()
            if not body_slot:
                missing.append("body")
        elif any(p in text for p in ["schedule a", "schedule an", "schedule event", "add event", "at event", "create event", "new event"]):
            intent = "create_event"
            m_ev = re.search(r'\b(?:schedule\s+(?:an?\s+)?|(?:add|at)\s+event\s+|create\s+event\s+|new\s+event\s+)(.+?)(?=\s+(?:tomorrow|today|kal|from|at\s+|on\s+)|$)', original_text, re.I)
            if m_ev:
                query_slot = m_ev.group(1).strip()
            else:
                start_idx = text.find("schedule a") + 10 if "schedule a" in text else 0
                end_idx = text.find("tomorrow") if "tomorrow" in text else len(text)
                query_slot = original_text[start_idx:end_idx].strip()
            if not query_slot: missing.append("title")
            if not time_slot: missing.append("time")
            if not end_time_slot: missing.append("end_time")
        elif any(p in text for p in ["what is on my calendar", "what's on my calendar", "search event", "search events", "check calendar"]):
            intent = "search_events"
            m_sev = re.search(r'\b(?:search\s+events?\s+(?:for\s+)?)(.+)$', original_text, re.I)
            if m_sev:
                query_slot = m_sev.group(1).strip()
            if not date_slot:
                date_slot = target_date.strftime("%Y-%m-%d")
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
        elif (is_negation or any(p in text for p in ["change time", "change to", "make it", "instead", "badal", "nahi"])) and (time_slot or date_slot):
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
