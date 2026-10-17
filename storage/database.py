import sqlite3
import os
import json
import logging
from cryptography.fernet import Fernet, InvalidToken
from contextlib import contextmanager
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class InvalidKeyError(Exception):
    """Raised when the provided encryption key fails the canary check."""
    pass

class ActionAlreadyConfirmedError(Exception):
    """Raised when trying to confirm an action that is not in pending state."""
    pass

@contextmanager
def get_db_conn(db_path, isolation_level=None):
    conn = sqlite3.connect(db_path, isolation_level=isolation_level)
    try:
        if isolation_level == 'EXCLUSIVE':
            yield conn
        else:
            with conn:
                yield conn
    finally:
        conn.close()

class StorageManager:
    def __init__(self, db_path: str, key: bytes = None):
        self.db_path = db_path
        self._init_db()
        self.key = self._load_or_create_key(key)
        self.cipher = Fernet(self.key)
        self._verify_key()

    def _init_db(self):
        schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
        with open(schema_path, 'r') as f:
            schema = f.read()
            
        with get_db_conn(self.db_path) as conn:
            conn.executescript(schema)
            
    def _load_or_create_key(self, provided_key: bytes) -> bytes:
        if provided_key:
            return provided_key
            
        key_path = self.db_path + '.key'
        if os.path.exists(key_path):
            with open(key_path, 'rb') as f:
                return f.read()
        
        with get_db_conn(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM meta WHERE key='canary'")
            row = cursor.fetchone()
            if row:
                raise ValueError("Database exists but no key provided and no .key file found.")
                
        new_key = Fernet.generate_key()
        with open(key_path, 'wb') as f:
            f.write(new_key)
        
        try:
            os.chmod(key_path, 0o600)
        except Exception:
            pass
            
        return new_key

    def _verify_key(self):
        with get_db_conn(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM meta WHERE key='canary'")
            row = cursor.fetchone()
            if row:
                encrypted_canary = row[0].encode('utf-8')
                try:
                    self.cipher.decrypt(encrypted_canary)
                except InvalidToken:
                    raise InvalidKeyError("Wrong key provided. Cannot decrypt database fields.")
            else:
                encrypted_canary = self.cipher.encrypt(b"earbud_valid_key").decode('utf-8')
                cursor.execute("INSERT INTO meta (key, value) VALUES ('canary', ?)", (encrypted_canary,))

    def create_action(self, action_id: str, intent: str, data: dict, idempotency_key: str = None):
        encrypted_data = self.cipher.encrypt(json.dumps(data).encode('utf-8')).decode('utf-8')
        try:
            with get_db_conn(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO actions (id, intent, state, encrypted_data, idempotency_key) VALUES (?, ?, ?, ?, ?)",
                    (action_id, intent, 'pending', encrypted_data, idempotency_key)
                )
        except sqlite3.IntegrityError:
            pass

    def get_action(self, action_id: str):
        with get_db_conn(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT intent, state, encrypted_data, idempotency_key FROM actions WHERE id = ?", (action_id,))
            row = cursor.fetchone()
            if not row:
                return None
            
            intent, state, encrypted_data, idempotency_key = row
            data = json.loads(self.cipher.decrypt(encrypted_data.encode('utf-8')).decode('utf-8'))
            return {
                'id': action_id,
                'intent': intent,
                'state': state,
                'data': data,
                'idempotency_key': idempotency_key
            }

    def confirm_action(self, action_id: str, effect_callback=None):
        with get_db_conn(self.db_path, isolation_level='EXCLUSIVE') as conn:
            try:
                cursor = conn.cursor()
                cursor.execute("BEGIN EXCLUSIVE")
                cursor.execute("SELECT state FROM actions WHERE id = ?", (action_id,))
                row = cursor.fetchone()
                
                if not row:
                    raise ValueError(f"Action {action_id} not found.")
                if row[0] != 'pending':
                    raise ActionAlreadyConfirmedError(f"Action {action_id} is in state '{row[0]}'.")
                    
                if effect_callback:
                    effect_callback(cursor)
                    
                cursor.execute("UPDATE actions SET state = 'confirmed', updated_at = CURRENT_TIMESTAMP WHERE id = ?", (action_id,))
                conn.commit()
            except:
                conn.rollback()
                raise

    def cancel_action(self, action_id: str):
        with get_db_conn(self.db_path, isolation_level='EXCLUSIVE') as conn:
            try:
                conn.execute("BEGIN EXCLUSIVE")
                conn.execute("UPDATE actions SET state = 'cancelled', updated_at = CURRENT_TIMESTAMP WHERE id = ? AND state = 'pending'", (action_id,))
                conn.commit()
            except:
                conn.rollback()
                raise

    # ---------------------------------------------------------
    # NOTES API
    # ---------------------------------------------------------

    def create_note(self, title: str, body: str, source_ref: str = None) -> str:
        import uuid
        note_id = f"note_{uuid.uuid4().hex[:8]}"
        enc_title = self.cipher.encrypt(title.encode('utf-8')).decode('utf-8')
        enc_body = self.cipher.encrypt(body.encode('utf-8')).decode('utf-8')
        
        with get_db_conn(self.db_path) as conn:
            conn.execute(
                "INSERT INTO notes (id, encrypted_title, encrypted_body, source_ref) VALUES (?, ?, ?, ?)",
                (note_id, enc_title, enc_body, source_ref)
            )
        return note_id

    def get_note(self, note_id: str) -> dict:
        with get_db_conn(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT encrypted_title, encrypted_body, revision, source_ref, created_at, updated_at FROM notes WHERE id = ?", (note_id,))
            row = cursor.fetchone()
            if not row:
                return None
            
            title = self.cipher.decrypt(row[0].encode('utf-8')).decode('utf-8')
            body = self.cipher.decrypt(row[1].encode('utf-8')).decode('utf-8')
            return {
                'id': note_id,
                'title': title,
                'body': body,
                'revision': row[2],
                'source_ref': row[3],
                'created_at': row[4],
                'updated_at': row[5]
            }

    def update_note(self, note_id: str, title: str, body: str, expected_revision: int):
        enc_title = self.cipher.encrypt(title.encode('utf-8')).decode('utf-8')
        enc_body = self.cipher.encrypt(body.encode('utf-8')).decode('utf-8')
        
        with get_db_conn(self.db_path, isolation_level='EXCLUSIVE') as conn:
            cursor = conn.cursor()
            cursor.execute("BEGIN EXCLUSIVE")
            cursor.execute("SELECT revision FROM notes WHERE id = ?", (note_id,))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Note {note_id} not found.")
            if row[0] != expected_revision:
                raise ValueError(f"Revision mismatch for note {note_id}. Expected {expected_revision}, got {row[0]}")
                
            cursor.execute(
                "UPDATE notes SET encrypted_title = ?, encrypted_body = ?, revision = revision + 1, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (enc_title, enc_body, note_id)
            )
            conn.commit()

    def delete_note(self, note_id: str, expected_revision: int):
        with get_db_conn(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM notes WHERE id = ? AND revision = ?", (note_id, expected_revision))
            if cursor.rowcount == 0:
                raise ValueError("Concurrency conflict: revision mismatch")

    def list_notes(self) -> list:
        notes = []
        with get_db_conn(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, encrypted_title, encrypted_body, revision, source_ref, created_at, updated_at FROM notes")
            for row in cursor.fetchall():
                title = self.cipher.decrypt(row[1].encode('utf-8')).decode('utf-8')
                body = self.cipher.decrypt(row[2].encode('utf-8')).decode('utf-8')
                notes.append({
                    'id': row[0],
                    'title': title,
                    'body': body,
                    'revision': row[3],
                    'source_ref': row[4],
                    'created_at': row[5],
                    'updated_at': row[6]
                })
        return notes

    def search_notes(self, query: str) -> list:
        query_lower = query.lower()
        notes = self.list_notes()
        results = []
        for note in notes:
            dec_title = note['title']
            dec_body = note['body']
            q = query_lower
            dt = dec_title.lower()
            db = dec_body.lower()
            if q in dt or q in db or (dt in q and len(dt) > 3):
                results.append(note)
        return results

    # ---------------------------------------------------------
    # CALENDAR EVENTS API
    # ---------------------------------------------------------

    def create_event(self, title: str, start_time: str, end_time: str, display_timezone: str, event_id: str = None) -> str:
        try:
            start_dt = datetime.fromisoformat(start_time)
            end_dt = datetime.fromisoformat(end_time)
            if start_dt.tzinfo is None or end_dt.tzinfo is None:
                raise ValueError("Timestamps must be timezone-aware")
            start_iso = start_dt.astimezone(timezone.utc).isoformat()
            end_iso = end_dt.astimezone(timezone.utc).isoformat()
        except ValueError as e:
            if "timezone-aware" not in str(e):
                raise ValueError("Invalid ISO format")
            raise

        if end_iso <= start_iso:
            raise ValueError("Event end time must be strictly after start time")
        import uuid
        if not event_id:
            event_id = f"evt_{uuid.uuid4().hex[:8]}"
        enc_title = self.cipher.encrypt(title.encode('utf-8')).decode('utf-8')
        
        with get_db_conn(self.db_path) as conn:
            conn.execute(
                "INSERT INTO calendar_events (id, encrypted_title, start_time, end_time, display_timezone) VALUES (?, ?, ?, ?, ?)",
                (event_id, enc_title, start_iso, end_iso, display_timezone)
            )
        return event_id

    def get_event(self, event_id: str) -> dict:
        with get_db_conn(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT encrypted_title, start_time, end_time, display_timezone, revision, created_at, updated_at FROM calendar_events WHERE id = ?", (event_id,))
            row = cursor.fetchone()
            if not row:
                return None
            
            title = self.cipher.decrypt(row[0].encode('utf-8')).decode('utf-8')
            return {
                'id': event_id,
                'title': title,
                'start_time': row[1],
                'end_time': row[2],
                'display_timezone': row[3],
                'revision': row[4],
                'created_at': row[5],
                'updated_at': row[6]
            }

    def update_event(self, event_id: str, title: str, start_time: str, end_time: str, display_timezone: str, expected_revision: int):
        try:
            start_dt = datetime.fromisoformat(start_time)
            end_dt = datetime.fromisoformat(end_time)
            if start_dt.tzinfo is None or end_dt.tzinfo is None:
                raise ValueError("Timestamps must be timezone-aware")
            start_iso = start_dt.astimezone(timezone.utc).isoformat()
            end_iso = end_dt.astimezone(timezone.utc).isoformat()
        except ValueError as e:
            if "timezone-aware" not in str(e):
                raise ValueError("Invalid ISO format")
            raise

        if end_iso <= start_iso:
            raise ValueError("Event end time must be strictly after start time")
        enc_title = self.cipher.encrypt(title.encode('utf-8')).decode('utf-8')
        
        with get_db_conn(self.db_path, isolation_level='EXCLUSIVE') as conn:
            cursor = conn.cursor()
            cursor.execute("BEGIN EXCLUSIVE")
            cursor.execute("SELECT revision FROM calendar_events WHERE id = ?", (event_id,))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Event {event_id} not found.")
            if row[0] != expected_revision:
                raise ValueError(f"Revision mismatch for event {event_id}. Expected {expected_revision}, got {row[0]}")
                
            cursor.execute(
                "UPDATE calendar_events SET encrypted_title = ?, start_time = ?, end_time = ?, display_timezone = ?, revision = revision + 1, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (enc_title, start_time, end_time, display_timezone, event_id)
            )
            conn.commit()

    def delete_event(self, event_id: str, expected_revision: int):
        with get_db_conn(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM calendar_events WHERE id = ? AND revision = ?", (event_id, expected_revision))
            if cursor.rowcount == 0:
                raise ValueError("Concurrency conflict: revision mismatch")

    def list_events_by_range(self, start_iso: str, end_iso: str) -> list:
        events = []
        with get_db_conn(self.db_path) as conn:
            cursor = conn.cursor()
            # Overlap condition: e.start < query.end AND e.end > query.start
            cursor.execute(
                "SELECT id, encrypted_title, start_time, end_time, display_timezone, revision, created_at, updated_at FROM calendar_events WHERE start_time < ? AND end_time > ?",
                (end_iso, start_iso)
            )
            for row in cursor.fetchall():
                title = self.cipher.decrypt(row[1].encode('utf-8')).decode('utf-8')
                events.append({
                    'id': row[0],
                    'title': title,
                    'start_time': row[2],
                    'end_time': row[3],
                    'display_timezone': row[4],
                    'revision': row[5],
                    'created_at': row[6],
                    'updated_at': row[7]
                })
        return events

    def check_conflict(self, start_iso: str, end_iso: str, exclude_event_id: str = None) -> bool:
        with get_db_conn(self.db_path) as conn:
            cursor = conn.cursor()
            if exclude_event_id:
                cursor.execute(
                    "SELECT 1 FROM calendar_events WHERE start_time < ? AND end_time > ? AND id != ? LIMIT 1",
                    (end_iso, start_iso, exclude_event_id)
                )
            else:
                cursor.execute(
                    "SELECT 1 FROM calendar_events WHERE start_time < ? AND end_time > ? LIMIT 1",
                    (end_iso, start_iso)
                )
            return cursor.fetchone() is not None

    def backup(self, backup_path: str = None) -> str:
        """Create an online hot backup snapshot of the database and its encryption key."""
        if not backup_path:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.db_path}.{timestamp}.bak"
        with get_db_conn(self.db_path) as src_conn:
            dst_conn = sqlite3.connect(backup_path)
            try:
                src_conn.backup(dst_conn)
            finally:
                dst_conn.close()
        key_src = self.db_path + ".key"
        if os.path.exists(key_src):
            import shutil
            shutil.copy2(key_src, backup_path + ".key")
        return backup_path

    @classmethod
    def restore_from_backup(cls, backup_path: str, target_db_path: str):
        """Restore a database and encryption key from a backup snapshot."""
        import shutil
        if not os.path.exists(backup_path):
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        os.makedirs(os.path.dirname(os.path.abspath(target_db_path)), exist_ok=True)
        shutil.copy2(backup_path, target_db_path)
        key_src = backup_path + ".key"
        if os.path.exists(key_src):
            shutil.copy2(key_src, target_db_path + ".key")
        return cls(target_db_path)
