import sqlite3
import uuid
from typing import Callable, Optional, List, Dict, Any
from datetime import datetime, timedelta, timezone

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from storage.database import get_db_conn

class ReminderQueue:
    def __init__(self, db_path: str, clock: Callable[[], datetime] = None):
        self.db_path = db_path
        
        def default_clock():
            return datetime.now(timezone.utc)
            
        self.clock = clock or default_clock
        self.max_retries = 3

    def _now(self) -> str:
        return self.clock().isoformat()

    def create_with_cursor(self, cursor, due_time: datetime, tz_name: str = 'UTC', action_id: str = None, missed_policy: str = 'deliver') -> str:
        reminder_id = f"rem_{uuid.uuid4().hex}"
        due_time_str = due_time.astimezone(timezone.utc).isoformat() if due_time.tzinfo else due_time.isoformat()
        
        cursor.execute(
            """
            INSERT INTO reminders 
            (id, action_id, due_time, timezone, status, attempt_count, missed_policy) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (reminder_id, action_id, due_time_str, tz_name, 'pending', 0, missed_policy)
        )
        return reminder_id

    def create(self, due_time: datetime, tz_name: str = 'UTC', action_id: str = None, missed_policy: str = 'deliver') -> str:
        with get_db_conn(self.db_path) as conn:
            cursor = conn.cursor()
            return self.create_with_cursor(cursor, due_time, tz_name, action_id, missed_policy)

    def reschedule(self, reminder_id: str, new_due_time: datetime):
        with get_db_conn(self.db_path) as conn:
            conn.execute(
                "UPDATE reminders SET due_time = ?, status = 'pending' WHERE id = ? AND status NOT IN ('delivered', 'acknowledged', 'cancelled')",
                (new_due_time.isoformat(), reminder_id)
            )

    def cancel(self, reminder_id: str):
        with get_db_conn(self.db_path) as conn:
            conn.execute(
                "UPDATE reminders SET status = 'cancelled' WHERE id = ? AND status IN ('pending', 'processing', 'snoozed')",
                (reminder_id,)
            )

    def snooze(self, reminder_id: str, snooze_minutes: int):
        new_due_time = (self.clock() + timedelta(minutes=snooze_minutes)).isoformat()
        with get_db_conn(self.db_path) as conn:
            conn.execute(
                "UPDATE reminders SET due_time = ?, status = 'snoozed' WHERE id = ? AND status NOT IN ('delivered', 'acknowledged', 'cancelled')",
                (new_due_time, reminder_id)
            )

    def acknowledge(self, reminder_id: str, delivery_token: str):
        if not delivery_token:
            raise ValueError("delivery_token is required for acknowledgment")
        with get_db_conn(self.db_path) as conn:
            conn.execute(
                "UPDATE reminders SET status = 'acknowledged' WHERE id = ? AND delivery_id = ? AND status IN ('processing', 'delivered')",
                (reminder_id, delivery_token)
            )

    def renew_lease(self, reminder_id: str, delivery_token: str, lease_seconds: int = 60):
        expiry_time_str = (self.clock() + timedelta(seconds=lease_seconds)).isoformat()
        with get_db_conn(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE reminders SET worker_lease_expiry = ? WHERE id = ? AND delivery_id = ? AND status = 'processing'",
                (expiry_time_str, reminder_id, delivery_token)
            )
            if cursor.rowcount == 0:
                raise RuntimeError("Failed to renew lease: token mismatch or not processing")

    def claim_due_reminders(self, worker_id: str, lease_seconds: int = 60, max_claims: int = 10) -> List[Dict[str, Any]]:
        current_time_str = self._now()
        expiry_time_str = (self.clock() + timedelta(seconds=lease_seconds)).isoformat()
        claimed = []
        
        with get_db_conn(self.db_path, isolation_level='EXCLUSIVE') as conn:
            try:
                conn.execute("BEGIN EXCLUSIVE")
                # Find due reminders
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, action_id, due_time, timezone, missed_policy, attempt_count 
                    FROM reminders 
                    WHERE status IN ('pending', 'snoozed') AND due_time <= ?
                    ORDER BY due_time ASC LIMIT ?
                    """,
                    (current_time_str, max_claims)
                )
                rows = cursor.fetchall()
                
                import uuid
                for row in rows:
                    r_id = row[0]
                    # Generate a unique token for this specific claim to prevent stale workers from completing it
                    # after the lease expires and it gets reclaimed.
                    delivery_token = uuid.uuid4().hex
                    
                    conn.execute(
                        """
                        UPDATE reminders 
                        SET status = 'processing', delivery_id = ?, worker_lease_expiry = ? 
                        WHERE id = ?
                        """,
                        (delivery_token, expiry_time_str, r_id)
                    )
                    claimed.append({
                        'id': r_id,
                        'action_id': row[1],
                        'due_time': row[2],
                        'timezone': row[3],
                        'missed_policy': row[4],
                        'attempt_count': row[5],
                        'delivery_id': delivery_token,
                        'delivery_token': delivery_token
                    })
                conn.commit()
            except Exception:
                conn.rollback()
                raise
                
        return claimed

    def get_next_due_time(self) -> Optional[datetime]:
        with get_db_conn(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT due_time 
                FROM reminders 
                WHERE status IN ('pending', 'snoozed')
                ORDER BY due_time ASC LIMIT 1
                """
            )
            row_due = cursor.fetchone()
            
            cursor.execute(
                """
                SELECT worker_lease_expiry 
                FROM reminders 
                WHERE status = 'processing' AND worker_lease_expiry IS NOT NULL
                ORDER BY worker_lease_expiry ASC LIMIT 1
                """
            )
            row_lease = cursor.fetchone()
            
            times = []
            if row_due:
                times.append(datetime.fromisoformat(row_due[0]))
            if row_lease:
                times.append(datetime.fromisoformat(row_lease[0]))
                
            if times:
                return min(times)
            return None

    def complete_delivery(self, reminder_id: str, delivery_id: str, success: bool):
        with get_db_conn(self.db_path, isolation_level='EXCLUSIVE') as conn:
            try:
                conn.execute("BEGIN EXCLUSIVE")
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT attempt_count, status FROM reminders WHERE id = ? AND delivery_id = ?",
                    (reminder_id, delivery_id)
                )
                row = cursor.fetchone()
                if not row:
                    conn.rollback()
                    return # Could be already completed or canceled, or wrong delivery_id

                attempt_count, status = row
                if status != 'processing':
                    conn.rollback()
                    return

                new_attempt_count = attempt_count + 1
                if success:
                    new_status = 'delivered'
                else:
                    new_status = 'pending' if new_attempt_count < self.max_retries else 'failed'

                conn.execute(
                    """
                    UPDATE reminders 
                    SET status = ?, attempt_count = ?, worker_lease_expiry = NULL, delivery_id = NULL
                    WHERE id = ? AND delivery_id = ?
                    """,
                    (new_status, new_attempt_count, reminder_id, delivery_id)
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def recover_stale_leases(self):
        current_time_str = self._now()
        with get_db_conn(self.db_path) as conn:
            conn.execute(
                """
                UPDATE reminders 
                SET status = 'pending', worker_lease_expiry = NULL, delivery_id = NULL
                WHERE status = 'processing' AND worker_lease_expiry < ?
                """,
                (current_time_str,)
            )

    def process_missed_reminders(self, cutoff_minutes: int = 60):
        # Apply the missed_policy for reminders that are way past due and still pending
        cutoff_time_str = (self.clock() - timedelta(minutes=cutoff_minutes)).isoformat()
        with get_db_conn(self.db_path) as conn:
            # Drop/skip policy
            conn.execute(
                """
                UPDATE reminders 
                SET status = 'missed' 
                WHERE status IN ('pending', 'snoozed') AND due_time < ? AND missed_policy = 'skip'
                """,
                (cutoff_time_str,)
            )
            # 'deliver' policy stays pending, will naturally be picked up by claim_due_reminders
