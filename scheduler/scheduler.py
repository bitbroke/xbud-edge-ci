import time
import threading
import sqlite3
import logging
from datetime import datetime, timezone
import uuid

logger = logging.getLogger(__name__)

class ReminderScheduler:
    def __init__(self, db_path: str, poll_interval: float = 1.0):
        from scheduler.reminder_queue import ReminderQueue
        self.db_path = db_path
        self.poll_interval = poll_interval
        self.running = False
        self.thread = None
        self.interrupt_event = threading.Event()
        self.callbacks = []
        self.worker_id = f"worker_{uuid.uuid4().hex[:8]}"
        self.queue = ReminderQueue(db_path)
        
    def start(self):
        self.running = True
        self._reconcile_startup()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        
    def interrupt(self):
        """Wake up the scheduler immediately to recalculate wait time."""
        self.interrupt_event.set()
        
    def stop(self, timeout=35.0):
        self.running = False
        self.interrupt()  # Wake up the thread so it can exit
        if self.thread:
            self.thread.join(timeout=timeout)
            if self.thread.is_alive():
                raise RuntimeError("Scheduler did not finish active delivery before shutdown timeout")
            
    def register_delivery_callback(self, callback):
        """Register a function to be called when a reminder is due.
        callback signature: callback(reminder_id, action_id, due_time)"""
        self.callbacks.append(callback)
        
    def schedule_reminder(self, action_id: str, due_time: str, timezone: str = 'UTC', missed_policy: str = 'deliver'):
        from datetime import datetime
        dt = datetime.fromisoformat(due_time)
        return self.queue.create(due_time=dt, tz_name=timezone, action_id=action_id, missed_policy=missed_policy)
        
    def _reconcile_startup(self):
        """Handle reminders that were missed during downtime and stale leases."""
        self.queue.recover_stale_leases()
        self.queue.process_missed_reminders(cutoff_minutes=60)
            
    def _loop(self):
        def block_until_time_sync():
            import datetime, time
            # If the year is before 2026, the CM5 clock has not synced.
            while datetime.datetime.now().year < 2026:
                logger.info("System clock unverified. Waiting for NTP sync...")
                time.sleep(5)
            logger.info("Clock synced. Starting ReminderScheduler.")
            
        block_until_time_sync()
        
        while self.running:
            try:
                self.queue.recover_stale_leases()
                
                if not self.callbacks:
                    self.interrupt_event.wait(self.poll_interval)
                    self.interrupt_event.clear()
                    continue
                    
                # Claim any already due reminders
                claims = self.queue.claim_due_reminders(self.worker_id, lease_seconds=60, max_claims=10)
                
                if claims:
                    for claim in claims:
                        if not self.running:
                            break
                        rem_id = claim['id']
                        action_id = claim['action_id']
                        due_time = claim['due_time']
                        delivery_token = claim['delivery_token']
                        
                        success = True
                        for cb in self.callbacks:
                            try:
                                cb(rem_id, action_id, due_time, delivery_token)
                            except Exception as e:
                                logger.error(f"Delivery callback failed: {e}")
                                success = False
                                
                        self.queue.complete_delivery(rem_id, delivery_token, success)
                    continue  # Loop immediately if we processed claims to check for more
                
                # No claims were due right now, so sleep until the next one
                next_due = self.queue.get_next_due_time()
                if next_due:
                    now = self.queue.clock()
                    delta = (next_due - now).total_seconds()
                    wait_time = max(0.1, min(delta, 3600.0))  # wait up to 1 hour max, but at least 100ms
                else:
                    wait_time = 3600.0  # sleep for an hour if nothing is scheduled
                    
                self.interrupt_event.wait(wait_time)
                self.interrupt_event.clear()
                        
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                self.interrupt_event.wait(self.poll_interval)
                self.interrupt_event.clear()
            
    def acknowledge_reminder(self, reminder_id: str, delivery_token: str):
        self.queue.acknowledge(reminder_id, delivery_token)

    def renew_reminder_lease(self, reminder_id: str, delivery_token: str, lease_seconds: int = 60):
        self.queue.renew_lease(reminder_id, delivery_token, lease_seconds)

    def cancel_reminder(self, reminder_id: str):
        self.queue.cancel(reminder_id)
