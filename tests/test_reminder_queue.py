import os
import sqlite3
import pytest
import threading
from datetime import datetime, timedelta
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from storage.database import StorageManager, get_db_conn
from scheduler.reminder_queue import ReminderQueue
from scheduler.output_adapter import MockNotificationSink

class MockClock:
    def __init__(self, start_time: datetime):
        self.current_time = start_time
    def __call__(self):
        return self.current_time
    def advance(self, delta: timedelta):
        self.current_time += delta

@pytest.fixture
def temp_db(tmp_path):
    db_path = str(tmp_path / "test_reminders.db")
    StorageManager(db_path) # Initialize schema and keys
    yield db_path
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            os.remove(db_path + '.key')
        except OSError:
            pass

def test_restart_persistence(temp_db):
    clock = MockClock(datetime(2026, 9, 18, 12, 0, 0))
    queue1 = ReminderQueue(temp_db, clock=clock)
    due_time = clock() + timedelta(minutes=5)
    r_id = queue1.create(due_time=due_time)
    
    # Simulate restart
    queue2 = ReminderQueue(temp_db, clock=clock)
    clock.advance(timedelta(minutes=10))
    claimed = queue2.claim_due_reminders("worker_1")
    
    assert len(claimed) == 1
    assert claimed[0]['id'] == r_id

def test_concurrent_workers_claim_once(temp_db):
    clock = MockClock(datetime(2026, 9, 18, 12, 0, 0))
    queue = ReminderQueue(temp_db, clock=clock)
    queue.create(due_time=clock())
    
    claims = []
    
    def worker_claim(worker_id):
        q = ReminderQueue(temp_db, clock=clock)
        claimed = q.claim_due_reminders(worker_id)
        claims.append(claimed)
        
    t1 = threading.Thread(target=worker_claim, args=("worker_1",))
    t2 = threading.Thread(target=worker_claim, args=("worker_2",))
    
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
    total_claimed = sum(len(c) for c in claims)
    assert total_claimed == 1

def test_cancellation_and_reschedule_invalidation(temp_db):
    clock = MockClock(datetime(2026, 9, 18, 12, 0, 0))
    queue = ReminderQueue(temp_db, clock=clock)
    
    r_cancel = queue.create(due_time=clock())
    r_resched = queue.create(due_time=clock())
    
    queue.cancel(r_cancel)
    queue.reschedule(r_resched, clock() + timedelta(minutes=60))
    
    claimed = queue.claim_due_reminders("w1")
    assert len(claimed) == 0 # Cancelled one shouldn't be claimed, rescheduled one is in future

def test_crash_recovery(temp_db):
    clock = MockClock(datetime(2026, 9, 18, 12, 0, 0))
    queue = ReminderQueue(temp_db, clock=clock)
    
    r1 = queue.create(due_time=clock())
    
    # Claim it but crash before completing
    claimed = queue.claim_due_reminders("w_crash", lease_seconds=60)
    assert len(claimed) == 1
    
    # Another worker tries immediately, gets nothing
    assert len(queue.claim_due_reminders("w_new")) == 0
    
    # Clock advances past lease
    clock.advance(timedelta(seconds=61))
    
    # Recovery run
    queue.recover_stale_leases()
    
    # Can claim again
    claimed2 = queue.claim_due_reminders("w_new")
    assert len(claimed2) == 1
    assert claimed2[0]['id'] == r1

def test_snooze_and_ack_persistence(temp_db):
    clock = MockClock(datetime(2026, 9, 18, 12, 0, 0))
    queue = ReminderQueue(temp_db, clock=clock)
    
    r_snooze = queue.create(due_time=clock())
    r_ack = queue.create(due_time=clock())
    
    queue.snooze(r_snooze, 10)
    ack_claim = queue.claim_due_reminders("w_ack")[0]
    assert ack_claim['id'] == r_ack
    queue.acknowledge(r_ack, ack_claim['delivery_token'])
    
    # Verify neither is claimable right now
    assert len(queue.claim_due_reminders("w1")) == 0
    
    # Advance time to snooze due
    clock.advance(timedelta(minutes=11))
    
    claimed = queue.claim_due_reminders("w2")
    assert len(claimed) == 1
    assert claimed[0]['id'] == r_snooze

def test_clock_jumps_and_missed_policy(temp_db):
    clock = MockClock(datetime(2026, 9, 18, 12, 0, 0))
    queue = ReminderQueue(temp_db, clock=clock)
    
    # Create one with drop policy and one with deliver policy
    queue.create(due_time=clock() + timedelta(minutes=5), missed_policy='skip')
    r_deliver = queue.create(due_time=clock() + timedelta(minutes=5), missed_policy='deliver')
    
    # Device powers off, jumps forward 2 hours
    clock.advance(timedelta(hours=2))
    
    # Run process_missed_reminders (cutoff is default 60 mins)
    queue.process_missed_reminders(cutoff_minutes=60)
    
    claimed = queue.claim_due_reminders("w1")
    assert len(claimed) == 1
    assert claimed[0]['id'] == r_deliver

def test_output_failure_and_retries(temp_db):
    clock = MockClock(datetime(2026, 9, 18, 12, 0, 0))
    queue = ReminderQueue(temp_db, clock=clock)
    
    r1 = queue.create(due_time=clock())
    
    # Fail 3 times
    for _ in range(3):
        claimed = queue.claim_due_reminders("w1")
        assert len(claimed) == 1
        queue.complete_delivery(claimed[0]['id'], claimed[0]['delivery_token'], success=False)
        
    # On 4th attempt, it should be marked 'failed' because max_retries is 3
    # Wait, the failure in complete_delivery already marked it 'failed' on the 3rd fail.
    # Actually attempt_count goes 1 -> 2 -> 3 (failed), let's check
    with get_db_conn(temp_db) as conn:
        c = conn.cursor()
        c.execute("SELECT attempt_count, status FROM reminders WHERE id=?", (r1,))
        row = c.fetchone()
        assert row[0] == 3
        assert row[1] == 'failed'
        
    assert len(queue.claim_due_reminders("w1")) == 0

def test_scheduler_callback_exception_retry(temp_db):
    clock = MockClock(datetime(2026, 9, 18, 12, 0, 0))
    queue = ReminderQueue(temp_db, clock=clock)
    
    from scheduler.scheduler import ReminderScheduler
    scheduler = ReminderScheduler(temp_db, poll_interval=0.1)
    scheduler.queue = queue # inject mock clock queue
    
    r1 = queue.create(due_time=clock())
    
    # Callback that raises an exception
    attempts = 0
    def failing_callback(rem_id, action_id, due_time, delivery_token):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("Intentional failure")
        
    scheduler.register_delivery_callback(failing_callback)
    
    # Run the scheduler loop once manually instead of starting the thread
    # to control the flow and avoid race conditions
    claims = queue.claim_due_reminders(scheduler.worker_id, lease_seconds=60, max_claims=10)
    for claim in claims:
        success = True
        for cb in scheduler.callbacks:
            try:
                cb(claim['id'], claim['action_id'], claim['due_time'], claim['delivery_token'])
            except Exception:
                success = False
        queue.complete_delivery(claim['id'], claim['delivery_token'], success)
        
    assert attempts == 1
    
    with get_db_conn(temp_db) as conn:
        c = conn.cursor()
        c.execute("SELECT status, attempt_count FROM reminders WHERE id=?", (r1,))
        status, attempt_count = c.fetchone()
        assert status == "pending"
        assert attempt_count == 1
        
    # simulate time passing
    clock.advance(timedelta(minutes=6))
    
    # Run 2 more times to reach max retries
    for _ in range(2):
        claims = queue.claim_due_reminders(scheduler.worker_id, lease_seconds=60, max_claims=10)
        for claim in claims:
            success = True
            for cb in scheduler.callbacks:
                try:
                    cb(claim['id'], claim['action_id'], claim['due_time'], claim['delivery_token'])
                except Exception:
                    success = False
            queue.complete_delivery(claim['id'], claim['delivery_token'], success)
        clock.advance(timedelta(minutes=6))
        
    assert attempts == 3
    
    with get_db_conn(temp_db) as conn:
        c = conn.cursor()
        c.execute("SELECT status, attempt_count FROM reminders WHERE id=?", (r1,))
        status, attempt_count = c.fetchone()
        assert status == "failed"
        assert attempt_count == 3

def test_clean_sqlite_teardown(temp_db):
    clock = MockClock(datetime(2026, 9, 18, 12, 0, 0))
    queue = ReminderQueue(temp_db, clock=clock)
    r = queue.create(due_time=clock())
    
    # Open connection and do something but ensure context manager closes it
    with get_db_conn(temp_db) as conn:
        pass # The context manager will close it.
        
    # On Windows, you can't remove a file if there's an open handle.
    # Let's try removing it right now to verify.
    os.remove(temp_db)
    assert not os.path.exists(temp_db)
