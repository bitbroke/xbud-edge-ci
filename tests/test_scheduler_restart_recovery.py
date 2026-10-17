"""Software regression for a restart occurring before a crashed worker lease expires."""
from datetime import datetime, timedelta, timezone
import threading
import time
import pytest
from unittest.mock import Mock
from app.coordinator import EventCoordinator

from scheduler.reminder_queue import ReminderQueue
from scheduler.scheduler import ReminderScheduler
from storage.database import StorageManager, get_db_conn


def test_restart_recovers_lease_that_expires_after_startup(tmp_path):
    path = str(tmp_path/'reminders.db')
    StorageManager(path)
    instant = [datetime(2026, 9, 20, 12, tzinfo=timezone.utc)]
    queue = ReminderQueue(path, clock=lambda: instant[0])
    rid = queue.create(instant[0])
    old = queue.claim_due_reminders('crashed-worker', lease_seconds=60)[0]
    scheduler = ReminderScheduler(path, poll_interval=0.01)
    scheduler.queue = ReminderQueue(path, clock=lambda: instant[0])
    played = threading.Event()
    scheduler.register_delivery_callback(lambda *_: played.set())
    scheduler.start()
    try:
        assert not played.wait(0.05), 'An unexpired lease must not be reclaimed'
        instant[0] += timedelta(seconds=61)
        scheduler.interrupt()
        assert played.wait(2), 'Expired old lease must be recovered without another restart'
        deadline = time.monotonic()+2
        while time.monotonic()<deadline:
            with get_db_conn(path) as conn:
                status=conn.execute('SELECT status FROM reminders WHERE id=?',(rid,)).fetchone()[0]
            if status=='delivered':
                break
            time.sleep(0.01)
        assert status=='delivered'
        # Completion from the abandoned worker cannot alter a finished delivery.
        queue.complete_delivery(rid, old['delivery_token'], False)
        with get_db_conn(path) as conn:
            assert conn.execute('SELECT status FROM reminders WHERE id=?',(rid,)).fetchone()[0]=='delivered'
    finally:
        scheduler.stop()


def test_scheduler_without_delivery_callback_keeps_reminder_pending(tmp_path):
    path = str(tmp_path/'reminders.db')
    StorageManager(path)
    queue = ReminderQueue(path)
    rid = queue.create(datetime.now(timezone.utc)-timedelta(seconds=1))
    scheduler = ReminderScheduler(path, poll_interval=0.01)
    scheduler.start()
    try:
        time.sleep(0.08)
        with get_db_conn(path) as conn:
            assert conn.execute('SELECT status,attempt_count FROM reminders WHERE id=?',(rid,)).fetchone()==('pending',0)
    finally:
        scheduler.stop()


@pytest.mark.parametrize('kind', ['coordinator', 'scheduler'])
def test_shutdown_reports_active_worker_instead_of_silent_success(kind, tmp_path):
    component = EventCoordinator(Mock(), Mock()) if kind=='coordinator' else ReminderScheduler(str(tmp_path/'unused.db'))
    release = threading.Event()
    component.thread = threading.Thread(target=release.wait)
    component.thread.start()
    try:
        with pytest.raises(RuntimeError, match='shutdown timeout'):
            component.stop(timeout=0.01)
        assert component.thread.is_alive()
    finally:
        release.set()
        component.stop(timeout=1)
    assert not component.thread.is_alive()
