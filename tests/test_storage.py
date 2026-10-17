import os
import sqlite3
import pytest
from cryptography.fernet import Fernet
from storage.database import StorageManager, InvalidKeyError, ActionAlreadyConfirmedError

@pytest.fixture
def temp_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    yield db_path
    if os.path.exists(db_path):
        os.remove(db_path)
    key_path = db_path + '.key'
    if os.path.exists(key_path):
        os.remove(key_path)

from contextlib import closing

def test_initialization_creates_key_and_schema(temp_db):
    manager = StorageManager(temp_db)
    assert os.path.exists(temp_db)
    assert os.path.exists(temp_db + '.key')
    
    with closing(sqlite3.connect(temp_db)) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='actions'")
        assert cursor.fetchone()[0] == 1

def test_wrong_key_fails_explicitly(temp_db):
    # Initialize with a random key
    manager = StorageManager(temp_db)
    
    # Try to initialize again with a wrong key
    wrong_key = Fernet.generate_key()
    with pytest.raises(InvalidKeyError):
        StorageManager(temp_db, key=wrong_key)

def test_create_and_get_action(temp_db):
    manager = StorageManager(temp_db)
    manager.create_action('act_1', 'create_reminder', {'time': '09:00'})
    
    action = manager.get_action('act_1')
    assert action is not None
    assert action['intent'] == 'create_reminder'
    assert action['state'] == 'pending'
    assert action['data']['time'] == '09:00'

def test_concurrent_confirmations_enforce_one_effect(temp_db):
    manager = StorageManager(temp_db)
    manager.create_action('act_2', 'create_reminder', {'time': '10:00'})
    
    effect_calls = []
    
    def effect_callback(cursor):
        effect_calls.append(1)
        cursor.execute("INSERT INTO reminders (id, action_id, due_time, status) VALUES (?, ?, ?, ?)", 
                       ('rem_1', 'act_2', '2026-09-19 10:00:00', 'pending'))

    # First confirmation should succeed
    manager.confirm_action('act_2', effect_callback)
    assert len(effect_calls) == 1
    
    action = manager.get_action('act_2')
    assert action['state'] == 'confirmed'
    
    # Second confirmation should fail explicitly and not trigger effect
    with pytest.raises(ActionAlreadyConfirmedError):
        manager.confirm_action('act_2', effect_callback)
        
    assert len(effect_calls) == 1

def test_idempotency_key(temp_db):
    manager = StorageManager(temp_db)
    manager.create_action('act_3', 'intent_a', {'a': 1}, idempotency_key='idem_1')
    
    # Should not crash and not create a duplicate
    manager.create_action('act_4', 'intent_b', {'b': 2}, idempotency_key='idem_1')
    
    action = manager.get_action('act_4')
    assert action is None
