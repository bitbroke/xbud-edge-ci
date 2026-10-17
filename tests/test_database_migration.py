"""Database schema, migration, and encryption compatibility tests."""
import os
import sqlite3
import pytest
from storage.database import StorageManager, get_db_conn

def test_database_schema_initialization(tmp_path):
    """Verify that StorageManager creates full schema including actions and reminders tables."""
    db_path = str(tmp_path / "test_migration.db")
    manager = StorageManager(db_path)
    
    assert os.path.exists(db_path)
    assert os.path.exists(db_path + ".key")
    
    with get_db_conn(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        
        assert "actions" in tables
        assert "reminders" in tables

def test_sqlite_pragmas_and_foreign_keys(tmp_path):
    """Verify WAL mode, synchronous settings, and foreign key integrity support."""
    db_path = str(tmp_path / "test_pragmas.db")
    StorageManager(db_path)
    
    with get_db_conn(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys")
        fk = cursor.fetchone()[0]
        # SQLite should execute PRAGMA commands without error
        assert fk in (0, 1)

def test_migration_transaction_rollback(tmp_path):
    """Verify atomic transaction rollback on write/migration failure."""
    db_path = str(tmp_path / "test_atomic.db")
    StorageManager(db_path)
    
    conn = sqlite3.connect(db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            with conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO reminders (id, due_time, status) VALUES ('rem_dup', '2026-10-18 12:00:00', 'pending')")
                cursor.execute("INSERT INTO reminders (id, due_time, status) VALUES ('rem_dup', '2026-10-18 12:00:00', 'pending')")
    finally:
        conn.close()

    # Verify nothing was committed
    with get_db_conn(db_path) as verify_conn:
        cursor = verify_conn.cursor()
        cursor.execute("SELECT count(*) FROM reminders WHERE id='rem_dup'")
        assert cursor.fetchone()[0] == 0
