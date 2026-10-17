-- Initial database schema for earbud assistant
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS actions (
    id TEXT PRIMARY KEY,
    intent TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('pending', 'confirmed', 'cancelled', 'expired')),
    encrypted_data TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    idempotency_key TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS reminders (
    id TEXT PRIMARY KEY,
    action_id TEXT,
    due_time DATETIME NOT NULL,
    timezone TEXT,
    status TEXT NOT NULL CHECK (status IN ('pending', 'processing', 'delivered', 'acknowledged', 'missed', 'snoozed', 'cancelled', 'failed')),
    delivery_id TEXT,
    attempt_count INTEGER DEFAULT 0,
    worker_lease_expiry DATETIME,
    missed_policy TEXT DEFAULT 'deliver',
    FOREIGN KEY (action_id) REFERENCES actions(id)
);

CREATE TABLE IF NOT EXISTS notes (
    id TEXT PRIMARY KEY,
    encrypted_title TEXT,
    encrypted_body TEXT,
    revision INTEGER DEFAULT 1,
    source_ref TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS calendar_events (
    id TEXT PRIMARY KEY,
    encrypted_title TEXT,
    start_time DATETIME NOT NULL,
    end_time DATETIME NOT NULL,
    display_timezone TEXT,
    revision INTEGER DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
