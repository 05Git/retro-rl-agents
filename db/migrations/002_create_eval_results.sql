CREATE TABLE IF NOT EXISTS eval_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    model_type          TEXT    NOT NULL,
    model_settings      TEXT    NOT NULL,  -- store as JSON string
    network_layers      TEXT,              -- store as JSON string
    model_path          TEXT    NOT NULL,
    env                 TEXT    NOT NULL,
    env_settings        TEXT,              -- store as JSON string
    avg_return          REAL,
    std_return          REAL,
    avg_ep_len          REAL,
    std_ep_len          REAL,
    full_results        TEXT,              -- store as JSON string
    started_at          TEXT,   DEFAULT (datetime('now')),
    finished_at         TEXT,
    sys_settings        TEXT               -- store as JSON string
);