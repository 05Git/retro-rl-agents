CREATE TABLE IF NOT EXISTS imitation_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    model_type          TEXT    NOT NULL,
    model_settings      TEXT    NOT NULL,  -- store as JSON string
    network_layers      TEXT,              -- store as JSON string
    model_path          TEXT,
    save_path           TEXT    NOT NULL,
    env                 TEXT    NOT NULL,
    env_settings        TEXT,              -- store as JSON string
    tb_path             TEXT,
    imitation_type      TEXT    NOT NULL,
    transitions_path    TEXT    NOT NULL,
    imitation_settings  TEXT,              -- store as JSON string
    avg_loss_final      REAL,
    std_loss_final      REAL,
    avg_ep_len_final    REAL,
    std_ep_len_final    REAL,
    started_at          TEXT    DEFAULT (datetime('now')),
    finished_at         TEXT,
    sys_settings        TEXT               -- store as JSON string
);