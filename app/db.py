import sqlite3
from pathlib import Path

#Railway volume mount point
DB_PATH = Path('/app/data') / 'app.db'

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    # volume directory 
    Path('/app/data').mkdir(parents=True, exist_ok=True)
    
    conn = get_db()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            coin_id TEXT NOT NULL,
            interval TEXT NOT NULL,
            horizon INTEGER NOT NULL,
            predicted_price REAL NOT NULL,
            current_price REAL NOT NULL,
            horizon_label TEXT,
            strategy TEXT,
            validation_mae_pct REAL,
            baseline_validation_mae_pct REAL,
            target_time_utc TEXT,
            created_at_utc TEXT DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'pending',
            actual_price REAL,
            absolute_error REAL,
            percentage_accuracy REAL,
            direction_correct INTEGER,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()