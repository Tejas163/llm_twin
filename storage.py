import sqlite3
import json
import os

DB_FILE = ".llm_twin_storage.db"

def init_db():
    """Initializes the SQLite schema layer to persist multi-tenant cluster configurations."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Create a table designed to store complete historical topology metadata
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scenarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            project_tag TEXT DEFAULT 'General',
            model_name TEXT NOT NULL,
            param_billion REAL NOT NULL,
            config_json TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_scenario(name, project_tag, model_name, param_billion, config_dict, metrics_dict):
    """Saves or overwrites a snapshot architecture mapping into the local vault."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO scenarios 
            (name, project_tag, model_name, param_billion, config_json, metrics_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            name, 
            project_tag, 
            model_name, 
            param_billion, 
            json.dumps(config_dict), 
            json.dumps(metrics_dict)
        ))
        conn.commit()
        return True, "Blueprint successfully locked to the local vault!"
    except Exception as e:
        return False, f"Database Write Error: {str(e)}"
    finally:
        conn.close()

def list_scenarios():
    """Queries and extracts summary metadata tables for all saved records."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, name, project_tag, model_name, param_billion, created_at 
        FROM scenarios 
        ORDER BY created_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    
    # Parse rows into scannable dictionary formats
    return [
        {
            "id": r[0], "name": r[1], "project_tag": r[2], 
            "model_name": r[3], "param_billion": r[4], "created_at": r[5]
        }
        for r in rows
    ]

def load_scenario_by_name(name):
    """Retrieves deep configuration configurations for an explicit record index key."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT config_json, metrics_json FROM scenarios WHERE name = ?", (name,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return json.loads(row[0]), json.loads(row[1])
    return None, None