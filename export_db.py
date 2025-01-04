import sqlite3
import json
import os

def export_sqlite_data(sqlite_db_path):
    conn = sqlite3.connect(sqlite_db_path)
    cursor = conn.cursor()

    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    data = {}
    for table in tables:
        table_name = table[0]
        cursor.execute(f"SELECT * FROM {table_name}")
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        data[table_name] = {
            'columns': columns,
            'rows': [list(row) for row in rows]  # Convert rows to lists for JSON serialization
        }

    conn.close()

    with open('sqlite_data.json', 'w') as f:
        json.dump(data, f, indent=2)

# Adjust this path to your SQLite database file
sqlite_db_path = 'instance/fact_checker.db'  # or wherever your SQLite file is located
export_sqlite_data(sqlite_db_path)
print("Data exported to sqlite_data.json")