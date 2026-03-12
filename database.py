import sqlite3

def create_connection():
    conn = sqlite3.connect("detections.db", check_same_thread=False)
    return conn

def create_table(conn):

    query = """
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        animal TEXT,
        confidence REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """

    conn.execute(query)
    conn.commit()

def insert_detection(conn, animal, confidence):

    query = """
    INSERT INTO detections (animal, confidence)
    VALUES (?, ?)
    """

    conn.execute(query, (animal, confidence))
    conn.commit()

def get_all_detections(conn):

    cursor = conn.cursor()

    cursor.execute("SELECT animal, confidence, timestamp FROM detections")

    return cursor.fetchall()