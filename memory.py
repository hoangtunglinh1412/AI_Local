import sqlite3
from datetime import datetime


DB_NAME = "chat_history.db"


def create_database():
    connection = sqlite3.connect(DB_NAME)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            role TEXT,
            content TEXT,
            timestamp TEXT
        )
    """)

    connection.commit()
    connection.close()


def save_message(conversation_id, role, content):

    connection = sqlite3.connect(DB_NAME)

    cursor = connection.cursor()

    timestamp = datetime.now().isoformat()

    cursor.execute("""
        INSERT INTO messages
        (conversation_id, role, content, timestamp)
        VALUES (?, ?, ?, ?)
    """, (
        conversation_id,
        role,
        content,
        timestamp
    ))

    connection.commit()
    connection.close()


def load_messages(conversation_id):

    connection = sqlite3.connect(DB_NAME)

    cursor = connection.cursor()

    cursor.execute("""
        SELECT role, content
        FROM messages
        WHERE conversation_id = ?
        ORDER BY id
    """, (conversation_id,))

    rows = cursor.fetchall()

    connection.close()

    messages = []

    for role, content in rows:

        messages.append({
            "role": role,
            "content": content
        })

    return messages

def save_memory(content):

    connection = sqlite3.connect(DB_NAME)

    cursor = connection.cursor()

    timestamp = datetime.now().isoformat()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT,
            timestamp TEXT
        )
    """)

    cursor.execute("""
        INSERT INTO memories
        (content, timestamp)
        VALUES (?, ?)
    """, (
        content,
        timestamp
    ))

    connection.commit()
    connection.close()


def load_memories():

    connection = sqlite3.connect(DB_NAME)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT,
            timestamp TEXT
        )
    """)

    cursor.execute("""
        SELECT id, content
        FROM memories
        ORDER BY id
    """)

    rows = cursor.fetchall()

    connection.close()

    return rows


def delete_memory(memory_id):

    connection = sqlite3.connect(DB_NAME)

    cursor = connection.cursor()

    cursor.execute("""
        DELETE FROM memories
        WHERE id = ?
    """, (memory_id,))

    connection.commit()
    connection.close()

def update_memory(memory_id, content):

    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()

    cursor.execute("""
        UPDATE memories
        SET content = ?
        WHERE id = ?
    """, (content, memory_id))

    connection.commit()
    connection.close()

def save_summary(conversation_id, summary):

    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
            conversation_id TEXT PRIMARY KEY,
            summary TEXT,
            timestamp TEXT
        )
    """)

    timestamp = datetime.now().isoformat()

    cursor.execute("""
        INSERT OR REPLACE INTO summaries
        (conversation_id, summary, timestamp)
        VALUES (?, ?, ?)
    """, (
        conversation_id,
        summary,
        timestamp
    ))

    connection.commit()
    connection.close()

def load_summary(conversation_id):

    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
            conversation_id TEXT PRIMARY KEY,
            summary TEXT,
            timestamp TEXT
        )
    """)

    cursor.execute("""
        SELECT summary
        FROM summaries
        WHERE conversation_id = ?
    """, (conversation_id,))

    row = cursor.fetchone()

    connection.close()

    if row is None:
        return ""

    return row[0]

def load_recent_messages(conversation_id, limit=10):

    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT role, content
        FROM messages
        WHERE conversation_id = ?
        ORDER BY id DESC
        LIMIT ?
    """, (conversation_id, limit))

    rows = cursor.fetchall()

    connection.close()

    rows.reverse()

    messages = []

    for role, content in rows:

        messages.append({
            "role": role,
            "content": content
        })

    return messages

def count_messages(conversation_id):

    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT COUNT(*)
        FROM messages
        WHERE conversation_id = ?
    """, (conversation_id,))

    result = cursor.fetchone()[0]

    connection.close()

    return result

def save_summary_checkpoint(conversation_id, message_id):

    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS summary_checkpoints (
            conversation_id TEXT PRIMARY KEY,
            message_id INTEGER
        )
    """)

    cursor.execute("""
        INSERT OR REPLACE INTO summary_checkpoints
        (conversation_id, message_id)
        VALUES (?, ?)
    """, (
        conversation_id,
        message_id
    ))

    connection.commit()
    connection.close()

def load_summary_checkpoint(conversation_id):

    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS summary_checkpoints (
            conversation_id TEXT PRIMARY KEY,
            message_id INTEGER
        )
    """)

    cursor.execute("""
        SELECT message_id
        FROM summary_checkpoints
        WHERE conversation_id = ?
    """, (conversation_id,))

    row = cursor.fetchone()

    connection.close()

    if row is None:
        return 0

    return row[0]

def load_messages_after(conversation_id, message_id):

    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT id, role, content
        FROM messages
        WHERE conversation_id = ?
        AND id > ?
        ORDER BY id
    """, (
        conversation_id,
        message_id
    ))

    rows = cursor.fetchall()

    connection.close()

    messages = []

    for message_id, role, content in rows:

        messages.append({
            "id": message_id,
            "role": role,
            "content": content
        })

    return messages