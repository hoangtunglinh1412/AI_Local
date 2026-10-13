import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional

DB_NAME = "chat_history.db"

_initialized_dbs = set()


@contextmanager
def get_connection(db_name: Optional[str] = None):
    """
    Context manager mở kết nối SQLite với cấu hình tối ưu:
    - Kích hoạt chế độ WAL (PRAGMA journal_mode=WAL) để tăng tốc độ ghi/đọc đồng thời
    - Kích hoạt PRAGMA synchronous=NORMAL để tối ưu hóa I/O an toàn
    - Đặt timeout 10.0s và busy_timeout 5000ms nhằm ngăn ngừa deadlock khi đa luồng
    - Tự động commit khi hoàn thành, rollback khi xảy ra exception
    - Đảm bảo luôn đóng kết nối (close) trong block finally
    """
    target_db = db_name or DB_NAME
    conn = sqlite3.connect(target_db, timeout=10.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def create_database(db_name: Optional[str] = None) -> None:
    """
    Tạo đầy đủ 4 bảng (messages, memories, summaries, summary_checkpoints)
    và các index cần thiết nếu chưa tồn tại.
    """
    target_db = db_name or DB_NAME
    with get_connection(target_db) as conn:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_conv_id
            ON messages (conversation_id, id)
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT,
                timestamp TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                conversation_id TEXT PRIMARY KEY,
                summary TEXT,
                timestamp TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS summary_checkpoints (
                conversation_id TEXT PRIMARY KEY,
                message_id INTEGER
            )
        """)

    _initialized_dbs.add(target_db)


def _ensure_initialized(db_name: Optional[str] = None) -> None:
    target_db = db_name or DB_NAME
    if target_db not in _initialized_dbs:
        create_database(target_db)


def save_message(conversation_id: str, role: str, content: str) -> None:
    """Lưu một tin nhắn mới vào bảng messages."""
    _ensure_initialized()
    with get_connection() as conn:
        cursor = conn.cursor()
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


def load_messages(conversation_id: str) -> List[Dict[str, str]]:
    """Tải toàn bộ tin nhắn của một cuộc hội thoại theo thứ tự thời gian."""
    _ensure_initialized()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT role, content
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id
        """, (conversation_id,))
        rows = cursor.fetchall()

    return [{"role": role, "content": content} for role, content in rows]


def save_memory(content: str) -> int:
    """Lưu một mẩu ghi nhớ dài hạn vào bảng memories và trả về id vừa tạo."""
    _ensure_initialized()
    with get_connection() as conn:
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        cursor.execute("""
            INSERT INTO memories
            (content, timestamp)
            VALUES (?, ?)
        """, (
            content,
            timestamp
        ))
        return cursor.lastrowid


def load_memories() -> List[Tuple[int, str]]:
    """Tải toàn bộ ghi nhớ dài hạn (id, content)."""
    _ensure_initialized()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, content
            FROM memories
            ORDER BY id
        """)
        return cursor.fetchall()


def delete_memory(memory_id: int) -> int:
    """Xóa ghi nhớ theo id. Trả về số dòng đã xóa."""
    _ensure_initialized()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM memories
            WHERE id = ?
        """, (memory_id,))
        return cursor.rowcount


def update_memory(memory_id: int, content: str) -> int:
    """Cập nhật nội dung ghi nhớ theo id. Trả về số dòng đã cập nhật."""
    _ensure_initialized()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE memories
            SET content = ?
            WHERE id = ?
        """, (content, memory_id))
        return cursor.rowcount


def save_summary(conversation_id: str, summary: str) -> None:
    """Lưu hoặc cập nhật tóm tắt cuộc hội thoại."""
    _ensure_initialized()
    with get_connection() as conn:
        cursor = conn.cursor()
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


def load_summary(conversation_id: str) -> str:
    """Tải tóm tắt cuộc hội thoại. Trả về chuỗi rỗng nếu chưa có."""
    _ensure_initialized()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT summary
            FROM summaries
            WHERE conversation_id = ?
        """, (conversation_id,))
        row = cursor.fetchone()

    if row is None:
        return ""
    return row[0]


def load_recent_messages(conversation_id: str, limit: int = 10) -> List[Dict[str, str]]:
    """Tải N tin nhắn gần nhất theo thứ tự thời gian tăng dần."""
    _ensure_initialized()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT role, content
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (conversation_id, limit))
        rows = cursor.fetchall()

    rows.reverse()
    return [{"role": role, "content": content} for role, content in rows]


def count_messages(conversation_id: str) -> int:
    """Đếm tổng số tin nhắn trong cuộc hội thoại."""
    _ensure_initialized()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*)
            FROM messages
            WHERE conversation_id = ?
        """, (conversation_id,))
        result = cursor.fetchone()[0]

    return result


def save_summary_checkpoint(conversation_id: str, message_id: int) -> None:
    """Lưu checkpoint id tin nhắn cuối cùng đã được summarize."""
    _ensure_initialized()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO summary_checkpoints
            (conversation_id, message_id)
            VALUES (?, ?)
        """, (
            conversation_id,
            message_id
        ))


def load_summary_checkpoint(conversation_id: str) -> int:
    """Tải checkpoint id tin nhắn cuối cùng đã summarize (mặc định 0)."""
    _ensure_initialized()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT message_id
            FROM summary_checkpoints
            WHERE conversation_id = ?
        """, (conversation_id,))
        row = cursor.fetchone()

    if row is None:
        return 0
    return row[0]


def load_messages_after(conversation_id: str, message_id: int) -> List[Dict[str, Any]]:
    """Tải các tin nhắn có id > message_id theo thứ tự tăng dần."""
    _ensure_initialized()
    with get_connection() as conn:
        cursor = conn.cursor()
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

    return [{"id": mid, "role": role, "content": content} for mid, role, content in rows]