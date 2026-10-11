"""
Unit test cho memory.py (SQLite).

Dùng database tạm (fixture temp_db) nên không ảnh hưởng chat_history.db.
"""

import sqlite3

import memory


CONVERSATION_ID = "test"


def _table_names(db_path):

    connection = sqlite3.connect(db_path)

    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()

    connection.close()

    return {row[0] for row in rows}


# =========================
# MESSAGES
# =========================

def test_create_database_creates_messages_table(temp_db):

    assert "messages" in _table_names(temp_db)


def test_create_database_is_idempotent(temp_db):

    memory.save_message(CONVERSATION_ID, "user", "xin chào")

    memory.create_database()

    assert memory.count_messages(CONVERSATION_ID) == 1


def test_save_and_load_messages_preserves_order_and_unicode(temp_db):

    memory.save_message(CONVERSATION_ID, "user", "Mean là gì?")
    memory.save_message(CONVERSATION_ID, "assistant", "Mean là trung bình cộng.")

    assert memory.load_messages(CONVERSATION_ID) == [
        {"role": "user", "content": "Mean là gì?"},
        {"role": "assistant", "content": "Mean là trung bình cộng."}
    ]


def test_messages_are_isolated_by_conversation(temp_db):

    memory.save_message("a", "user", "hello A")
    memory.save_message("b", "user", "hello B")

    assert memory.load_messages("a") == [
        {"role": "user", "content": "hello A"}
    ]

    assert memory.count_messages("a") == 1
    assert memory.count_messages("b") == 1
    assert memory.count_messages("missing") == 0


def test_load_recent_messages_returns_last_n_in_chronological_order(temp_db):

    for i in range(15):
        memory.save_message(CONVERSATION_ID, "user", f"msg {i}")

    recent = memory.load_recent_messages(CONVERSATION_ID, 10)

    assert len(recent) == 10

    assert [m["content"] for m in recent] == [
        f"msg {i}" for i in range(5, 15)
    ]


def test_load_recent_messages_with_fewer_messages_than_limit(temp_db):

    memory.save_message(CONVERSATION_ID, "user", "only one")

    assert memory.load_recent_messages(CONVERSATION_ID, 10) == [
        {"role": "user", "content": "only one"}
    ]


# =========================
# LONG-TERM MEMORY
# =========================

def test_load_memories_empty_by_default(temp_db):

    assert memory.load_memories() == []


def test_save_update_delete_memory(temp_db):

    memory.save_memory("Người dùng học Data Science.")
    memory.save_memory("Người dùng thích chụp ảnh.")

    memories = memory.load_memories()

    assert [content for _, content in memories] == [
        "Người dùng học Data Science.",
        "Người dùng thích chụp ảnh."
    ]

    first_id = memories[0][0]
    second_id = memories[1][0]

    memory.update_memory(first_id, "Người dùng học Data Science và AI.")

    assert memory.load_memories()[0] == (
        first_id,
        "Người dùng học Data Science và AI."
    )

    memory.delete_memory(second_id)

    assert memory.load_memories() == [
        (first_id, "Người dùng học Data Science và AI.")
    ]


# =========================
# SUMMARY
# =========================

def test_load_summary_returns_empty_string_when_missing(temp_db):

    assert memory.load_summary(CONVERSATION_ID) == ""


def test_save_summary_replaces_previous_summary(temp_db):

    memory.save_summary(CONVERSATION_ID, "Summary 1")
    memory.save_summary(CONVERSATION_ID, "Summary 2")

    assert memory.load_summary(CONVERSATION_ID) == "Summary 2"
    assert memory.load_summary("other") == ""


# =========================
# SUMMARY CHECKPOINT
# =========================

def test_summary_checkpoint_defaults_to_zero(temp_db):

    assert memory.load_summary_checkpoint(CONVERSATION_ID) == 0


def test_save_summary_checkpoint_replaces_previous_value(temp_db):

    memory.save_summary_checkpoint(CONVERSATION_ID, 5)
    memory.save_summary_checkpoint(CONVERSATION_ID, 12)

    assert memory.load_summary_checkpoint(CONVERSATION_ID) == 12


def test_load_messages_after_checkpoint(temp_db):

    for i in range(5):
        memory.save_message(CONVERSATION_ID, "user", f"msg {i}")

    all_after_zero = memory.load_messages_after(CONVERSATION_ID, 0)

    assert len(all_after_zero) == 5
    assert set(all_after_zero[0].keys()) == {"id", "role", "content"}

    checkpoint = all_after_zero[2]["id"]

    after = memory.load_messages_after(CONVERSATION_ID, checkpoint)

    assert [m["content"] for m in after] == ["msg 3", "msg 4"]
    assert all(m["id"] > checkpoint for m in after)


def test_summary_trigger_condition_uses_messages_since_checkpoint(temp_db):
    """(count_messages - checkpoint) >= 20 mới kích hoạt summarize."""

    threshold = 20

    for i in range(threshold - 1):
        memory.save_message(CONVERSATION_ID, "user", f"msg {i}")

    count = memory.count_messages(CONVERSATION_ID)
    checkpoint = memory.load_summary_checkpoint(CONVERSATION_ID)

    assert count - checkpoint < threshold

    memory.save_message(CONVERSATION_ID, "user", "msg 19")

    count = memory.count_messages(CONVERSATION_ID)

    assert count - checkpoint >= threshold

