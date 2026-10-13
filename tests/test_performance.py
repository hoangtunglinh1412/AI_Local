"""
Bộ kiểm thử hiệu năng, độ trễ và các trường hợp biên (Performance & Edge Cases).
Bao gồm:
1. SQLite WAL mode và Concurrency (Đa luồng đồng thời ghi/đọc).
2. Lazy Loading Embedding model & ChromaDB Client.
3. Fast Heuristic RAG Router (0 LLM call).
4. Error Handling khi Ollama gặp sự cố kết nối hoặc timeout.
5. Background Memory Worker, hàng đợi non-blocking và xử lý định dạng.
6. Bảo vệ biên (Input quá dài, xóa nhầm prefix tài liệu).
"""

import os
import sqlite3
import threading
import time
from unittest import mock

import numpy as np
import pytest

import memory
import rag
import chatbot


# ========================================================
# 1. SQLITE CONCURRENCY & WAL MODE
# ========================================================

def test_sqlite_wal_mode_enabled(temp_db):
    """Kiểm tra chế độ WAL được kích hoạt trên file database."""
    with memory.get_connection(temp_db) as conn:
        cursor = conn.cursor()
        mode = cursor.execute("PRAGMA journal_mode;").fetchone()[0]
        assert mode.lower() == "wal"


def test_sqlite_connection_always_closed_on_exception(temp_db):
    """Kiểm tra connection luôn được đóng an toàn kể cả khi có Exception."""
    conn_ref = None
    try:
        with memory.get_connection(temp_db) as conn:
            conn_ref = conn
            raise RuntimeError("Cố tình ném lỗi trong transaction")
    except RuntimeError:
        pass

    assert conn_ref is not None
    # Khi đã đóng, execute sẽ ném ProgrammingError: Cannot operate on a closed database
    with pytest.raises(sqlite3.ProgrammingError):
        conn_ref.execute("SELECT 1;")


def test_sqlite_multithreaded_concurrency(temp_db):
    """Kiểm thử 8 luồng đồng thời ghi/đọc SQLite mà không gặp lock lỗi."""
    num_threads = 8
    messages_per_thread = 15
    errors = []

    def worker(thread_idx):
        try:
            for i in range(messages_per_thread):
                conv = f"thread_{thread_idx}"
                memory.save_message(conv, "user", f"msg_{i}")
                mems = memory.load_messages(conv)
                assert len(mems) >= 1
                memory.save_memory(f"memory from thread {thread_idx} #{i}")
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=worker, args=(t,))
        for t in range(num_threads)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert errors == []

    total_mems = memory.load_memories()
    assert len(total_mems) == num_threads * messages_per_thread


# ========================================================
# 2. LAZY LOADING EMBEDDING & CHROMADB
# ========================================================

def test_embedding_device_defaults_to_cpu(monkeypatch):
    """Đảm bảo embedding model mặc định khởi tạo trên CPU để tiết kiệm VRAM."""
    monkeypatch.delenv("EMBEDDING_DEVICE", raising=False)
    monkeypatch.setattr(rag, "_embedding_model", None)

    fake_st_cls = mock.MagicMock()
    with mock.patch("sentence_transformers.SentenceTransformer", fake_st_cls):
        rag.get_embedding_model()
        fake_st_cls.assert_called_once_with(
            rag.EMBEDDING_MODEL,
            device="cpu"
        )


def test_search_documents_returns_empty_when_no_documents(fake_collection):
    """ChromaDB trống hoặc query rỗng thì không gọi embedding model tốn tài nguyên."""
    fake_collection.count.return_value = 0

    fake_model = mock.MagicMock()
    with mock.patch.object(rag, "get_embedding_model", return_value=fake_model):
        res = rag.search_documents("")
        assert res["documents"] == [[]]
        fake_model.encode.assert_not_called()

        res2 = rag.search_documents("Mean là gì?")
        assert res2["documents"] == [[]]
        fake_model.encode.assert_not_called()


def test_delete_document_exact_match_does_not_delete_prefix_backup(fake_collection):
    """
    Xóa 'doc.txt' chỉ xóa vector thuộc về 'doc.txt',
    không xóa vector của file 'doc.txt_backup.txt'.
    """
    fake_collection.get.return_value = {"ids": ["doc.txt_0", "doc.txt_1"]}

    deleted = rag.delete_document_from_database("doc.txt")

    assert deleted == 2
    fake_collection.get.assert_called_once_with(where={"filename": "doc.txt"})
    fake_collection.delete.assert_called_once_with(ids=["doc.txt_0", "doc.txt_1"])


# ========================================================
# 3. FAST HEURISTIC RAG ROUTER (0 LLM CALL)
# ========================================================

@pytest.mark.parametrize("query", [
    "Xin chào",
    "chào bạn",
    "hello",
    "Hi!",
    "Cảm ơn bạn nhé",
    "tạm biệt",
    "Bạn khỏe không?",
    "viết cho tôi một bài thơ",
    "kể một câu chuyện"
])
def test_heuristic_router_classifies_greetings_as_false(query):
    """Các câu chào hỏi, xã giao, sáng tạo được router heuristic phân loại là False (0 LLM call)."""
    assert chatbot.classify_query_heuristic(query) is False


@pytest.mark.parametrize("query", [
    "Mean là gì?",
    "Định nghĩa Outlier",
    "Giải thích Supervised Learning",
    "Công thức tính phương sai",
    "What is Multilayer Perceptron?",
    "Đặc điểm của neural network",
    "Cho tôi xem tài liệu thống kê"
])
def test_heuristic_router_classifies_definitions_as_true(query):
    """Các câu hỏi khái niệm, định nghĩa, tài liệu được phân loại là True (0 LLM call)."""
    assert chatbot.classify_query_heuristic(query) is True


def test_heuristic_router_returns_none_for_ambiguous_queries():
    """Câu hỏi không rõ ràng trả về None để chuyển sang LLM router."""
    assert chatbot.classify_query_heuristic("Hôm nay tôi muốn xem lại việc đó") is None
    assert chatbot.classify_query_heuristic("Tôi đồng ý với quan điểm này") is None


# ========================================================
# 4. ERROR HANDLING & FAULT TOLERANCE
# ========================================================

def test_stream_chat_response_handles_connection_error(chatbot_module, capsys):
    """Khi Ollama ngắt kết nối đột ngột, hiển thị thông báo lỗi thân thiện thay vì crash."""
    with mock.patch("ollama.chat", side_effect=ConnectionError("Connection refused by Ollama")):
        answer = chatbot_module.stream_chat_response([{"role": "user", "content": "Hi"}])
        assert answer == ""

    out = capsys.readouterr().out
    assert "[AI Error]" in out
    assert "Không thể kết nối tới mô hình Ollama" in out


def test_answer_question_does_not_save_when_ollama_fails(chatbot_module):
    """Khi Ollama lỗi, tin nhắn không bị lưu vào lịch sử SQLite."""
    with mock.patch.object(chatbot_module, "should_use_rag", return_value=False), \
         mock.patch.object(chatbot_module, "stream_chat_response", return_value=""):
        answer = chatbot_module.answer_question("Câu hỏi thất bại")
        assert answer == ""

    # Tin nhắn không được lưu vào SQLite
    saved = memory.load_messages(chatbot_module.CONVERSATION_ID)
    assert len(saved) == 0


def test_check_memory_returns_none_on_ollama_exception():
    """Hàm check_memory trả về 'NONE' khi Ollama lỗi."""
    with mock.patch("ollama.chat", side_effect=RuntimeError("Ollama Timeout")):
        res = chatbot.check_memory("Tôi thích học Python")
        assert res == "NONE"


def test_manage_memory_returns_ignore_on_ollama_exception():
    """Hàm manage_memory trả về 'IGNORE' khi Ollama lỗi."""
    with mock.patch("ollama.chat", side_effect=RuntimeError("Ollama Error")):
        res = chatbot.manage_memory("Python", [])
        assert res == "IGNORE"


# ========================================================
# 5. BACKGROUND MEMORY PROCESSING & PARSING
# ========================================================

@pytest.mark.parametrize("raw_llm_output, expected_action, expected_id", [
    ("NEW", "NEW", None),
    ("NEW - Thêm sở thích mới", "NEW", None),
    ("<think>Tôi đang suy nghĩ</think>\nNEW", "NEW", None),
    ("IGNORE", "IGNORE", None),
    ("UPDATE: ID: 7\nCONTENT: Người dùng thích Python và PyTorch", "UPDATE", 7),
    ("UPDATE ID: 12\nCONTENT: Cập nhật mới", "UPDATE", 12),
    ("<think>Cần update ID 3</think>\nUPDATE: ID: 3\nCONTENT: Dữ liệu mới", "UPDATE", 3)
])
def test_parse_memory_decision_robustness(raw_llm_output, expected_action, expected_id):
    """Trích xuất quyết định bộ nhớ chính xác kể cả khi có thẻ <think> hoặc format lỏng lẻo."""
    action, target_id, content = chatbot.parse_memory_decision(raw_llm_output)
    assert action == expected_action
    assert target_id == expected_id


def test_background_memory_fallback_to_new_when_id_not_found(temp_db):
    """Khi LLM yêu cầu UPDATE ID không tồn tại, tự động chuyển sang lưu NEW để tránh mất thông tin."""
    with mock.patch.object(chatbot, "check_memory", return_value="SAVE: Người dùng làm AI Engineer"), \
         mock.patch.object(chatbot, "manage_memory", return_value="UPDATE: ID: 999\nCONTENT: Người dùng làm AI Engineer"):

        chatbot._process_memory_task("Tôi làm AI Engineer")

    mems = memory.load_memories()
    assert len(mems) == 1
    assert "Người dùng làm AI Engineer" in mems[0][1]


def test_enqueue_memory_is_non_blocking():
    """Kiểm tra enqueue_memory_check chạy tức thì không làm trễ câu trả lời."""
    start_time = time.time()
    for i in range(10):
        chatbot.enqueue_memory_check(f"Question #{i}")
    elapsed = time.time() - start_time
    assert elapsed < 0.05  # Nhét 10 task vào queue chỉ tốn < 50ms


# ========================================================
# 6. EDGE CASES & GUARDRAILS
# ========================================================

def test_long_input_rejected_with_friendly_message(chatbot_module, capsys):
    """Input vượt quá MAX_INPUT_CHARS (16.000 ký tự) bị từ chối kèm thông báo lỗi."""
    huge_input = "A" * (chatbot.MAX_INPUT_CHARS + 50)
    answer = chatbot_module.answer_question(huge_input)
    assert answer == ""

    out = capsys.readouterr().out
    assert "[Lỗi]: Câu hỏi quá dài" in out
    assert "Giới hạn tối đa là 16000 ký tự" in out


def test_index_document_empty_content_returns_zero(tmp_path, monkeypatch, fake_collection):
    """File rỗng hoặc không có chunk thì index_document trả về 0 và không gọi upsert."""
    monkeypatch.setattr(rag, "DOCUMENT_FOLDER", str(tmp_path))
    (tmp_path / "empty.txt").write_text("   \n\n  ", encoding="utf-8")

    count = rag.index_document("empty.txt")
    assert count == 0
    fake_collection.upsert.assert_not_called()
