"""
Unit tests cho các hàm tiện ích của Streamlit Web UI (app.py).
Không cần GPU, ChromaDB hay Ollama thật (đã được mock trong conftest và unit test).
"""

import os
import io
from unittest import mock
import pytest

import memory
import app


class DummyUploadedFile:
    """Giả lập đối tượng UploadedFile của Streamlit."""

    def __init__(self, name, content_bytes):
        self.name = name
        self._buffer = io.BytesIO(content_bytes)

    def getbuffer(self):
        return self._buffer.getvalue()

    @property
    def size(self):
        return len(self._buffer.getvalue())


def test_save_uploaded_document(tmp_path):
    """Kiểm tra lưu file tải lên và gọi index_document()."""
    dummy_file = DummyUploadedFile("tailieu_mau.txt", b"Noi dung tai lieu mau")
    dest_dir = str(tmp_path / "documents")

    with mock.patch("app.index_document", return_value=3) as mock_index:
        chunks = app.save_uploaded_document(dummy_file, dest_folder=dest_dir)

        assert chunks == 3
        mock_index.assert_called_once_with("tailieu_mau.txt")

        saved_path = os.path.join(dest_dir, "tailieu_mau.txt")
        assert os.path.exists(saved_path)
        with open(saved_path, "rb") as f:
            assert f.read() == b"Noi dung tai lieu mau"


def test_save_uploaded_document_fallback_read(tmp_path):
    """Kiểm tra file-like object chỉ có hàm read()."""
    class SimpleFile:
        def __init__(self):
            self.name = "sample.txt"
        def read(self):
            return "Nội dung văn bản"

    dest_dir = str(tmp_path / "documents")
    with mock.patch("app.index_document", return_value=1) as mock_index:
        chunks = app.save_uploaded_document(SimpleFile(), dest_folder=dest_dir)
        assert chunks == 1
        mock_index.assert_called_once_with("sample.txt")


def test_get_rag_context_and_sources_filtering():
    """Kiểm tra lọc chunk theo distance_threshold và format dữ liệu."""
    fake_results = {
        "ids": [["doc1.txt_0", "doc1.txt_1"]],
        "documents": [[
            "Mean là giá trị trung bình cộng.",
            "Nội dung này quá xa và không liên quan."
        ]],
        "metadatas": [[
            {"filename": "doc1.txt", "chunk": 0, "section": "STATISTICS"},
            {"filename": "doc1.txt", "chunk": 1, "section": "OTHER"}
        ]],
        "distances": [[4.5, 25.0]]
    }

    with mock.patch("app.search_documents", return_value=fake_results):
        context, sources = app.get_rag_context_and_sources(
            "Mean là gì?",
            top_k=2,
            distance_threshold=18.0
        )

        # Chunk 1 (khoảng cách 4.5 <= 18.0) được giữ lại
        # Chunk 2 (khoảng cách 25.0 > 18.0) bị loại
        assert len(sources) == 1
        assert sources[0]["filename"] == "doc1.txt"
        assert sources[0]["chunk"] == 0
        assert sources[0]["section"] == "STATISTICS"
        assert sources[0]["distance"] == 4.5
        assert sources[0]["text"] == "Mean là giá trị trung bình cộng."

        assert "[1] doc1.txt | Section: STATISTICS" in context
        assert "Mean là giá trị trung bình cộng." in context


def test_get_rag_context_and_sources_empty():
    """Kiểm tra khi không có kết quả hoặc có ngoại lệ từ search_documents."""
    empty_results = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    with mock.patch("app.search_documents", return_value=empty_results):
        context, sources = app.get_rag_context_and_sources("Không có gì")
        assert context == ""
        assert sources == []

    with mock.patch("app.search_documents", side_effect=RuntimeError("DB Error")):
        context, sources = app.get_rag_context_and_sources("Lỗi")
        assert context == ""
        assert sources == []


def test_stream_ollama_response():
    """Kiểm tra generator streaming token từ Ollama."""
    mock_chunks = [
        {"message": {"role": "assistant", "content": "Xin "}},
        {"message": {"role": "assistant", "content": "chào "}},
        {"message": {"role": "assistant", "content": "bạn!"}},
        {"message": {"role": "assistant", "content": ""}},
    ]

    with mock.patch("app.ollama.chat", return_value=iter(mock_chunks)) as mock_chat:
        generator = app.stream_ollama_response([{"role": "user", "content": "Hi"}])
        tokens = list(generator)

        assert tokens == ["Xin ", "chào ", "bạn!"]
        mock_chat.assert_called_once()
        kwargs = mock_chat.call_args.kwargs
        assert kwargs.get("stream") is True


def test_render_sources_expander():
    """Kiểm tra render_sources_expander gọi Streamlit components đúng cách."""
    sources = [
        {
            "filename": "guide.pdf",
            "chunk": 2,
            "section": "INTRO",
            "distance": 3.1415,
            "text": "Trích đoạn hướng dẫn."
        }
    ]

    with mock.patch("app.st.expander") as mock_expander, \
         mock.patch("app.st.markdown") as mock_markdown, \
         mock.patch("app.st.caption") as mock_caption:

        app.render_sources_expander(sources)
        mock_expander.assert_called_once_with("📚 Nguồn tài liệu tham khảo")
        mock_markdown.assert_called_once()
        mock_caption.assert_called_once_with("Trích đoạn hướng dẫn.")

    # Với danh sách rỗng, không mở expander
    with mock.patch("app.st.expander") as mock_expander:
        app.render_sources_expander([])
        mock_expander.assert_not_called()


def test_ui_long_term_memory_management(temp_db):
    """Kiểm tra các hàm thao tác bộ nhớ dài hạn trong UI (lưu, đọc, xóa)."""
    # Ban đầu không có memory
    assert memory.load_memories() == []

    # Thêm memory mới
    memory.save_memory("Người dùng thích lập trình Python")
    memory.save_memory("Người dùng sử dụng hệ điều hành Windows")

    memories = memory.load_memories()
    assert len(memories) == 2
    mem_id_1, content_1 = memories[0]
    assert content_1 == "Người dùng thích lập trình Python"

    # Xóa memory
    memory.delete_memory(mem_id_1)
    remaining = memory.load_memories()
    assert len(remaining) == 1
    assert remaining[0][1] == "Người dùng sử dụng hệ điều hành Windows"


def test_ui_reset_conversation_loads_recent_10(temp_db):
    """Kiểm tra logic nạp lại 10 tin nhắn gần nhất khi reset conversation."""
    conversation_id = "main"

    # Lưu 15 tin nhắn vào DB
    for i in range(15):
        role = "user" if i % 2 == 0 else "assistant"
        memory.save_message(conversation_id, role, f"Message {i}")

    # Reset nạp 10 tin nhắn gần nhất
    recent = memory.load_recent_messages(conversation_id, limit=10)
    assert len(recent) == 10
    # Tin nhắn đầu tiên trong danh sách 10 tin là Message 5, cuối cùng là Message 14
    assert recent[0]["content"] == "Message 5"
    assert recent[-1]["content"] == "Message 14"

