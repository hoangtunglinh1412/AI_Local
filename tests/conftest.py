"""
Fixtures dùng chung cho test suite.

Toàn bộ test chạy offline, không cần GPU / Ollama / model embedding:

- `sentence_transformers` và `chromadb` được thay bằng module giả
  TRƯỚC khi `rag.py` được import (rag.py tạo model + client ở
  module level).
- `ollama.chat` được mock trong từng test.
- SQLite dùng file tạm trong tmp_path, không chạm `chat_history.db`.
"""

import os
import sys
import types

from unittest import mock

import pytest


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _install_fake_heavy_modules():

    fake_sentence_transformers = types.ModuleType(
        "sentence_transformers"
    )

    fake_sentence_transformers.SentenceTransformer = mock.MagicMock(
        name="SentenceTransformer"
    )

    fake_chromadb = types.ModuleType("chromadb")

    fake_chromadb.PersistentClient = mock.MagicMock(
        name="PersistentClient"
    )

    sys.modules["sentence_transformers"] = fake_sentence_transformers
    sys.modules["chromadb"] = fake_chromadb


_install_fake_heavy_modules()


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Trỏ memory.DB_NAME sang một SQLite tạm và tạo bảng messages."""

    import memory

    db_path = str(tmp_path / "test_chat_history.db")

    monkeypatch.setattr(memory, "DB_NAME", db_path)

    memory.create_database()

    return db_path


@pytest.fixture
def chatbot_module(temp_db, monkeypatch):
    """
    Import chatbot.py (vòng lặp chat không chạy khi import)
    và reset trạng thái module cho từng test.
    """

    import chatbot

    monkeypatch.setattr(chatbot, "rag_enabled", True)

    chatbot.messages.clear()

    chatbot.messages.append({
        "role": "system",
        "content": "BASE SYSTEM PROMPT"
    })

    return chatbot

