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

    fake_config = types.ModuleType("chromadb.config")
    fake_config.Settings = mock.MagicMock(name="Settings")
    fake_chromadb.config = fake_config
    sys.modules["chromadb.config"] = fake_config

    fake_default_collection = mock.MagicMock(name="fake_default_collection")
    fake_default_collection.count.return_value = 1
    fake_default_collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}

    fake_client_instance = mock.MagicMock(name="fake_client_instance")
    fake_client_instance.get_or_create_collection.return_value = fake_default_collection

    fake_chromadb.PersistentClient = mock.MagicMock(
        name="PersistentClient",
        return_value=fake_client_instance
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


@pytest.fixture
def fake_embedding_model(monkeypatch):
    import rag
    import numpy as np

    model = mock.MagicMock(name="embedding_model")
    model.encode.side_effect = lambda texts: np.zeros(
        (len(texts), 384),
        dtype=np.float32
    )

    monkeypatch.setattr(rag, "embedding_model", model)
    monkeypatch.setattr(rag, "get_embedding_model", lambda: model)
    monkeypatch.setattr(rag, "_get_active_embedding_model", lambda: model)

    return model


@pytest.fixture
def fake_collection(monkeypatch):
    import rag

    collection = mock.MagicMock(name="collection")
    collection.count.return_value = 1
    collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}

    monkeypatch.setattr(rag, "collection", collection)
    monkeypatch.setattr(rag, "get_collection", lambda: collection)
    monkeypatch.setattr(rag, "_get_active_collection", lambda: collection)

    return collection

