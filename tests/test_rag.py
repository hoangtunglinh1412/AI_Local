"""
Unit test cho rag.py.

- split_text(): Recursive Character Splitting (\\n\\n -> \\n -> câu -> space).
- search / retrieve / index: embedding model và ChromaDB được mock.
"""

import os
import re

from unittest import mock

import numpy as np
import pytest

import rag


CHUNK_SIZE = 800
OVERLAP = 150


# =========================
# HELPERS
# =========================

def _texts(chunks):

    return [chunk["text"] for chunk in chunks]


def _sentence(index, words=12):
    """Câu duy nhất, mỗi từ có định danh riêng để kiểm tra cắt giữa từ."""

    body = " ".join(f"s{index}w{j}" for j in range(words))

    return body[0].upper() + body[1:] + "."


def _paragraph(start, sentences):

    return " ".join(
        _sentence(i) for i in range(start, start + sentences)
    )


def _longest_suffix_prefix(previous, current):
    """Độ dài phần cuối của previous trùng với phần đầu của current."""

    best = 0

    for length in range(1, min(len(previous), len(current)) + 1):

        if previous.endswith(current[:length]):
            best = length

    return best


# =========================
# split_text: CƠ BẢN
# =========================

def test_split_text_default_parameters():

    import inspect

    params = inspect.signature(rag.split_text).parameters

    assert params["chunk_size"].default == 800
    assert params["overlap"].default == 150


def test_split_text_empty_text_returns_no_chunks():

    assert rag.split_text("") == []
    assert rag.split_text("   \n\n  \n") == []


def test_split_text_short_text_is_single_chunk():

    chunks = rag.split_text("Mean là giá trị trung bình cộng.")

    assert chunks == [
        {"text": "Mean là giá trị trung bình cộng.", "section": ""}
    ]


@pytest.mark.parametrize(
    "chunk_size, overlap",
    [(0, 0), (-1, 0), (100, 100), (100, 150), (100, -1)]
)
def test_split_text_rejects_invalid_parameters(chunk_size, overlap):

    with pytest.raises(ValueError):
        rag.split_text("abc", chunk_size=chunk_size, overlap=overlap)


def test_split_text_handles_crlf_line_endings():

    text = "Dòng một.\r\n\r\nDòng hai."

    chunks = rag.split_text(text)

    assert len(chunks) == 1
    assert "\r" not in chunks[0]["text"]
    assert chunks[0]["text"] == "Dòng một.\n\nDòng hai."


# =========================
# split_text: SECTION METADATA
# =========================

def test_split_text_detects_uppercase_sections():

    text = (
        "MEAN\n\n"
        "Mean là giá trị trung bình.\n\n"
        "MEDIAN\n\n"
        "Median là giá trị ở giữa."
    )

    chunks = rag.split_text(text)

    assert chunks == [
        {"text": "Mean là giá trị trung bình.", "section": "MEAN"},
        {"text": "Median là giá trị ở giữa.", "section": "MEDIAN"}
    ]


def test_split_text_keeps_legacy_section_alias():

    chunks = rag.split_text("THỐNG KÊ CƠ BẢN\nNội dung thống kê.")

    assert chunks == [
        {"text": "Nội dung thống kê.", "section": "MEAN"}
    ]


def test_split_text_never_merges_across_sections():

    text = (
        "MEAN\n" + _paragraph(0, 2) + "\n"
        "MEDIAN\n" + _paragraph(10, 2)
    )

    chunks = rag.split_text(text)

    assert [c["section"] for c in chunks] == ["MEAN", "MEDIAN"]
    assert "s10w0" not in chunks[0]["text"]
    assert "s0w0" not in chunks[1]["text"].lower()


# =========================
# split_text: RECURSIVE HIERARCHY
# =========================

def test_split_text_all_chunks_within_chunk_size():

    paragraphs = [_paragraph(i * 10, 6) for i in range(10)]

    chunks = rag.split_text("\n\n".join(paragraphs))

    assert len(chunks) > 1
    assert all(len(text) <= CHUNK_SIZE for text in _texts(chunks))


def test_split_text_level1_keeps_paragraphs_intact():
    """Đoạn văn vừa với chunk_size thì không bị cắt."""

    paragraphs = [_paragraph(i * 10, 4) for i in range(6)]

    assert all(len(p) < CHUNK_SIZE for p in paragraphs)

    texts = _texts(rag.split_text("\n\n".join(paragraphs)))

    for paragraph in paragraphs:
        assert any(paragraph in text for text in texts)


def test_split_text_level2_splits_long_paragraph_by_lines():

    lines = [_sentence(i, words=10) for i in range(40)]

    paragraph = "\n".join(lines)

    assert len(paragraph) > CHUNK_SIZE

    texts = _texts(rag.split_text(paragraph))

    assert len(texts) > 1

    for line in lines:
        assert any(line in text for text in texts)

    # Mỗi chunk được nối bằng "\n" (ranh giới dòng), không phải khoảng trắng.
    for text in texts:
        for part in text.split("\n"):
            assert part in lines


def test_split_text_level3_splits_long_line_by_sentences():

    sentences = [_sentence(i, words=10) for i in range(30)]

    line = " ".join(sentences)

    assert "\n" not in line
    assert len(line) > CHUNK_SIZE

    texts = _texts(rag.split_text(line))

    assert len(texts) > 1

    for sentence in sentences:
        assert any(sentence in text for text in texts)

    # Chunk luôn kết thúc tại cuối một câu.
    for text in texts:
        assert text.endswith(".")


def test_split_text_supports_question_and_exclamation_terminators():

    sentences = []

    for i in range(30):
        terminator = ["?", "!", "."][i % 3]
        sentences.append(_sentence(i, words=10)[:-1] + terminator)

    texts = _texts(rag.split_text(" ".join(sentences)))

    for sentence in sentences:
        assert any(sentence in text for text in texts)

    for text in texts:
        assert text[-1] in ".?!"


def test_split_text_level4_whitespace_fallback_never_cuts_words():

    words = [f"word{i:04d}" for i in range(400)]

    text = " ".join(words)  # Không có \n và không có dấu câu.

    chunks = rag.split_text(text)

    assert len(chunks) > 1

    original_words = set(words)

    for chunk_text in _texts(chunks):

        assert len(chunk_text) <= CHUNK_SIZE

        for token in chunk_text.split():
            assert token in original_words


def test_split_text_word_longer_than_chunk_size_is_kept_whole():

    long_word = "x" * (CHUNK_SIZE + 50)

    texts = _texts(rag.split_text("đầu " + long_word + " cuối"))

    assert long_word in texts


def test_split_text_without_overlap_preserves_all_content():

    paragraphs = [_paragraph(i * 10, 7) for i in range(8)]

    text = "\n\n".join(paragraphs)

    texts = _texts(rag.split_text(text, overlap=0))

    assert " ".join(texts).split() == text.split()


def test_split_text_small_custom_chunk_size():

    text = _paragraph(0, 20)

    chunks = rag.split_text(text, chunk_size=200, overlap=40)

    assert all(len(c["text"]) <= 200 for c in chunks)


# =========================
# split_text: OVERLAP
# =========================

def test_split_text_consecutive_chunks_overlap():

    sentences = [_sentence(i, words=10) for i in range(40)]

    texts = _texts(rag.split_text(" ".join(sentences)))

    assert len(texts) >= 3

    for previous, current in zip(texts, texts[1:]):

        overlap = _longest_suffix_prefix(previous, current)

        assert 0 < overlap <= OVERLAP


def test_split_text_overlap_starts_at_sentence_boundary_when_possible():

    sentences = [_sentence(i, words=6) for i in range(60)]

    texts = _texts(rag.split_text(" ".join(sentences)))

    for current in texts[1:]:

        first_sentence = re.split(r"(?<=[.?!])\s+", current)[0]

        assert first_sentence in sentences


def test_split_text_overlap_never_starts_mid_word():

    words = [f"word{i:04d}" for i in range(400)]

    texts = _texts(rag.split_text(" ".join(words)))

    for current in texts[1:]:
        assert current.split()[0] in words


def test_split_text_on_real_document():

    path = os.path.join(rag.DOCUMENT_FOLDER, "statistics.txt")

    if not os.path.exists(os.path.join(os.path.dirname(rag.__file__), path)):
        pytest.skip("documents/statistics.txt không tồn tại")

    text = rag.read_txt(os.path.join(os.path.dirname(rag.__file__), path))

    chunks = rag.split_text(text)

    sections = [c["section"] for c in chunks]

    assert "MEAN" in sections
    assert "OUTLIER" in sections
    assert all(len(c["text"]) <= CHUNK_SIZE for c in chunks)
    assert all(c["text"].strip() for c in chunks)


# =========================
# EMBEDDING + CHROMADB (MOCK)
# =========================



def test_create_embeddings_encodes_chunk_texts(fake_embedding_model):

    chunks = [
        {"text": "a", "section": ""},
        {"text": "b", "section": "S"}
    ]

    embeddings = rag.create_embeddings(chunks)

    fake_embedding_model.encode.assert_called_once_with(["a", "b"])

    assert embeddings.shape == (2, 384)


def test_save_to_vector_database_upserts_ids_and_metadata(fake_collection):

    chunks = [
        {"text": "chunk 0", "section": "MEAN"},
        {"text": "chunk 1", "section": "MEDIAN"}
    ]

    rag.save_to_vector_database(
        chunks,
        np.zeros((2, 384)),
        "doc.txt"
    )

    kwargs = fake_collection.upsert.call_args.kwargs

    assert kwargs["ids"] == ["doc.txt_0", "doc.txt_1"]
    assert kwargs["documents"] == ["chunk 0", "chunk 1"]
    assert kwargs["metadatas"] == [
        {"filename": "doc.txt", "chunk": 0, "section": "MEAN"},
        {"filename": "doc.txt", "chunk": 1, "section": "MEDIAN"}
    ]
    assert len(kwargs["embeddings"]) == 2
    assert len(kwargs["embeddings"][0]) == 384


def test_search_documents_queries_collection_with_embedding(
    fake_embedding_model,
    fake_collection
):

    fake_collection.query.return_value = {"documents": [["x"]]}

    results = rag.search_documents("Mean là gì?", top_k=2)

    fake_embedding_model.encode.assert_called_once_with(["Mean là gì?"])

    kwargs = fake_collection.query.call_args.kwargs

    assert kwargs["n_results"] == 2
    assert len(kwargs["query_embeddings"][0]) == 384
    assert results == {"documents": [["x"]]}


def test_retrieve_documents_filters_by_distance(
    fake_embedding_model,
    fake_collection
):

    fake_collection.query.return_value = {
        "documents": [["gần", "xa"]],
        "metadatas": [[
            {"filename": "a.txt", "chunk": 0, "section": "MEAN"},
            {"filename": "b.txt", "chunk": 3, "section": "MODE"}
        ]],
        "distances": [[5.0, 30.0]]
    }

    context, sources = rag.retrieve_documents(
        "query",
        top_k=2,
        distance_threshold=18
    )

    assert "gần" in context
    assert "xa" not in context
    assert sources == [
        {"filename": "a.txt", "chunk": 0, "section": "MEAN", "distance": 5.0}
    ]


def test_index_document_splits_embeds_and_saves(
    tmp_path,
    monkeypatch,
    fake_embedding_model,
    fake_collection
):

    monkeypatch.setattr(rag, "DOCUMENT_FOLDER", str(tmp_path))

    text = "MEAN\n\n" + "\n\n".join(
        _paragraph(i * 10, 6) for i in range(5)
    )

    (tmp_path / "doc.txt").write_text(text, encoding="utf-8")

    chunk_count = rag.index_document("doc.txt")

    expected = rag.split_text(text)

    assert chunk_count == len(expected) > 1

    kwargs = fake_collection.upsert.call_args.kwargs

    assert kwargs["documents"] == _texts(expected)
    assert all(m["section"] == "MEAN" for m in kwargs["metadatas"])


def test_index_document_missing_file_returns_zero(
    tmp_path,
    monkeypatch,
    fake_collection
):

    monkeypatch.setattr(rag, "DOCUMENT_FOLDER", str(tmp_path))

    assert rag.index_document("missing.txt") == 0

    fake_collection.upsert.assert_not_called()

