"""
Integration test cho chatbot.py:

- /rag on | /rag off | /rag status
- Chèn context tài liệu vào system prompt (CONTEXT TÀI LIỆU:)
  mà không ghi đè LONG-TERM MEMORY / CONVERSATION SUMMARY.
- Token streaming (stream=True) và lưu full text vào SQLite.
- Auto summarization reset RAM về 1 system + 10 message gần nhất.

Ollama và search_documents đều được mock, không cần GPU.
"""

from unittest import mock

import pytest

import memory


QUESTION = "Mean là gì?"

STREAM_TOKENS = ["Mean ", "là ", "giá trị ", "trung bình ", "cộng."]

FULL_ANSWER = "".join(STREAM_TOKENS)

SEARCH_RESULTS = {
    "ids": [["statistics.txt_0", "statistics.txt_5"]],
    "documents": [[
        "Mean là giá trị trung bình cộng của một tập dữ liệu.",
        "Outlier là quan sát khác biệt đáng kể."
    ]],
    "metadatas": [[
        {"filename": "statistics.txt", "chunk": 0, "section": "MEAN"},
        {"filename": "statistics.txt", "chunk": 5, "section": "OUTLIER"}
    ]],
    "distances": [[4.2, 9.7]]
}


# =========================
# HELPERS
# =========================

def _stream(tokens):

    for token in tokens:
        yield {"message": {"role": "assistant", "content": token}}


class FakeOllama:
    """
    Giả lập ollama.chat:
    - stream=True  -> generator token (câu trả lời chính)
    - stream khác  -> response thường (RAG router / summary)
    """

    def __init__(self, router_answer="YES", tokens=STREAM_TOKENS):

        self.router_answer = router_answer
        self.tokens = tokens
        self.calls = []

    def __call__(self, *args, **kwargs):

        self.calls.append(kwargs)

        if kwargs.get("stream"):
            return _stream(self.tokens)

        return {"message": {"role": "assistant", "content": self.router_answer}}

    @property
    def stream_calls(self):

        return [c for c in self.calls if c.get("stream")]

    @property
    def non_stream_calls(self):

        return [c for c in self.calls if not c.get("stream")]


# =========================
# IMPORT
# =========================

def test_chatbot_import_does_not_start_chat_loop(chatbot_module):

    assert chatbot_module.RUN_CHAT_LOOP is False
    assert callable(chatbot_module.answer_question)


def test_chatbot_uses_search_documents_from_rag(chatbot_module):

    import rag

    assert chatbot_module.search_documents is rag.search_documents


# =========================
# /rag COMMANDS
# =========================

def test_rag_enabled_by_default(chatbot_module):

    assert chatbot_module.rag_enabled is True

    assert "ON" in chatbot_module.handle_rag_command("/rag status")


def test_rag_off_on_status_commands(chatbot_module):

    message = chatbot_module.handle_rag_command("/rag off")

    assert chatbot_module.rag_enabled is False
    assert "TẮT" in message
    assert "OFF" in chatbot_module.handle_rag_command("/rag status")

    message = chatbot_module.handle_rag_command("/rag on")

    assert chatbot_module.rag_enabled is True
    assert "BẬT" in message
    assert "ON" in chatbot_module.handle_rag_command("/rag status")


def test_rag_command_is_case_insensitive(chatbot_module):

    chatbot_module.handle_rag_command("  /RAG OFF ")

    assert chatbot_module.rag_enabled is False


@pytest.mark.parametrize("command", ["/rag", "/rag maybe"])
def test_rag_command_invalid_prints_usage_without_changing_state(
    chatbot_module,
    command
):

    message = chatbot_module.handle_rag_command(command)

    assert "Cú pháp" in message
    assert chatbot_module.rag_enabled is True


# =========================
# RAG CONTEXT
# =========================

def test_get_rag_context_calls_search_documents_with_top_k_2(chatbot_module):

    with mock.patch.object(
        chatbot_module,
        "search_documents",
        return_value=SEARCH_RESULTS
    ) as fake_search:

        context, sources = chatbot_module.get_rag_context(QUESTION)

    fake_search.assert_called_once_with(QUESTION, top_k=2)

    assert "Mean là giá trị trung bình cộng" in context
    assert "[1] statistics.txt | Section: MEAN" in context
    assert [s["chunk"] for s in sources] == [0, 5]


def test_format_rag_context_filters_far_chunks(chatbot_module):

    results = {
        "documents": [["gần", "xa"]],
        "metadatas": [[
            {"filename": "a.txt", "chunk": 0, "section": ""},
            {"filename": "b.txt", "chunk": 1, "section": ""}
        ]],
        "distances": [[1.0, 99.0]]
    }

    context, sources = chatbot_module.format_rag_context(results)

    assert context == "[1] a.txt\ngần"
    assert sources == [
        {"filename": "a.txt", "chunk": 0, "section": "", "distance": 1.0}
    ]


def test_format_rag_context_empty_results(chatbot_module):

    empty = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    assert chatbot_module.format_rag_context(empty) == ("", [])
    assert chatbot_module.format_rag_context({}) == ("", [])


def test_get_rag_context_survives_search_error(chatbot_module, capsys):

    with mock.patch.object(
        chatbot_module,
        "search_documents",
        side_effect=RuntimeError("chroma down")
    ):

        assert chatbot_module.get_rag_context(QUESTION) == ("", [])

    assert "chroma down" in capsys.readouterr().out


def test_build_system_prompt_keeps_memory_and_summary_with_context(
    chatbot_module
):

    memory.save_memory("Người dùng đang học Data Science.")
    memory.save_summary(chatbot_module.CONVERSATION_ID, "Đang ôn thống kê.")

    prompt = chatbot_module.build_system_prompt("NỘI DUNG TÀI LIỆU XYZ")

    assert "Người dùng đang học Data Science." in prompt
    assert "Đang ôn thống kê." in prompt
    assert "CONTEXT TÀI LIỆU:" in prompt
    assert "NỘI DUNG TÀI LIỆU XYZ" in prompt

    assert (
        prompt.index("LONG-TERM MEMORY:")
        < prompt.index("CONVERSATION SUMMARY:")
        < prompt.index("CONTEXT TÀI LIỆU:")
    )


def test_build_system_prompt_without_context_has_no_context_header(
    chatbot_module
):

    prompt = chatbot_module.build_system_prompt()

    assert "CONTEXT TÀI LIỆU:" not in prompt
    assert "LONG-TERM MEMORY:" in prompt


def test_rebuild_system_prompt_never_contains_rag_context(chatbot_module):

    chatbot_module.rebuild_system_prompt()

    assert chatbot_module.messages[0]["role"] == "system"
    assert "CONTEXT TÀI LIỆU:" not in chatbot_module.messages[0]["content"]


# =========================
# STREAMING
# =========================

def test_stream_chat_response_uses_stream_true_and_prints_tokens(
    chatbot_module,
    capsys
):

    fake = FakeOllama()

    with mock.patch("ollama.chat", side_effect=fake):

        answer = chatbot_module.stream_chat_response(
            [{"role": "user", "content": QUESTION}]
        )

    assert answer == FULL_ANSWER

    call = fake.calls[0]

    assert call["stream"] is True
    assert call["model"] == chatbot_module.MODEL
    assert call["options"] == {"num_ctx": 8192}

    assert "AI: " + FULL_ANSWER in capsys.readouterr().out


def test_stream_chat_response_prints_each_token_as_it_arrives(
    chatbot_module
):

    printed = []

    def generator():

        for token in STREAM_TOKENS:
            # Token trước phải được in xong trước khi token sau được sinh ra.
            yield {"message": {"content": token}}

    with mock.patch("ollama.chat", return_value=generator()), \
         mock.patch("builtins.print", side_effect=lambda *a, **k: printed.append((a, k))):

        chatbot_module.stream_chat_response([])

    token_prints = [a[0] for a, k in printed if a and a[0] in STREAM_TOKENS]

    assert token_prints == STREAM_TOKENS

    for args, kwargs in printed:
        if args and args[0] in STREAM_TOKENS:
            assert kwargs.get("end") == ""
            assert kwargs.get("flush") is True


def test_stream_chat_response_ignores_empty_tokens(chatbot_module):

    with mock.patch(
        "ollama.chat",
        return_value=_stream(["", "Xin ", None, "chào"])
    ):

        assert chatbot_module.stream_chat_response([]) == "Xin chào"


def test_stream_chat_response_with_real_ollama_response_objects(
    chatbot_module
):

    ollama = pytest.importorskip("ollama")

    chunks = [
        ollama.ChatResponse(
            model="qwen3:8b",
            message=ollama.Message(role="assistant", content=token)
        )
        for token in STREAM_TOKENS
    ]

    with mock.patch("ollama.chat", return_value=iter(chunks)):

        assert chatbot_module.stream_chat_response([]) == FULL_ANSWER


# =========================
# FULL CHAT TURN
# =========================

def test_answer_question_rag_on_injects_context_streams_and_saves(
    chatbot_module,
    capsys
):

    memory.save_memory("Người dùng đang học Data Science.")
    memory.save_summary(chatbot_module.CONVERSATION_ID, "Đang ôn thống kê.")

    chatbot_module.rebuild_system_prompt()

    base_system_prompt = chatbot_module.messages[0]["content"]

    fake = FakeOllama(router_answer="YES")

    with mock.patch("ollama.chat", side_effect=fake), \
         mock.patch.object(
             chatbot_module,
             "search_documents",
             return_value=SEARCH_RESULTS
         ) as fake_search:

        answer = chatbot_module.answer_question(QUESTION)

    assert answer == FULL_ANSWER

    fake_search.assert_called_once_with(QUESTION, top_k=2)

    # 1 lần router (không stream) + 1 lần trả lời chính (stream)
    assert len(fake.non_stream_calls) == 1
    assert len(fake.stream_calls) == 1

    sent_messages = fake.stream_calls[0]["messages"]

    system_content = sent_messages[0]["content"]

    assert sent_messages[0]["role"] == "system"
    assert "CONTEXT TÀI LIỆU:" in system_content
    assert "Mean là giá trị trung bình cộng" in system_content
    assert "Người dùng đang học Data Science." in system_content
    assert "Đang ôn thống kê." in system_content

    # Câu hỏi được gửi nguyên văn
    assert sent_messages[-1] == {"role": "user", "content": QUESTION}

    # Context không bị lưu lại vào system prompt cố định
    assert chatbot_module.messages[0]["content"] == base_system_prompt

    # RAM: user + assistant (full text)
    assert chatbot_module.messages[-2:] == [
        {"role": "user", "content": QUESTION},
        {"role": "assistant", "content": FULL_ANSWER}
    ]

    # SQLite: full text của câu trả lời được lưu
    assert memory.load_messages(chatbot_module.CONVERSATION_ID) == [
        {"role": "user", "content": QUESTION},
        {"role": "assistant", "content": FULL_ANSWER}
    ]

    output = capsys.readouterr().out

    assert "[RAG Router]: YES" in output
    assert "AI: " + FULL_ANSWER in output
    assert "File: statistics.txt" in output
    assert "Section: MEAN" in output


def test_answer_question_rag_off_skips_router_and_search(
    chatbot_module,
    capsys
):

    chatbot_module.handle_rag_command("/rag off")

    fake = FakeOllama()

    with mock.patch("ollama.chat", side_effect=fake), \
         mock.patch.object(chatbot_module, "search_documents") as fake_search:

        answer = chatbot_module.answer_question(QUESTION)

    assert answer == FULL_ANSWER

    fake_search.assert_not_called()

    assert fake.non_stream_calls == []
    assert len(fake.stream_calls) == 1

    system_content = fake.stream_calls[0]["messages"][0]["content"]

    assert "CONTEXT TÀI LIỆU:" not in system_content

    output = capsys.readouterr().out

    assert "[RAG Router]: OFF" in output
    assert "Không sử dụng tài liệu." in output

    assert memory.count_messages(chatbot_module.CONVERSATION_ID) == 2


def test_answer_question_router_no_skips_search(chatbot_module):

    fake = FakeOllama(router_answer="NO")

    with mock.patch("ollama.chat", side_effect=fake), \
         mock.patch.object(chatbot_module, "search_documents") as fake_search:

        chatbot_module.answer_question("Xin chào")

    fake_search.assert_not_called()

    system_content = fake.stream_calls[0]["messages"][0]["content"]

    assert "CONTEXT TÀI LIỆU:" not in system_content


def test_answer_question_no_relevant_chunks_keeps_base_prompt(
    chatbot_module,
    capsys
):

    far_results = {
        "documents": [["không liên quan"]],
        "metadatas": [[{"filename": "a.txt", "chunk": 0, "section": ""}]],
        "distances": [[99.0]]
    }

    fake = FakeOllama(router_answer="YES")

    with mock.patch("ollama.chat", side_effect=fake), \
         mock.patch.object(
             chatbot_module,
             "search_documents",
             return_value=far_results
         ):

        chatbot_module.answer_question(QUESTION)

    sent_messages = fake.stream_calls[0]["messages"]

    assert sent_messages[0] == {
        "role": "system",
        "content": "BASE SYSTEM PROMPT"
    }

    assert "Không tìm thấy tài liệu liên quan" in capsys.readouterr().out


# =========================
# AUTO SUMMARIZATION
# =========================

def test_auto_summarize_saves_summary_checkpoint_and_resets_ram(
    chatbot_module
):

    conversation_id = chatbot_module.CONVERSATION_ID

    for i in range(11):
        memory.save_message(conversation_id, "user", f"câu hỏi {i}")
        memory.save_message(conversation_id, "assistant", f"trả lời {i}")

    last_id = memory.load_messages_after(conversation_id, 0)[-1]["id"]

    fake = FakeOllama(router_answer="SUMMARY: người dùng hỏi về thống kê")

    with mock.patch("ollama.chat", side_effect=fake):

        chatbot_module.auto_summarize()

    assert memory.load_summary(conversation_id) == (
        "SUMMARY: người dùng hỏi về thống kê"
    )

    assert memory.load_summary_checkpoint(conversation_id) == last_id

    # 1 system prompt + 10 message gần nhất
    assert len(chatbot_module.messages) == 11
    assert chatbot_module.messages[0]["role"] == "system"
    assert "SUMMARY: người dùng hỏi về thống kê" in (
        chatbot_module.messages[0]["content"]
    )
    assert chatbot_module.messages[-1] == {
        "role": "assistant",
        "content": "trả lời 10"
    }

