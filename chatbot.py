import re
import queue
import threading
from typing import Tuple, Optional, List, Dict, Any

import ollama

from memory import (
    create_database,
    save_message,
    load_messages,
    save_memory,
    load_memories,
    delete_memory,
    update_memory,
    save_summary,
    load_summary,
    load_recent_messages,
    count_messages,
    save_summary_checkpoint,
    load_summary_checkpoint,
    load_messages_after
)

from rag import (
    search_documents,
    list_documents,
    index_document,
    delete_document_from_database,
    delete_document_file,
    reindex_all_documents,
    has_indexed_documents
)

from data_analyzer import (
    load_data,
    generate_report
)

from data_agent import (
    run_data_agent
)

MODEL = "qwen3:8b"

NUM_CTX = 8192

MAX_INPUT_CHARS = 16000

# =========================
# DATA AGENT PENDING STATE
# =========================

pending_data_question = None
pending_data_filename = None

CONVERSATION_ID = "main"

SUMMARY_THRESHOLD = 20

# =========================
# RAG STATE
# =========================

rag_enabled = True

RAG_TOP_K = 2

RAG_DISTANCE_THRESHOLD = 18

# =========================
# HEURISTIC RAG ROUTER PATTERNS
# =========================

GREETING_PATTERNS = re.compile(
    r"^(?:xin\s*chào|chào(?:\s*(?:bạn|cậu|ad|bot|em|anh|chị))?|hello|hi|hey|alo|alô|"
    r"cảm\s*ơn(?:\s*(?:bạn|cậu|nhiều|nhé|ad))*|thank(?:\s*you)?(?:\s*so\s*much)?|"
    r"tạm\s*biệt|bye(?:\s*bye)?|"
    r"bạn\s*khỏe\s*không|bạn\s*tên\s*(?:là\s*)?gì|who\s*are\s*you|"
    r"viết\s*(?:cho\s*tôi\s*)?(?:một\s*)?(?:bài\s*)?thơ|"
    r"kể\s*(?:cho\s*tôi\s*)?(?:một\s*)?câu\s*chuyện)\b[!?.\s]*$",
    re.IGNORECASE
)

RAG_TRIGGER_PATTERNS = re.compile(
    r"(?:\blà\s*gì\b|\bđịnh\s*nghĩa\b|\bkhái\s*niệm\b|\bgiải\s*thích\b|"
    r"\bcông\s*thức\b|\btính\s*(?:như\s*thế\s*nào|như\s*nào|sao)\b|"
    r"\bđặc\s*điểm\b|\bý\s*nghĩa\b|\bvai\s*trò\b|"
    r"\bwhat\s+is\b|\bdefine\b|\bexplain\b|"
    r"\bmean\b|\bmedian\b|\bmode\b|\bvariance\b|\bstd\b|\boutlier\b|"
    r"\bmachine\s*learning\b|\bsupervised\s*learning\b|\bunsupervised\b|"
    r"\bmultilayer\s*perceptron\b|\bneural\s*network\b|\bperceptron\b|"
    r"\btài\s*liệu\b|\btệp\b|\bfile\b|\btrong\s*sách\b)",
    re.IGNORECASE
)

# =========================
# CREATE DATABASE
# =========================

create_database()

# =========================
# SYSTEM PROMPT BUILDER
# =========================

def build_system_prompt(rag_context=""):
    """
    Tạo system prompt gồm LONG-TERM MEMORY + CONVERSATION SUMMARY.
    Nếu có rag_context, thêm mục riêng CONTEXT TÀI LIỆU ở cuối
    (không ghi đè memory/summary).
    """
    memories_list = load_memories()

    mem_text = ""
    for memory_id, content in memories_list:
        mem_text += f"- {content}\n"

    conversation_summary = load_summary(CONVERSATION_ID)

    new_system_prompt = f"""
Bạn là một trợ lý AI local.

Hãy trả lời bằng tiếng Việt nếu người dùng sử dụng tiếng Việt.
Trả lời rõ ràng, chính xác và dễ hiểu.


LONG-TERM MEMORY:

{mem_text}


CONVERSATION SUMMARY:

{conversation_summary}


Hãy sử dụng những thông tin trên khi chúng thực sự hữu ích.

Không được nói rằng bạn biết thông tin này
từ hệ thống memory hoặc summary.
"""

    if rag_context:
        new_system_prompt += f"""

CONTEXT TÀI LIỆU:

{rag_context}


Nếu CONTEXT TÀI LIỆU có thông tin liên quan,
hãy ưu tiên sử dụng thông tin trong CONTEXT TÀI LIỆU.

Nếu CONTEXT TÀI LIỆU không có đủ thông tin,
hãy trả lời dựa trên kiến thức thông thường của bạn.

Không được tự ý bịa thông tin từ tài liệu.
"""

    return new_system_prompt


# Khởi tạo messages với đầy đủ Long-term Memory và Summary ngay từ đầu
messages = [
    {
        "role": "system",
        "content": build_system_prompt()
    }
]

recent_messages = load_recent_messages(
    CONVERSATION_ID,
    10
)

messages.extend(recent_messages)


def rebuild_system_prompt():
    messages[0] = {
        "role": "system",
        "content": build_system_prompt()
    }


# =========================
# BACKGROUND MEMORY WORKER
# =========================

memory_queue = queue.Queue()
_system_prompt_dirty = False


def parse_memory_decision(raw_output: str) -> Tuple[str, Optional[int], str]:
    """
    Phân tích quyết định của Memory Manager từ raw output của LLM.
    Trả về: (action, target_id, updated_content)
    action: "NEW" | "UPDATE" | "IGNORE"
    """
    if not raw_output:
        return "IGNORE", None, ""

    text = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()

    if "UPDATE" in text.upper():
        id_match = re.search(r"ID\s*[:#]?\s*(\d+)", text, re.IGNORECASE)
        content_match = re.search(r"CONTENT\s*[:#]?\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        target_id = int(id_match.group(1)) if id_match else None
        content = content_match.group(1).strip() if content_match else ""
        if target_id is not None:
            return "UPDATE", target_id, content

    if re.search(r"\bNEW\b", text, re.IGNORECASE):
        return "NEW", None, ""

    if re.search(r"\bIGNORE\b", text, re.IGNORECASE):
        return "IGNORE", None, ""

    return "IGNORE", None, ""


def check_memory(question: str) -> str:
    memory_prompt = f"""
Bạn là hệ thống quản lý bộ nhớ cho một AI assistant.

Hãy phân tích câu nói của người dùng bên dưới.

Chỉ lưu những thông tin có khả năng hữu ích trong các cuộc trò chuyện sau này.

NÊN LƯU:
- Sở thích lâu dài
- Thông tin về người dùng
- Mục tiêu học tập hoặc công việc
- Dự án người dùng đang thực hiện
- Cách người dùng muốn AI trả lời
- Những thông tin người dùng nói rõ là muốn AI ghi nhớ

KHÔNG NÊN LƯU:
- Câu hỏi thông thường
- Kiến thức chung
- Những thông tin chỉ có giá trị trong vài phút
- Nội dung của câu hỏi
- Câu trả lời của AI

Nếu cần lưu, trả về:

SAVE: <thông tin cần lưu>

Nếu không cần lưu, chỉ trả về:

NONE

Câu nói của người dùng:
{question[:2000]}
"""

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": memory_prompt
                }
            ],
            options={
                "num_ctx": NUM_CTX
            },
            think=False
        )

        result = response["message"]["content"]
        result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
        return result
    except Exception as error:
        print("[Memory]: Lỗi khi gọi Ollama:", error)
        return "NONE"


def manage_memory(new_memory: str, current_memories: list) -> str:
    manager_prompt = f"""
Bạn là hệ thống quản lý bộ nhớ cho AI assistant.

Nhiệm vụ của bạn là quyết định memory mới nên:
- NEW
- UPDATE
- IGNORE

MEMORY MỚI:
{new_memory}

CÁC MEMORY HIỆN TẠI:
"""

    for memory_id, content in current_memories:
        manager_prompt += f"""
ID {memory_id}: {content}
"""

    manager_prompt += """
QUY TẮC:

1. NEW: Chọn NEW nếu thông tin mới hoàn toàn khác.
2. UPDATE: Chọn UPDATE nếu memory mới bổ sung hoặc cập nhật memory cũ.
3. IGNORE: Chọn IGNORE nếu thông tin đã tồn tại hoặc không cần thiết.

FORMAT TRẢ VỀ:
- NEW
- IGNORE
- UPDATE: ID: <id>
CONTENT: <nội dung memory mới>

CHỈ trả về đúng một trong các format trên. Không giải thích.
"""

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": manager_prompt
                }
            ],
            options={
                "num_ctx": NUM_CTX
            },
            think=False
        )

        result = response["message"]["content"]
        result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
        return result
    except Exception as error:
        print("[Memory Manager]: Lỗi khi gọi Ollama:", error)
        return "IGNORE"


def _process_memory_task(question: str) -> None:
    global _system_prompt_dirty
    try:
        memory_result = check_memory(question)
        if not memory_result or not memory_result.startswith("SAVE:"):
            return

        new_memory = memory_result[5:].strip()
        if not new_memory:
            return

        current_memories = load_memories()
        manager_result = manage_memory(new_memory, current_memories)
        action, target_id, updated_content = parse_memory_decision(manager_result)

        if action == "NEW":
            save_memory(new_memory)
            _system_prompt_dirty = True
            print(f"\n[Memory Background]: Đã tạo memory mới: {new_memory}")
        elif action == "UPDATE":
            existing_ids = {m[0] for m in current_memories}
            if target_id is not None and target_id in existing_ids:
                content_to_save = updated_content if updated_content else new_memory
                update_memory(target_id, content_to_save)
                _system_prompt_dirty = True
                print(f"\n[Memory Background]: Đã cập nhật memory ID {target_id}")
            else:
                save_memory(new_memory)
                _system_prompt_dirty = True
                print(f"\n[Memory Background]: ID {target_id} không tồn tại -> Đã lưu thành memory mới.")
        elif action == "IGNORE":
            pass
    except Exception as err:
        print("[Memory Background]: Lỗi khi xử lý memory:", err)


def enqueue_memory_check(question: str) -> None:
    """Đưa câu hỏi vào hàng đợi xử lý bộ nhớ chạy ngầm."""
    if not question or not str(question).strip():
        return
    memory_queue.put(str(question).strip())


def _background_memory_worker():
    while True:
        try:
            task_q = memory_queue.get()
            if task_q is None:
                break
            _process_memory_task(task_q)
        except Exception as e:
            print("[Memory Worker Error]:", e)
        finally:
            memory_queue.task_done()


_memory_thread = threading.Thread(target=_background_memory_worker, daemon=True)
_memory_thread.start()


def wait_for_background_memory(timeout: float = 3.0) -> None:
    """Chờ các tác vụ memory chạy ngầm hoàn tất (dùng khi thoát hoặc kiểm thử)."""
    import time
    start = time.time()
    while not memory_queue.empty() and (time.time() - start < timeout):
        time.sleep(0.05)


# =========================
# SUMMARIZATION
# =========================

def summarize_conversation(chat_history_messages, old_summary=""):
    conversation_text = ""
    for message in chat_history_messages:
        if message["role"] == "user":
            conversation_text += "User: " + message["content"] + "\n"
        elif message["role"] == "assistant":
            conversation_text += "Assistant: " + message["content"] + "\n"

    summary_prompt = f"""
Bạn là hệ thống tóm tắt cuộc hội thoại.

Hãy cập nhật conversation summary dựa trên summary trước đó và các message mới.
Chỉ giữ lại chủ đề chính, quyết định quan trọng và thông tin cần thiết.
Viết summary bằng tiếng Việt, ngắn gọn và rõ ràng.

SUMMARY TRƯỚC ĐÓ:
{old_summary}

CÁC MESSAGE MỚI:
{conversation_text}
"""

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": summary_prompt
                }
            ],
            options={
                "num_ctx": NUM_CTX
            },
            think=False
        )

        summary = response["message"]["content"]
        summary = re.sub(r"<think>.*?</think>", "", summary, flags=re.DOTALL).strip()
        return summary
    except Exception as error:
        print("[Summary]: Lỗi khi gọi Ollama:", error)
        return old_summary


def auto_summarize():
    print("\n[Summary] Cuộc hội thoại đang dài. Đang tạo summary...")

    checkpoint = load_summary_checkpoint(CONVERSATION_ID)
    messages_to_summarize = load_messages_after(CONVERSATION_ID, checkpoint)

    if len(messages_to_summarize) == 0:
        print("[Summary] Không có message mới để summarize.")
        return

    simple_messages = []
    for message in messages_to_summarize:
        simple_messages.append({
            "role": message["role"],
            "content": message["content"]
        })

    old_summary = load_summary(CONVERSATION_ID)
    conversation_summary = summarize_conversation(simple_messages, old_summary)

    save_summary(CONVERSATION_ID, conversation_summary)
    last_message_id = messages_to_summarize[-1]["id"]
    save_summary_checkpoint(CONVERSATION_ID, last_message_id)

    reset_recent_messages()
    print("[DEBUG] Messages trong RAM:", len(messages))
    print("[Summary] Đã cập nhật summary.")
    print("[Summary] Checkpoint:", last_message_id)


def reset_recent_messages():
    rec_msgs = load_recent_messages(CONVERSATION_ID, 10)
    messages.clear()
    messages.append({
        "role": "system",
        "content": ""
    })
    messages.extend(rec_msgs)
    rebuild_system_prompt()


# =========================
# RAG ROUTER & CONTEXT
# =========================

def classify_query_heuristic(question: str) -> Optional[bool]:
    """
    Phân loại nhanh bằng quy tắc regex/heuristic (0 LLM call):
    - Trả về False: Chắc chắn không cần RAG (chào hỏi, xã giao, yêu cầu sáng tạo).
    - Trả về True: Chắc chắn cần RAG (hỏi định nghĩa, khái niệm, công thức, tài liệu).
    - Trả về None: Không rõ ràng -> Chuyển sang LLM router.
    """
    q = question.strip()
    if not q:
        return False
    if GREETING_PATTERNS.match(q):
        return False
    if RAG_TRIGGER_PATTERNS.search(q):
        return True
    return None


def _llm_route_rag(question: str) -> bool:
    router_prompt = f"""
Bạn là bộ định tuyến RAG cho một chatbot có tài liệu nội bộ.

Nhiệm vụ của bạn là quyết định câu hỏi của người dùng
có NÊN được tìm kiếm trong tài liệu nội bộ hay không.

Trả về DUY NHẤT một trong hai giá trị:
YES
NO

QUY TẮC:
Trả về YES nếu câu hỏi:
- Có thể liên quan đến kiến thức trong tài liệu nội bộ.
- Hỏi về một khái niệm, thuật ngữ hoặc chủ đề có thể xuất hiện trong tài liệu.
- Hỏi định nghĩa, đặc điểm, công thức, quy trình hoặc nội dung học tập.

Trả về NO nếu câu hỏi:
- Chào hỏi, cảm xúc, hội thoại xã giao.
- Yêu cầu sáng tạo như viết thơ, viết truyện.

Câu hỏi của người dùng:
{question[:2000]}

Chỉ trả lời YES hoặc NO.
"""

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": router_prompt
                }
            ],
            options={
                "num_ctx": NUM_CTX
            },
            think=False
        )

        result = response["message"]["content"]
        result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip().upper()
        return "YES" in result
    except Exception as error:
        print("[RAG Router]: Lỗi khi gọi Ollama router, mặc định kích hoạt RAG:", error)
        return True


def should_use_rag(question: str) -> bool:
    """
    Bộ định tuyến RAG 2 tầng:
    1. Kiểm tra collection có dữ liệu không.
    2. Heuristic nhanh (0 LLM call) cho các câu hỏi rõ ràng.
    3. LLM Router cho các trường hợp còn lại.
    """
    if not has_indexed_documents():
        return False

    heuristic = classify_query_heuristic(question)
    if heuristic is not None:
        return heuristic

    return _llm_route_rag(question)


def format_rag_context(results, distance_threshold=RAG_DISTANCE_THRESHOLD):
    documents = (results.get("documents") or [[]])[0] or []
    metadatas = (results.get("metadatas") or [[]])[0] or []
    distances = (results.get("distances") or [[]])[0] or []

    context_parts = []
    sources = []

    for i, document in enumerate(documents):
        if not document or not document.strip():
            continue

        metadata = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
        distance = distances[i] if i < len(distances) else None

        if distance is not None and distance > distance_threshold:
            continue

        source = {
            "filename": metadata.get("filename", "unknown"),
            "chunk": metadata.get("chunk", i),
            "section": metadata.get("section", ""),
            "distance": distance
        }

        sources.append(source)
        header = f"[{len(sources)}] {source['filename']}"
        if source["section"]:
            header += f" | Section: {source['section']}"

        context_parts.append(header + "\n" + document.strip())

    return "\n\n".join(context_parts), sources


def get_rag_context(query):
    try:
        results = search_documents(
            query,
            top_k=RAG_TOP_K
        )
    except Exception as error:
        print("[RAG]: Lỗi khi truy vấn tài liệu:", error)
        return "", []

    return format_rag_context(results)


def handle_rag_command(command):
    global rag_enabled
    parts = command.strip().split()
    action = parts[1].lower() if len(parts) > 1 else ""

    if action == "on":
        rag_enabled = True
        return "[RAG] Đã BẬT tự động tìm kiếm tài liệu."

    if action == "off":
        rag_enabled = False
        return "[RAG] Đã TẮT tìm kiếm tài liệu."

    if action == "status":
        status = "ON" if rag_enabled else "OFF"
        return f"[RAG] Trạng thái hiện tại: {status}"

    return "[RAG] Cú pháp: /rag on | /rag off | /rag status"


def stream_chat_response(chat_messages):
    """
    Gọi Ollama với stream=True, in từng token ra terminal.
    Bắt lỗi kết nối/timeout để hiển thị thông báo thân thiện thay vì crash.
    """
    try:
        response_stream = ollama.chat(
            model=MODEL,
            messages=chat_messages,
            options={
                "num_ctx": NUM_CTX
            },
            stream=True
        )

        print("\nAI: ", end="", flush=True)
        answer_parts = []

        for chunk in response_stream:
            token = chunk.get("message", {}).get("content", "") or ""
            if token:
                print(token, end="", flush=True)
                answer_parts.append(token)

        print()
        return "".join(answer_parts)
    except Exception as error:
        print(f"\n[AI Error]: Không thể kết nối tới mô hình Ollama ({MODEL}). Chi tiết: {error}")
        return ""


def print_sources(sources):
    print("\n===== SOURCES =====")
    if len(sources) == 0:
        print("Không sử dụng tài liệu.")
        return

    for source in sources:
        print("File:", source["filename"])
        print("Section:", source["section"])
        print("Chunk:", source["chunk"])
        print("Distance:", source["distance"])
        print()


def answer_question(question):
    """
    Xử lý một lượt chat thông thường:
    - Kiểm tra độ dài đầu vào
    - Đồng bộ system prompt nếu có thay đổi ngầm
    - RAG (nếu bật) -> Stream câu trả lời -> Lưu SQLite -> Đưa vào hàng đợi memory
    """
    global _system_prompt_dirty

    if len(question) > MAX_INPUT_CHARS:
        print(f"\n[Lỗi]: Câu hỏi quá dài ({len(question)} ký tự). Giới hạn tối đa là {MAX_INPUT_CHARS} ký tự. Vui lòng rút gọn câu hỏi.")
        return ""

    if _system_prompt_dirty:
        rebuild_system_prompt()
        _system_prompt_dirty = False

    use_rag = False
    if rag_enabled:
        use_rag = should_use_rag(question)
        print("[RAG Router]:", "YES" if use_rag else "NO")
    else:
        print("[RAG Router]: OFF")

    chat_messages = messages.copy()
    sources = []

    if use_rag:
        context, sources = get_rag_context(question)
        if len(sources) == 0:
            print("[RAG]: Không tìm thấy tài liệu liên quan.")
        else:
            chat_messages[0] = {
                "role": "system",
                "content": build_system_prompt(context)
            }

    chat_messages.append({
        "role": "user",
        "content": question
    })

    answer = stream_chat_response(chat_messages)

    if not answer:
        print("[AI]: Không có câu trả lời do lỗi kết nối tới Ollama. Lượt hội thoại này chưa được lưu.")
        return ""

    save_message(
        CONVERSATION_ID,
        "user",
        question
    )

    messages.append({
        "role": "user",
        "content": question
    })

    print_sources(sources)

    save_message(
        CONVERSATION_ID,
        "assistant",
        answer
    )

    messages.append({
        "role": "assistant",
        "content": answer
    })

    # Chạy tác vụ ghi nhớ ở Background Thread sau khi đã stream xong câu trả lời
    enqueue_memory_check(question)

    return answer


# =========================
# START CHAT
# =========================

RUN_CHAT_LOOP = __name__ == "__main__"

if RUN_CHAT_LOOP:
    print("=" * 50)
    print("LOCAL CHATGPT - QWEN3 8B")
    print("Gõ 'exit' để thoát.")
    print("=" * 50)


while RUN_CHAT_LOOP:
    try:
        question = input("\nBạn: ")
    except (EOFError, KeyboardInterrupt):
        print("\nĐang thoát...")
        wait_for_background_memory(timeout=1.0)
        break

    if not question.strip():
        continue

    if question.lower() == "exit":
        print("Đang hoàn tất tác vụ ngầm trước khi thoát...")
        wait_for_background_memory(timeout=2.0)
        break

    if len(question) > MAX_INPUT_CHARS:
        print(f"\n[Lỗi]: Câu hỏi quá dài ({len(question)} ký tự). Giới hạn tối đa là {MAX_INPUT_CHARS} ký tự. Vui lòng rút gọn câu hỏi.")
        continue

    # =========================
    # CANCEL DATA CLARIFICATION
    # =========================

    if question.lower() == "/cancel":
        if pending_data_question is not None:
            pending_data_question = None
            pending_data_filename = None
            print("\n[Data Agent] Đã hủy yêu cầu phân tích.")
        else:
            print("\n[Data Agent] Không có yêu cầu nào đang chờ.")
        continue

    # =========================
    # /rag on | /rag off | /rag status
    # =========================

    rag_command = question.strip().lower()
    if rag_command == "/rag" or rag_command.startswith("/rag "):
        print("\n" + handle_rag_command(question))
        continue

    # =========================
    # NEW DATA REQUEST
    # =========================

    if pending_data_question is not None and question.startswith("/data "):
        print("\n[Data Agent] Đã chuyển sang yêu cầu mới.")
        pending_data_question = None
        pending_data_filename = None

    # =========================
    # PENDING DATA CLARIFICATION
    # =========================

    if pending_data_question is not None:
        combined_question = pending_data_question + "\n" + question
        print("\n[Data Agent] Đang tiếp tục phân tích...")
        result, error = run_data_agent(combined_question, pending_data_filename)

        if error:
            print("\n[Data Agent] ERROR:", error)
            pending_data_question = None
            pending_data_filename = None
            continue

        if result.get("tool") == "clarification":
            pending_data_question = combined_question
            print("\n[DEBUG PENDING DATA]")
            print("Question = " + pending_data_question)
            print("Filename = " + pending_data_filename)
            print("\n===== AI ANALYSIS =====")
            print(result["explanation"])
            continue

        print("\n===== AI ANALYSIS =====")
        print(result["explanation"])
        pending_data_question = None
        pending_data_filename = None
        continue

    # =========================
    # /list_documents
    # =========================

    if question == "/list_documents":
        docs = list_documents()
        print("\n===== DOCUMENTS =====")
        if len(docs) == 0:
            print("Chưa có tài liệu.")
        else:
            for doc in docs:
                print("-", doc)
        continue

    # =========================
    # /index_documents
    # =========================

    if question == "/index_documents":
        docs = list_documents()
        print("\n[Index] Đang index tài liệu...")
        for doc in docs:
            chunk_cnt = index_document(doc)
            print(f"[Index] {doc} -> {chunk_cnt} chunks")
        print("[Index] Hoàn thành.")
        continue

    # =========================
    # /reindex
    # =========================

    if question == "/reindex":
        print("\n[Index] Đang reindex tất cả tài liệu...")
        total_chunks = reindex_all_documents()
        print("[Index] Hoàn thành.")
        print("[Index] Tổng số chunk:", total_chunks)
        continue

    # =========================
    # /delete_document
    # =========================

    if question.startswith("/delete_document "):
        doc_filename = question[len("/delete_document "):].strip()
        print("\n[Document Manager] Bạn có chắc muốn xóa", f'"{doc_filename}"? (y/n)')
        try:
            confirm = input("Xác nhận: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            confirm = "n"

        if confirm != "y":
            print("[Document Manager] Đã hủy.")
            continue

        deleted = delete_document_from_database(doc_filename)
        print(f"\n[Document Manager] Đã xóa {deleted} vector.")
        file_deleted = delete_document_file(doc_filename)
        if file_deleted:
            print("[Document Manager] Đã xóa file:", doc_filename)
        else:
            print("[Document Manager] Không tìm thấy file:", doc_filename)
        continue

    # =========================
    # /data
    # =========================

    if question.startswith("/data "):
        parts = question[len("/data "):].strip().split(" ", 1)
        if len(parts) < 2:
            print("\n[Data Agent] Cú pháp: /data <filename> <câu hỏi>")
            continue

        target_file = parts[0]
        data_q = parts[1]
        print("\n[Data Agent] Đang phân tích...")
        res_data, err_data = run_data_agent(data_q, target_file)

        if res_data is not None and res_data.get("tool") == "clarification":
            pending_data_question = data_q
            pending_data_filename = target_file
            print("\n[DEBUG PENDING DATA]")
            print("Question = " + pending_data_question)
            print("Filename = " + pending_data_filename)

        if err_data:
            print("\n[Data Agent] ERROR:", err_data)
            continue

        print("\n===== AI ANALYSIS =====")
        print(res_data["explanation"])
        continue

    # =========================
    # /analyze
    # =========================

    if question.startswith("/analyze "):
        filename = question[len("/analyze "):].strip()
        data = load_data(filename)

        if data is None:
            print("\n[Data Analysis] Không thể đọc file.")
            continue

        report = generate_report(data)
        print("\n===== DATASET ANALYSIS =====")
        print("File:", filename)
        print("Rows:", report["rows"])
        print("Columns:", report["columns"])
        print("Column names:", report["column_names"])
        print("\nNumeric columns:", report["numeric_columns"])
        print("Categorical columns:", report["categorical_columns"])
        print("\nMissing values:")
        for col, cnt in report["missing_values"].items():
            print(col, ":", cnt)

        print("\n===== STATISTICS =====")
        if report["statistics"] is None:
            print("Không có dữ liệu số.")
        else:
            print(report["statistics"])

        print("\n===== OUTLIERS =====")
        for col, r in report["outliers"].items():
            print(col, ":", r["values"])

        print("\n===== CORRELATION =====")
        if report["correlation"] is None:
            print("Không có dữ liệu số.")
        else:
            print(report["correlation"])

        analysis_prompt = f"""
Bạn là một trợ lý phân tích dữ liệu.
Hãy phân tích dataset dựa CHỈ trên kết quả do Python/Pandas cung cấp bên dưới.
Không tự bịa thêm dữ liệu. Không tự tính lại các số liệu.
Trả lời bằng tiếng Việt, rõ ràng và dễ hiểu theo 4 phần:
1. Tổng quan dữ liệu
2. Chất lượng dữ liệu
3. Thống kê và các mối quan hệ đáng chú ý
4. Nhận xét và lưu ý

===== DATASET REPORT =====
File: {filename}
Rows: {report["rows"]}
Columns: {report["columns"]}
Statistics: {report["statistics"]}
"""
        print("\n===== AI ANALYSIS =====")
        try:
            resp = ollama.chat(
                model=MODEL,
                messages=[{"role": "user", "content": analysis_prompt}],
                options={"num_ctx": NUM_CTX}
            )
            ai_analysis = resp["message"]["content"]
            print(ai_analysis)
        except Exception as e:
            print("[Data Analysis Error]: Không thể gọi Ollama:", e)
        continue

    # =========================
    # /remember
    # =========================

    if question.startswith("/remember "):
        mem_text = question[10:].strip()
        save_memory(mem_text)
        rebuild_system_prompt()
        print("\nAI: Đã ghi nhớ.")
        continue

    # =========================
    # /memory
    # =========================

    if question == "/memory":
        mems = load_memories()
        print("\n===== LONG-TERM MEMORY =====")
        if len(mems) == 0:
            print("Chưa có memory.")
        else:
            for memory_id, content in mems:
                print(f"{memory_id}. {content}")
        continue

    # =========================
    # /forget
    # =========================

    if question.startswith("/forget "):
        mid_str = question[8:].strip()
        if mid_str.isdigit():
            del_cnt = delete_memory(int(mid_str))
            rebuild_system_prompt()
            if del_cnt > 0:
                print("\nAI: Đã xóa memory.")
            else:
                print("\nAI: Không tìm thấy memory với ID này.")
        else:
            print("\nAI: Vui lòng nhập ID memory là số.")
        continue

    # =========================
    # /summarize
    # =========================

    if question == "/summarize":
        print("\n[Summary] Đang tạo summary...")
        full_messages = load_messages(CONVERSATION_ID)
        conversation_summary = summarize_conversation(full_messages)
        save_summary(CONVERSATION_ID, conversation_summary)
        reset_recent_messages()
        print("\n===== CONVERSATION SUMMARY =====")
        print(conversation_summary)
        continue

    # =========================
    # REGULAR CHAT TURN
    # =========================

    answer_question(question)

    # =========================
    # AUTO SUMMARIZATION CHECK
    # =========================

    message_count = count_messages(CONVERSATION_ID)
    checkpoint = load_summary_checkpoint(CONVERSATION_ID)
    new_messages = message_count - checkpoint

    print(f"[DEBUG] Tổng: {message_count} | Checkpoint: {checkpoint} | Message mới: {new_messages}")

    if new_messages >= SUMMARY_THRESHOLD:
        auto_summarize()