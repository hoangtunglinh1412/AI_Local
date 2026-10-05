import ollama

from memory import create_database
from memory import save_message
from memory import load_messages

from memory import save_memory
from memory import load_memories
from memory import delete_memory
from memory import update_memory

from memory import save_summary
from memory import load_summary
from memory import load_recent_messages
from memory import count_messages

from memory import save_summary_checkpoint
from memory import load_summary_checkpoint
from memory import load_messages_after

from rag import (
    retrieve_documents,
    get_document_list
)

from rag import (
    retrieve_documents,
    list_documents,
    index_document,
    delete_document_from_database,
    delete_document_file,
    reindex_all_documents
)

from data_analyzer import (
    load_data,
    generate_report
)

from data_agent import (
    run_data_agent
)

MODEL = "qwen3:8b"

# =========================
# DATA AGENT PENDING STATE
# =========================

pending_data_question = None
pending_data_filename = None

CONVERSATION_ID = "main"

SUMMARY_THRESHOLD = 20

def get_rag_context(query):

    context, sources = retrieve_documents(
        query,
        top_k=3,
        distance_threshold=18
    )

    return context, sources

# =========================
# CREATE DATABASE
# =========================

create_database()


# =========================
# LOAD LONG-TERM MEMORY
# =========================

memories = load_memories()

memory_text = ""

for memory_id, content in memories:
    memory_text += f"- {content}\n"


# =========================
# SYSTEM PROMPT
# =========================

system_prompt = f"""
Bạn là một trợ lý AI local.

Hãy trả lời bằng tiếng Việt nếu người dùng sử dụng tiếng Việt.
Trả lời rõ ràng, chính xác và dễ hiểu.

Đây là những thông tin người dùng đã yêu cầu bạn ghi nhớ:

{memory_text}

Hãy sử dụng những thông tin này khi chúng thực sự hữu ích.
Không được nói rằng bạn biết thông tin này từ hệ thống memory.
"""


messages = [
    {
        "role": "system",
        "content": system_prompt
    }
]


# =========================
# LOAD OLD CONVERSATION
# =========================

recent_messages = load_recent_messages(
    CONVERSATION_ID,
    10
)

messages.extend(recent_messages)

# =========================
# AUTOMATIC MEMORY FUNCTION
# =========================

def check_memory(question):

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
{question}
"""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": memory_prompt
            }
        ],
        options={
            "num_ctx": 8192
        }
    )

    result = response["message"]["content"].strip()

    return result

def manage_memory(new_memory, memories):

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

    for memory_id, content in memories:
        manager_prompt += f"""
ID {memory_id}: {content}
"""

    manager_prompt += """

QUY TẮC:

1. NEW

Chọn NEW nếu thông tin mới hoàn toàn khác
và không liên quan đến các memory hiện tại.

Ví dụ:

Memory hiện tại:
Người dùng đang học Data Science.

Memory mới:
Người dùng thích chụp ảnh.

→ NEW


2. UPDATE

Chọn UPDATE nếu memory mới:

- bổ sung thông tin cho memory cũ
- mở rộng memory cũ
- thay đổi memory cũ
- làm memory cũ chi tiết hơn
- chứa phần lớn nội dung của một memory cũ

ĐẶC BIỆT:

Nếu memory mới bao gồm nội dung của memory cũ
và thêm thông tin mới, PHẢI chọn UPDATE.

Ví dụ:

Memory hiện tại:
ID 7: Người dùng muốn AI trả lời ngắn gọn.

Memory mới:
Người dùng thích câu trả lời ngắn gọn nhưng phải có ví dụ.

→ UPDATE ID 7

Nội dung mới:
Người dùng muốn AI trả lời ngắn gọn nhưng phải có ví dụ.


3. IGNORE

Chọn IGNORE nếu:

- thông tin đã tồn tại
- thông tin mới không có giá trị lâu dài
- thông tin mới chỉ là cách diễn đạt lại memory cũ
- không cần thay đổi memory hiện tại


FORMAT TRẢ VỀ:

Nếu là NEW:

NEW


Nếu là IGNORE:

IGNORE


Nếu là UPDATE:

UPDATE: ID: <id>
CONTENT: <nội dung memory mới>


CHỈ trả về đúng một trong các format trên.
Không giải thích.
Không thêm nội dung khác.
"""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": manager_prompt
            }
        ],
        options={
            "num_ctx": 8192
        }
    )

    result = response["message"]["content"].strip()

    return result


def summarize_conversation(messages, old_summary=""):

    conversation_text = ""

    for message in messages:

        if message["role"] == "user":
            conversation_text += "User: " + message["content"] + "\n"

        elif message["role"] == "assistant":
            conversation_text += "Assistant: " + message["content"] + "\n"


    summary_prompt = f"""
Bạn là hệ thống tóm tắt cuộc hội thoại.

Hãy cập nhật conversation summary dựa trên
summary trước đó và các message mới.

Chỉ giữ lại:

- Chủ đề chính
- Những việc người dùng đang làm
- Những quyết định quan trọng
- Những thông tin cần thiết để tiếp tục cuộc hội thoại

Không cần giữ:

- Câu hỏi nhỏ
- Câu trả lời dài
- Nội dung lặp lại
- Thông tin không còn quan trọng

Hãy viết summary bằng tiếng Việt,
ngắn gọn và rõ ràng.

SUMMARY TRƯỚC ĐÓ:

{old_summary}


CÁC MESSAGE MỚI:

{conversation_text}
"""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": summary_prompt
            }
        ],
        options={
            "num_ctx": 8192
        }
    )

    summary = response["message"]["content"].strip()

    return summary

def auto_summarize():

    print("\n[Summary] Cuộc hội thoại đang dài. Đang tạo summary...")

    checkpoint = load_summary_checkpoint(
        CONVERSATION_ID
    )

    messages_to_summarize = load_messages_after(
        CONVERSATION_ID,
        checkpoint
    )

    if len(messages_to_summarize) == 0:

        print("[Summary] Không có message mới để summarize.")

        return

    simple_messages = []

    for message in messages_to_summarize:

        simple_messages.append({
            "role": message["role"],
            "content": message["content"]
        })

    old_summary = load_summary(
        CONVERSATION_ID
    )

    conversation_summary = summarize_conversation(
        simple_messages,
        old_summary
    )

    save_summary(
        CONVERSATION_ID,
        conversation_summary
    )

    last_message_id = messages_to_summarize[-1]["id"]

    save_summary_checkpoint(
        CONVERSATION_ID,
        last_message_id
    )

    reset_recent_messages()

    print("[DEBUG] Messages trong RAM:", len(messages))

    print("[Summary] Đã cập nhật summary.")

    print(
        "[Summary] Checkpoint:",
        last_message_id
    )

def get_rag_context(query):

    context, sources = retrieve_documents(
        query,
        top_k=2
    )

    return context, sources

def rebuild_system_prompt():

    memories = load_memories()

    memory_text = ""

    for memory_id, content in memories:
        memory_text += f"- {content}\n"


    conversation_summary = load_summary(
        CONVERSATION_ID
    )


    new_system_prompt = f"""
Bạn là một trợ lý AI local.

Hãy trả lời bằng tiếng Việt nếu người dùng sử dụng tiếng Việt.
Trả lời rõ ràng, chính xác và dễ hiểu.


LONG-TERM MEMORY:

{memory_text}


CONVERSATION SUMMARY:

{conversation_summary}


Hãy sử dụng những thông tin trên khi chúng thực sự hữu ích.

Không được nói rằng bạn biết thông tin này
từ hệ thống memory hoặc summary.
"""


    messages[0] = {
        "role": "system",
        "content": new_system_prompt
    }


def reset_recent_messages():

    recent_messages = load_recent_messages(
        CONVERSATION_ID,
        10
    )

    messages.clear()

    messages.append({
        "role": "system",
        "content": ""
    })

    messages.extend(recent_messages)

    rebuild_system_prompt()

def should_use_rag(question):

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
- Có khả năng tài liệu nội bộ cung cấp câu trả lời cụ thể hơn.

Trả về NO nếu câu hỏi:
- Chào hỏi.
- Trò chuyện thông thường.
- Hỏi về cảm xúc hoặc hội thoại.
- Yêu cầu sáng tạo như viết thơ, viết truyện.
- Rõ ràng không liên quan đến tài liệu nội bộ.

QUAN TRỌNG:

Nếu câu hỏi có thể liên quan đến tài liệu,
hãy ưu tiên YES.

Không được trả về NO chỉ vì bạn đã biết câu trả lời
từ kiến thức chung.

Ví dụ:

"Machine Learning là gì?" -> YES
"Mean là gì?" -> YES
"Median là gì?" -> YES
"Variance là gì?" -> YES
"Outlier là gì?" -> YES
"Giải thích Supervised Learning" -> YES
"Xin chào" -> NO
"Bạn khỏe không?" -> NO
"Viết cho tôi một bài thơ" -> NO

Câu hỏi của người dùng:

{question}

Chỉ trả lời YES hoặc NO.
"""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": router_prompt
            }
        ],
        options={
            "num_ctx": 4096
        }
    )

    result = response["message"]["content"].strip().upper()

    if "YES" in result:
        return True

    return False


# =========================
# START CHAT
# =========================

print("=" * 50)
print("LOCAL CHATGPT - QWEN3 8B")
print("Gõ 'exit' để thoát.")
print("=" * 50)


while True:

    question = input("\nBạn: ")

    if question.lower() == "exit":
        break

    # =========================
    # CANCEL DATA CLARIFICATION
    # =========================

    if question.lower() == "/cancel":

        if pending_data_question is not None:

            pending_data_question = None
            pending_data_filename = None

            print(
                "\n[Data Agent] "
                "Đã hủy yêu cầu phân tích."
            )

        else:

            print(
                "\n[Data Agent] "
                "Không có yêu cầu nào đang chờ."
            )

        continue
    # =========================
    # NEW DATA REQUEST
    # =========================

    if (
        pending_data_question is not None
        and question.startswith("/data ")
    ):

        print(
            "\n[Data Agent] "
            "Đã chuyển sang yêu cầu mới."
        )

        pending_data_question = None
        pending_data_filename = None

    # =========================
    # PENDING DATA CLARIFICATION
    # =========================

    if pending_data_question is not None:

        combined_question = (
            pending_data_question
            + "\n"
            + question
        )

        print(
            "\n[Data Agent] "
            "Đang tiếp tục phân tích..."
        )

        result, error = run_data_agent(
            combined_question,
            pending_data_filename
        )

        if error:

            print(
                "\n[Data Agent] ERROR:",
                error
            )

            pending_data_question = None
            pending_data_filename = None

            continue

        # =========================
        # TIẾP TỤC CLARIFICATION
        # =========================

        if result.get("tool") == "clarification":

            pending_data_question = combined_question

            print(
                "\n[DEBUG PENDING DATA]"
            )

            print(
                "Question = "
                + pending_data_question
            )

            print(
                "Filename = "
                + pending_data_filename
            )

            print(
                "\n===== AI ANALYSIS ====="
            )

            print(
                result["explanation"]
            )

            continue

        # =========================
        # ĐÃ CÓ KẾT QUẢ
        # =========================

        print(
            "\n===== AI ANALYSIS ====="
        )

        print(
            result["explanation"]
        )

        # Xóa trạng thái pending
        pending_data_question = None
        pending_data_filename = None

        continue
    # =========================
    # /list_documents
    # =========================

    if question == "/list_documents":

        documents = list_documents()

        print("\n===== DOCUMENTS =====")

        if len(documents) == 0:

            print("Chưa có tài liệu.")

        else:

            for document in documents:

                print(
                    "-",
                    document
                )

        continue
    # =========================
    # /index_documents
    # =========================

    if question == "/index_documents":

        documents = list_documents()

        print(
            "\n[Index] Đang index tài liệu..."
        )

        for document in documents:

            chunk_count = index_document(
                document
            )

            print(
                "[Index]",
                document,
                "->",
                chunk_count,
                "chunks"
            )

        print(
            "[Index] Hoàn thành."
        )

        continue

    # =========================
    # /reindex
    # =========================

    if question == "/reindex":

        print(
            "\n[Index] Đang reindex tất cả tài liệu..."
        )

        total_chunks = reindex_all_documents()

        print(
            "[Index] Hoàn thành."
        )

        print(
            "[Index] Tổng số chunk:",
            total_chunks
        )

        continue

    # =========================
    # /delete_document
    # =========================

    if question.startswith(
        "/delete_document "
    ):

        filename = question[
            len("/delete_document "):
        ].strip()

        print(
            "\n[Document Manager] Bạn có chắc muốn xóa",
            f'"{filename}"?',
            "(y/n)"
        )

        confirm = input("Xác nhận: ").strip().lower()

        if confirm != "y":

            print(
                "[Document Manager] Đã hủy."
            )

            continue

        # Xóa vector trong ChromaDB
        deleted = delete_document_from_database(
            filename
        )

        print(
            "\n[Document Manager] Đã xóa",
            deleted,
            "vector."
        )

        # Xóa file vật lý
        file_deleted = delete_document_file(
            filename
        )

        if file_deleted:

            print(
                "[Document Manager] Đã xóa file:",
                filename
            )

        else:

            print(
                "[Document Manager] Không tìm thấy file:",
                filename
            )

        continue

    # =========================
    # /data
    # =========================

    if question.startswith("/data "):

        parts = question[
            len("/data "):
        ].strip().split(
            " ",
            1
        )


        if len(parts) < 2:

            print(
                "\n[Data Agent] "
                "Cú pháp:"
            )

            print(
                "/data <filename> <câu hỏi>"
            )

            continue


        filename = parts[0]

        data_question = parts[1]


        print(
            "\n[Data Agent] "
            "Đang phân tích..."
        )


        result, error = run_data_agent(
            data_question,
            filename
        )


        if result is not None:

            if result.get("tool") == "clarification":

                pending_data_question = data_question
                pending_data_filename = filename

                print(
                    "\n[DEBUG PENDING DATA]"
                )

                print(
                    "Question = "
                    + pending_data_question
                )

                print(
                    "Filename = "
                    + pending_data_filename
                )

        if error:

            print(
                "\n[Data Agent] ERROR:",
                error
            )

            continue


        print(
            "\n===== AI ANALYSIS ====="
        )

        print(
            result["explanation"]
        )


        continue

    # =========================
    # /analyze
    # =========================

    if question.startswith("/analyze "):

        filename = question[
            len("/analyze "):
        ].strip()

        data = load_data(
            filename
        )

        if data is None:

            print(
                "\n[Data Analysis] "
                "Không thể đọc file."
            )

            continue

        report = generate_report(
            data
        )

        print(
            "\n===== DATASET ANALYSIS ====="
        )

        print(
            "File:",
            filename
        )

        print(
            "Rows:",
            report["rows"]
        )

        print(
            "Columns:",
            report["columns"]
        )

        print(
            "Column names:",
            report["column_names"]
        )

        print(
            "\nNumeric columns:",
            report["numeric_columns"]
        )

        print(
            "Categorical columns:",
            report["categorical_columns"]
        )

        print(
            "\nMissing values:"
        )

        for column, count in report[
            "missing_values"
        ].items():

            print(
                column,
                ":",
                count
            )

        print(
            "\n===== STATISTICS ====="
        )

        if report["statistics"] is None:

            print(
                "Không có dữ liệu số."
            )

        else:

            print(
                report["statistics"]
            )

        print(
            "\n===== OUTLIERS ====="
        )

        for column, result in report[
            "outliers"
        ].items():

            print(
                column,
                ":",
                result["values"]
            )

        print(
            "\n===== CORRELATION ====="
        )

        if report["correlation"] is None:

            print(
                "Không có dữ liệu số."
            )

        else:

            print(
                report["correlation"]
            )

        # =========================
        # AI ANALYSIS
        # =========================

        analysis_prompt = f"""
    Bạn là một trợ lý phân tích dữ liệu.

    Hãy phân tích dataset dựa CHỈ trên kết quả
    do Python/Pandas cung cấp bên dưới.

    Không được tự bịa thêm dữ liệu.

    Không được tự tính lại hoặc thay đổi
    các số liệu đã được cung cấp.

    Hãy trả lời bằng tiếng Việt, rõ ràng và
    dễ hiểu.

    Hãy trình bày theo 4 phần:

    1. Tổng quan dữ liệu
    2. Chất lượng dữ liệu
    3. Thống kê và các mối quan hệ đáng chú ý
    4. Nhận xét và lưu ý

    Ở phần nhận xét, cần phân biệt rõ:
    - kết quả quan sát được
    - những điều chưa thể kết luận

    Đặc biệt, tương quan không đồng nghĩa với
    quan hệ nhân quả.

    ===== DATASET REPORT =====

    File:
    {filename}

    Rows:
    {report["rows"]}

    Columns:
    {report["columns"]}

    Column names:
    {report["column_names"]}

    Numeric columns:
    {report["numeric_columns"]}

    Categorical columns:
    {report["categorical_columns"]}

    Missing values:
    {report["missing_values"]}

    Statistics:
    {report["statistics"]}

    Outliers:
    {report["outliers"]}

    Correlation:
    {report["correlation"]}
    """

        print(
            "\n===== AI ANALYSIS ====="
        )

        response = ollama.chat(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": analysis_prompt
                }
            ],
            options={
                "num_ctx": 8192
            }
        )

        ai_analysis = response[
            "message"
        ][
            "content"
        ]

        print(
            ai_analysis
        )

        continue
        # =========================
        # /remember
        # =========================

        if question.startswith("/remember "):

            memory = question[10:]

            save_memory(memory)

            rebuild_system_prompt()

            print("\nAI: Đã ghi nhớ.")

            continue


    # =========================
    # /memory
    # =========================

    if question == "/memory":

        memories = load_memories()

        print("\n===== LONG-TERM MEMORY =====")

        if len(memories) == 0:

            print("Chưa có memory.")

        else:

            for memory_id, content in memories:

                print(f"{memory_id}. {content}")

        continue


    # =========================
    # /forget
    # =========================

    if question.startswith("/forget "):

        memory_id = question[8:]

        if memory_id.isdigit():

            delete_memory(int(memory_id))

            rebuild_system_prompt()

            print("\nAI: Đã xóa memory.")

        else:

            print("\nAI: Vui lòng nhập ID memory.")

        continue

# =========================
# /summarize
# =========================

    if question == "/summarize":

        print("\n[Summary] Đang tạo summary...")

        # Lấy toàn bộ lịch sử từ database
        full_messages = load_messages(
            CONVERSATION_ID
        )

        # Tạo summary
        conversation_summary = summarize_conversation(
            full_messages
        )

        # Lưu summary
        save_summary(
            CONVERSATION_ID,
            conversation_summary
        )

        # Cập nhật system prompt
        reset_recent_messages()

        print("\n===== CONVERSATION SUMMARY =====")
        print(conversation_summary)

        continue


    # =========================
    # AUTOMATIC MEMORY
    # =========================

    memory_result = check_memory(question)

    print("\n[Memory]:", memory_result)


    if memory_result.startswith("SAVE:"):

        memory = memory_result[5:].strip()

        memories = load_memories()

        manager_result = manage_memory(
            memory,
            memories
        )

        print("[Memory Manager]:", manager_result)


        # =========================
        # NEW MEMORY
        # =========================

        if manager_result == "NEW":

            save_memory(memory)

            print("[Memory Manager]: Đã tạo memory mới.")

            rebuild_system_prompt()


        # =========================
        # UPDATE MEMORY
        # =========================

        elif manager_result.startswith("UPDATE:"):

            lines = manager_result.splitlines()

            memory_id = None
            new_content = None

            for line in lines:

                if line.startswith("UPDATE: ID:"):

                    memory_id = line.replace(
                        "UPDATE: ID:",
                        ""
                    ).strip()


                if line.startswith("CONTENT:"):

                    new_content = line.replace(
                        "CONTENT:",
                        ""
                    ).strip()


            if memory_id is not None and new_content is not None:

                if memory_id.isdigit():

                    update_memory(
                        int(memory_id),
                        new_content
                    )

                    print("[Memory Manager]: Đã cập nhật memory.")

                    rebuild_system_prompt()

                else:

                    print("[Memory Manager]: ID memory không hợp lệ.")


        # =========================
        # IGNORE MEMORY
        # =========================

        elif manager_result == "IGNORE":

            print("[Memory Manager]: Bỏ qua memory.")


    # =========================
    # RAG ROUTER
    # =========================

    use_rag = should_use_rag(
        question
    )

    print(
        "[RAG Router]:",
        "YES" if use_rag else "NO"
    )


    # =========================
    # CHAT MESSAGE
    # =========================

    chat_messages = messages.copy()


    # =========================
    # USE RAG
    # =========================

    sources = []

    if use_rag:

        context, sources = get_rag_context(
            question
        )

        if len(sources) == 0:

            print(
                "[RAG]: Không tìm thấy tài liệu liên quan."
            )

        rag_prompt = f"""
    Hãy trả lời câu hỏi của người dùng.

    Nếu CONTEXT có thông tin liên quan,
    hãy ưu tiên sử dụng thông tin trong CONTEXT.

    Nếu CONTEXT không có đủ thông tin,
    hãy trả lời dựa trên kiến thức thông thường của bạn.

    Không được tự ý bịa thông tin từ tài liệu.

    CONTEXT:

    {context}

    CÂU HỎI:

    {question}
    """

        chat_messages.append({
            "role": "user",
            "content": rag_prompt
        })


    # =========================
    # NORMAL CHAT
    # =========================

    else:

        chat_messages.append({
            "role": "user",
            "content": question
        })


    # =========================
    # CALL QWEN
    # =========================

    response = ollama.chat(
        model=MODEL,
        messages=chat_messages,
        options={
            "num_ctx": 8192
        }
    )

    answer = response["message"]["content"]


    # =========================
    # SAVE USER QUESTION
    # =========================

    save_message(
        CONVERSATION_ID,
        "user",
        question
    )

    messages.append({
        "role": "user",
        "content": question
    })


    # =========================
    # SHOW AI RESPONSE
    # =========================

    print("\nAI:", answer)


    # =========================
    # SHOW SOURCES
    # =========================

    print("\n===== SOURCES =====")

    if len(sources) == 0:

        print("Không sử dụng tài liệu.")

    else:

        for source in sources:

            print(
                "File:",
                source["filename"]
            )

            print(
                "Section:",
                source["section"]
            )

            print(
                "Chunk:",
                source["chunk"]
            )

            print(
                "Distance:",
                source["distance"]
            )

            print()


    # =========================
    # SAVE AI RESPONSE
    # =========================

    save_message(
        CONVERSATION_ID,
        "assistant",
        answer
    )

    messages.append({
        "role": "assistant",
        "content": answer
    })

    # =========================
    # AUTO SUMMARIZATION
    # =========================

    message_count = count_messages(
        CONVERSATION_ID
    )

    checkpoint = load_summary_checkpoint(
        CONVERSATION_ID
    )

    new_messages = message_count - checkpoint

    print(
        "[DEBUG] Tổng:",
        message_count,
        "| Checkpoint:",
        checkpoint,
        "| Message mới:",
        new_messages
    )

    if new_messages >= SUMMARY_THRESHOLD:

        auto_summarize()