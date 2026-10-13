"""
Streamlit Web UI cho Local AI Assistant.
Tích hợp:
- Khung Chat trung tâm với Streaming theo thời gian thực và lịch sử hội thoại
- Sidebar điều khiển: Công tắc RAG, nạp file PDF/DOCX/TXT trực tiếp, quản lý Long-term Memory
- Tạo đoạn chat mới (New Chat) và nạp lại 10 tin nhắn gần nhất
- Trích dẫn nguồn tài liệu tham khảo từ RAG
- Đồng bộ kiến trúc tối ưu (NUM_CTX = 8192, guardrails 16.000 ký tự)
"""

import os
import io
from typing import List, Dict, Any, Tuple

import streamlit as st
import ollama

from memory import (
    create_database,
    save_message,
    load_recent_messages,
    save_memory,
    load_memories,
    delete_memory,
    load_summary
)

from rag import (
    DOCUMENT_FOLDER,
    index_document,
    search_documents,
    list_documents
)

from chatbot import (
    MODEL,
    NUM_CTX,
    MAX_INPUT_CHARS,
    CONVERSATION_ID,
    should_use_rag,
    build_system_prompt
)


# =========================
# HELPER FUNCTIONS
# =========================

def save_uploaded_document(uploaded_file, dest_folder: str = DOCUMENT_FOLDER) -> int:
    """
    Lưu file từ uploader vào dest_folder và gọi index_document(filename) từ rag.py.
    Trả về số chunk đã được index vào ChromaDB.
    """
    os.makedirs(dest_folder, exist_ok=True)
    file_path = os.path.join(dest_folder, uploaded_file.name)

    if hasattr(uploaded_file, "getbuffer"):
        content = uploaded_file.getbuffer()
        with open(file_path, "wb") as f:
            f.write(content)
    elif hasattr(uploaded_file, "read"):
        content = uploaded_file.read()
        if isinstance(content, str):
            content = content.encode("utf-8")
        with open(file_path, "wb") as f:
            f.write(content)
    else:
        with open(file_path, "wb") as f:
            f.write(bytes(uploaded_file))

    chunk_count = index_document(uploaded_file.name)
    return chunk_count


def get_rag_context_and_sources(
    query: str,
    top_k: int = 2,
    distance_threshold: float = 18.0
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Tìm kiếm tài liệu liên quan trong ChromaDB và lọc theo distance_threshold.
    Trả về: (context_text, sources)
    """
    try:
        results = search_documents(query, top_k=top_k)
    except Exception as error:
        return "", []

    if not results or not results.get("documents"):
        return "", []

    documents = results["documents"][0] if results["documents"] else []
    metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
    distances = results.get("distances", [[]])[0] if results.get("distances") else []

    context_parts = []
    sources = []

    for i, document in enumerate(documents):
        if not document or not str(document).strip():
            continue

        distance = distances[i] if i < len(distances) else None
        if distance is not None and distance > distance_threshold:
            continue

        metadata = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
        source = {
            "filename": metadata.get("filename", "unknown"),
            "chunk": metadata.get("chunk", i),
            "section": metadata.get("section", ""),
            "distance": round(distance, 4) if distance is not None else None,
            "text": str(document).strip()
        }
        sources.append(source)

        header = f"[{len(sources)}] {source['filename']}"
        if source["section"]:
            header += f" | Section: {source['section']}"
        context_parts.append(header + "\n" + str(document).strip())

    return "\n\n".join(context_parts), sources


def stream_ollama_response(chat_messages: List[Dict[str, str]], model: str = MODEL):
    """
    Generator streaming câu trả lời từ Ollama theo thời gian thực (options: num_ctx=8192).
    """
    response_stream = ollama.chat(
        model=model,
        messages=chat_messages,
        options={
            "num_ctx": NUM_CTX
        },
        stream=True
    )

    for chunk in response_stream:
        token = chunk.get("message", {}).get("content", "")
        if token:
            yield token


def render_sources_expander(sources: List[Dict[str, Any]], title: str = "📚 Nguồn tài liệu tham khảo"):
    """
    Hiển thị các trích dẫn tài liệu tham khảo bên dưới câu trả lời của AI.
    """
    if not sources:
        return

    with st.expander(title):
        for i, src in enumerate(sources, 1):
            sec_info = f" | Section: `{src['section']}`" if src.get("section") else ""
            dist_info = f" | Distance: `{src['distance']}`" if src.get("distance") is not None else ""
            st.markdown(f"**[{i}] 📄 `{src['filename']}`** (Chunk #{src['chunk']}{sec_info}{dist_info})")
            if src.get("text"):
                st.caption(src["text"])


# =========================
# MAIN STREAMLIT APPLICATION
# =========================

def main():
    st.set_page_config(
        page_title="Local AI Assistant",
        page_icon="🤖",
        layout="wide"
    )

    # Khởi tạo cơ sở dữ liệu SQLite
    create_database()

    # Khởi tạo Session State cho lịch sử hội thoại
    if "messages" not in st.session_state:
        st.session_state.messages = []
        recent_messages = load_recent_messages(CONVERSATION_ID, limit=10)
        for msg in recent_messages:
            st.session_state.messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

    if "last_indexed_key" not in st.session_state:
        st.session_state.last_indexed_key = None

    # =========================
    # THANH ĐIỀU KHIỂN (SIDEBAR)
    # =========================
    with st.sidebar:
        st.title("⚙️ Điều Khiển")

        # Nút "Tạo đoạn chat mới"
        if st.button("➕ Tạo đoạn chat mới", use_container_width=True, help="Xóa tin nhắn tạm trong session state và nạp lại 10 tin nhắn gần nhất"):
            st.session_state.messages = []
            recent_messages = load_recent_messages(CONVERSATION_ID, limit=10)
            for msg in recent_messages:
                st.session_state.messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            st.rerun()

        st.divider()

        # Công tắc bật/tắt RAG
        st.subheader("🔍 Cài đặt RAG")
        rag_enabled = st.toggle("Kích hoạt RAG", value=True)

        # Nạp tài liệu tự động vào ChromaDB
        st.subheader("📁 Nạp tài liệu tự động")
        uploaded_file = st.file_uploader(
            "Tải lên file .txt, .pdf, .docx",
            type=["pdf", "docx", "txt"],
            help="Tự động lưu vào documents/ và gọi index_document() để chia chunk nạp vào ChromaDB"
        )

        if uploaded_file is not None:
            file_key = f"{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.last_indexed_key != file_key:
                with st.spinner(f"Đang xử lý và index '{uploaded_file.name}'..."):
                    chunk_count = save_uploaded_document(uploaded_file)
                st.session_state.last_indexed_key = file_key
                st.success(f"Đã nạp '{uploaded_file.name}' vào ChromaDB thành công ({chunk_count} chunks)!")

        # Danh sách tài liệu hiện có trong documents/
        existing_docs = list_documents()
        with st.expander(f"Tài liệu đã index ({len(existing_docs)})", expanded=False):
            if existing_docs:
                for doc in existing_docs:
                    st.markdown(f"- 📄 `{doc}`")
            else:
                st.caption("Chưa có tài liệu nào trong thư mục documents/.")

        st.divider()

        # Quản lý Trí nhớ dài hạn (Long-term Memory)
        st.subheader("🧠 Trí nhớ dài hạn")
        memories = load_memories()

        with st.expander(f"Danh sách ghi nhớ ({len(memories)})", expanded=False):
            if not memories:
                st.caption("Chưa có ghi nhớ nào.")
            else:
                for mem_id, mem_content in memories:
                    col_txt, col_del = st.columns([0.8, 0.2])
                    with col_txt:
                        st.markdown(f"• {mem_content}")
                    with col_del:
                        if st.button("🗑️", key=f"del_mem_{mem_id}", help="Xóa ghi nhớ này"):
                            delete_memory(mem_id)
                            st.rerun()

        # Form thêm nhanh trí nhớ thủ công
        with st.form("add_memory_form", clear_on_submit=True):
            new_mem_text = st.text_input("Thêm trí nhớ mới:", placeholder="Ví dụ: Người dùng thích câu trả lời ngắn gọn...")
            submit_mem = st.form_submit_button("Lưu trí nhớ", use_container_width=True)
            if submit_mem:
                if new_mem_text.strip():
                    save_memory(new_mem_text.strip())
                    st.rerun()
                else:
                    st.warning("Vui lòng nhập nội dung ghi nhớ.")

    # =========================
    # GIAO DIỆN CHAT TRUNG TÂM
    # =========================
    st.header("🤖 Local AI Assistant")
    st.caption(f"Mô hình: `{MODEL}` | Context: `{NUM_CTX}` tokens | RAG: `{'BẬT' if rag_enabled else 'TẮT'}`")

    # Hiển thị toàn bộ lịch sử hội thoại
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources_expander(msg["sources"])

    # Nhận input từ người dùng qua st.chat_input
    if user_prompt := st.chat_input("Nhập câu hỏi của bạn..."):
        # Guardrail: Kiểm tra độ dài đầu vào
        if len(user_prompt) > MAX_INPUT_CHARS:
            st.error(f"⚠️ Câu hỏi quá dài ({len(user_prompt):,} ký tự). Giới hạn tối đa là {MAX_INPUT_CHARS:,} ký tự. Vui lòng rút gọn câu hỏi.")
        else:
            # Hiển thị tin nhắn người dùng ngay lập tức
            with st.chat_message("user"):
                st.markdown(user_prompt)

            # Lưu tin nhắn người dùng vào session state và SQLite
            st.session_state.messages.append({
                "role": "user",
                "content": user_prompt
            })
            save_message(CONVERSATION_ID, "user", user_prompt)

            # Xử lý RAG nếu được kích hoạt
            rag_context = ""
            sources = []

            if rag_enabled:
                use_rag = False
                try:
                    use_rag = should_use_rag(user_prompt)
                except Exception:
                    use_rag = True

                if use_rag:
                    rag_context, sources = get_rag_context_and_sources(user_prompt)

            # Xây dựng system prompt hoàn chỉnh (Long-term Memory + Summary + RAG Context)
            system_prompt = build_system_prompt(rag_context)

            # Lấy 10 tin nhắn gần nhất làm context hội thoại
            chat_messages = [{"role": "system", "content": system_prompt}]
            for m in st.session_state.messages[-10:]:
                chat_messages.append({
                    "role": m["role"],
                    "content": m["content"]
                })

            # Hiển thị câu trả lời dạng Streaming thời gian thực
            with st.chat_message("assistant"):
                try:
                    full_answer = st.write_stream(stream_ollama_response(chat_messages))
                except Exception as e:
                    st.error(f"Lỗi khi kết nối tới mô hình Ollama ({MODEL}): {e}")
                    full_answer = "Xin lỗi, không thể kết nối tới mô hình AI lúc này. Vui lòng kiểm tra dịch vụ Ollama."

                # Hiển thị trích dẫn nguồn tài liệu tham khảo nếu có
                if sources:
                    render_sources_expander(sources)

            # Lưu câu trả lời của AI vào SQLite và session state sau khi stream xong
            save_message(CONVERSATION_ID, "assistant", full_answer)
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "sources": sources
            })


if __name__ == "__main__":
    main()
