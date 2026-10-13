"""
Streamlit Web UI cho Local AI Assistant.
Tích hợp:
- Khung Chat trung tâm với Streaming và lịch sử hội thoại
- Sidebar điều khiển: Công tắc RAG, nạp file PDF/DOCX/TXT trực tiếp, quản lý Long-term Memory
- Reset hội thoại và nạp lại 10 tin nhắn gần nhất
- Trích dẫn nguồn tài liệu tham khảo từ RAG
"""

import os
import io
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
    CONVERSATION_ID,
    should_use_rag,
    build_system_prompt
)


# =========================
# HELPER FUNCTIONS
# =========================

def save_uploaded_document(uploaded_file, dest_folder=DOCUMENT_FOLDER):
    """
    Lưu file từ uploader vào dest_folder và gọi index_document(filename) từ rag.py.
    Trả về số chunk đã được index.
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


def get_rag_context_and_sources(query, top_k=2, distance_threshold=18.0):
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


def stream_ollama_response(chat_messages, model=MODEL):
    """
    Generator streaming câu trả lời từ Ollama theo thời gian thực.
    """
    response_stream = ollama.chat(
        model=model,
        messages=chat_messages,
        options={
            "num_ctx": 8192
        },
        stream=True
    )

    for chunk in response_stream:
        token = chunk.get("message", {}).get("content", "")
        if token:
            yield token


def render_sources_expander(sources):
    """
    Hiển thị các trích dẫn tài liệu tham khảo bên dưới câu trả lời của AI.
    """
    if not sources:
        return

    with st.expander("📚 Nguồn tài liệu tham khảo"):
        for i, src in enumerate(sources, 1):
            sec_info = f" | Section: `{src['section']}`" if src.get("section") else ""
            dist_info = f" | Distance: `{src['distance']}`" if src.get("distance") is not None else ""
            st.markdown(f"**[{i}] 📄 `{src['filename']}`** (Chunk #{src['chunk']}{sec_info}{dist_info})")
            if src.get("text"):
                st.caption(src["text"])


# =========================
# MAIN STREAMLIT APP
# =========================

def main():
    st.set_page_config(
        page_title="Local AI Assistant",
        page_icon="🤖",
        layout="wide"
    )

    # Khởi tạo cơ sở dữ liệu SQLite
    create_database()

    # Khởi tạo Session State
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
    # SIDEBAR ĐIỀU KHIỂN
    # =========================
    with st.sidebar:
        st.title("⚙️ Điều Khiển")

        # Nút "Đoạn chat mới"
        if st.button("➕ Đoạn chat mới", use_container_width=True, help="Xóa tin nhắn tạm trong session state và nạp lại 10 tin nhắn gần nhất"):
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
        rag_enabled = st.toggle("Bật tìm kiếm tài liệu (RAG)", value=True)

        # Nạp tài liệu trực tiếp
        st.subheader("📁 Nạp tài liệu")
        uploaded_file = st.file_uploader(
            "Tải lên PDF, DOCX, TXT",
            type=["pdf", "docx", "txt"],
            help="Tự động lưu vào documents/ và nạp vào ChromaDB"
        )

        if uploaded_file is not None:
            file_key = f"{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.last_indexed_key != file_key:
                with st.spinner(f"Đang xử lý '{uploaded_file.name}'..."):
                    chunk_count = save_uploaded_document(uploaded_file)
                st.session_state.last_indexed_key = file_key
                st.success(f"Đã nạp '{uploaded_file.name}' vào ChromaDB ({chunk_count} chunks)!")

        # Danh sách tài liệu hiện có
        existing_docs = list_documents()
        with st.expander(f"Tài liệu hiện có ({len(existing_docs)})", expanded=False):
            if existing_docs:
                for doc in existing_docs:
                    st.markdown(f"- 📄 `{doc}`")
            else:
                st.caption("Chưa có tài liệu nào trong documents/.")

        st.divider()

        # Quản lý Trí nhớ dài hạn
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

        # Thêm nhanh trí nhớ
        with st.form("add_memory_form", clear_on_submit=True):
            new_mem_text = st.text_input("Thêm trí nhớ mới:", placeholder="Ví dụ: Người dùng thích câu trả lời ngắn gọn...")
            submit_mem = st.form_submit_button("Lưu trí nhớ", use_container_width=True)
            if submit_mem:
                if new_mem_text.strip():
                    save_memory(new_mem_text.strip())
                    st.rerun()
                else:
                    st.warning("Vui lòng nhập nội dung.")

    # =========================
    # KHUNG CHAT TRUNG TÂM
    # =========================
    st.header("🤖 Local AI Assistant")
    st.caption(f"Mô hình: `{MODEL}` | RAG: `{'BẬT' if rag_enabled else 'TẮT'}`")

    # Hiển thị lịch sử hội thoại
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources_expander(msg["sources"])

    # Nhận câu hỏi từ người dùng
    if user_prompt := st.chat_input("Nhập câu hỏi của bạn..."):
        # Hiển thị tin nhắn người dùng ngay lập tức
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Lưu tin nhắn người dùng
        st.session_state.messages.append({
            "role": "user",
            "content": user_prompt
        })
        save_message(CONVERSATION_ID, "user", user_prompt)

        # Xử lý RAG nếu được bật
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

        # Xây dựng system prompt với Long-term memory, Summary và RAG Context
        system_prompt = build_system_prompt(rag_context)

        # Lấy 10 tin nhắn gần nhất làm context trò chuyện
        chat_messages = [{"role": "system", "content": system_prompt}]
        for m in st.session_state.messages[-10:]:
            chat_messages.append({
                "role": m["role"],
                "content": m["content"]
            })

        # Hiển thị câu trả lời dạng Streaming
        with st.chat_message("assistant"):
            try:
                full_answer = st.write_stream(stream_ollama_response(chat_messages))
            except Exception as e:
                st.error(f"Lỗi khi gọi mô hình Ollama ({MODEL}): {e}")
                full_answer = "Xin lỗi, không thể kết nối tới mô hình AI lúc này."

            # Hiển thị trích dẫn RAG
            if sources:
                render_sources_expander(sources)

        # Lưu câu trả lời của AI vào SQLite và session state
        save_message(CONVERSATION_ID, "assistant", full_answer)
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_answer,
            "sources": sources
        })


if __name__ == "__main__":
    main()

