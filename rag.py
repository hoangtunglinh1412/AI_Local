import os
import re
import threading
from typing import List, Dict, Any, Tuple, Optional

from pypdf import PdfReader
from docx import Document

MODEL = "qwen3:8b"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

DOCUMENT_FOLDER = "documents"

CHROMA_DB_PATH = "chroma_db"

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}

# =========================
# LAZY LOADING SINGLETONS
# =========================

_model_lock = threading.RLock()
_chroma_lock = threading.RLock()
_embedding_model = None
_chroma_client = None
_collection = None


def get_embedding_model():
    """
    Lazy load SentenceTransformer:
    - Chỉ import và tải weights khi thực sự có tác vụ embedding/search.
    - Mặc định chạy trên CPU (tiết kiệm VRAM cho qwen3:8b trên RTX 3060 Ti 8GB).
    - Có thể cấu hình qua biến môi trường EMBEDDING_DEVICE=cuda.
    """
    global _embedding_model
    if "embedding_model" in globals() and globals()["embedding_model"] is not None:
        return globals()["embedding_model"]

    if _embedding_model is None:
        with _model_lock:
            if _embedding_model is None:
                from sentence_transformers import SentenceTransformer
                device = os.environ.get("EMBEDDING_DEVICE", "cpu")
                _embedding_model = SentenceTransformer(
                    EMBEDDING_MODEL,
                    device=device
                )
    return _embedding_model


def get_chroma_client():
    """Lazy load PersistentClient của ChromaDB với tắt telemetry."""
    global _chroma_client
    if "chroma_client" in globals() and globals()["chroma_client"] is not None:
        return globals()["chroma_client"]

    if _chroma_client is None:
        with _chroma_lock:
            if _chroma_client is None:
                import chromadb
                try:
                    import chromadb.config
                    settings = chromadb.config.Settings(anonymized_telemetry=False)
                    _chroma_client = chromadb.PersistentClient(
                        path=CHROMA_DB_PATH,
                        settings=settings
                    )
                except Exception:
                    _chroma_client = chromadb.PersistentClient(
                        path=CHROMA_DB_PATH
                    )
    return _chroma_client


def get_collection():
    """Lazy load collection 'documents' trong ChromaDB."""
    global _collection
    if "collection" in globals() and globals()["collection"] is not None:
        return globals()["collection"]

    if _collection is None:
        with _chroma_lock:
            if _collection is None:
                client = get_chroma_client()
                _collection = client.get_or_create_collection(
                    name="documents"
                )
    return _collection


def _get_active_embedding_model():
    if "embedding_model" in globals() and globals()["embedding_model"] is not None:
        return globals()["embedding_model"]
    return get_embedding_model()


def _get_active_collection():
    if "collection" in globals() and globals()["collection"] is not None:
        return globals()["collection"]
    return get_collection()


def __getattr__(name: str):
    """Hỗ trợ tương thích ngược cho 'from rag import collection, embedding_model'."""
    if name == "embedding_model":
        return get_embedding_model()
    if name == "collection":
        return get_collection()
    if name == "chroma_client":
        return get_chroma_client()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def has_indexed_documents() -> bool:
    """Kiểm tra nhanh xem collection trong ChromaDB đã có dữ liệu hay chưa."""
    try:
        coll = _get_active_collection()
        return coll.count() > 0
    except Exception:
        return False


# =========================
# EMBEDDING & DATABASE OPS
# =========================

def create_embeddings(chunks):
    if not chunks:
        import numpy as np
        return np.empty((0, 384), dtype=np.float32)

    texts = [chunk["text"] for chunk in chunks]
    model = _get_active_embedding_model()
    embeddings = model.encode(texts)
    return embeddings


def save_to_vector_database(chunks, embeddings, filename):
    if not chunks or len(chunks) == 0:
        return

    coll = _get_active_collection()
    ids = []
    metadatas = []
    documents = []

    for i in range(len(chunks)):
        ids.append(f"{filename}_{i}")
        metadatas.append({
            "filename": filename,
            "chunk": i,
            "section": chunks[i].get("section", "")
        })
        documents.append(chunks[i]["text"])

    emb_list = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings

    coll.upsert(
        ids=ids,
        embeddings=emb_list,
        documents=documents,
        metadatas=metadatas
    )


def read_txt(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, "r", encoding="latin-1") as file:
                return file.read()
        except Exception:
            return ""
    except Exception:
        return ""


def read_pdf(file_path: str) -> str:
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as error:
        print(f"[RAG]: Lỗi khi đọc file PDF '{file_path}':", error)
        return ""


def read_docx(file_path: str) -> str:
    try:
        document = Document(file_path)
        text = ""
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as error:
        print(f"[RAG]: Lỗi khi đọc file DOCX '{file_path}':", error)
        return ""


def read_document(file_path: str) -> str:
    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".txt":
        return read_txt(file_path)
    elif extension == ".pdf":
        return read_pdf(file_path)
    elif extension == ".docx":
        return read_docx(file_path)
    else:
        return ""


def load_documents():
    documents = []
    if not os.path.exists(DOCUMENT_FOLDER):
        return documents

    for filename in os.listdir(DOCUMENT_FOLDER):
        file_path = os.path.join(DOCUMENT_FOLDER, filename)
        if os.path.isfile(file_path):
            text = read_document(file_path)
            if text and text.strip():
                documents.append({
                    "filename": filename,
                    "text": text
                })
    return documents


# =========================
# RECURSIVE CHARACTER SPLITTING
# =========================

SPLIT_LEVELS = [
    (re.compile(r"\n[ \t]*\n\s*"), "\n\n"),
    (re.compile(r"\n"), "\n"),
    (re.compile(r"(?<=[.?!])\s+"), " "),
    (re.compile(r"\s+"), " ")
]

OVERLAP_BOUNDARY = re.compile(r"[.?!]\s+|\n+")


def _recursive_segments(text, chunk_size, level=0):
    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size or level >= len(SPLIT_LEVELS):
        return [("", text)]

    pattern, separator = SPLIT_LEVELS[level]
    segments = []

    for part in pattern.split(text):
        sub_segments = _recursive_segments(
            part,
            chunk_size,
            level + 1
        )
        if not sub_segments:
            continue

        first_text = sub_segments[0][1]
        segments.append(
            (separator if segments else "", first_text)
        )
        segments.extend(sub_segments[1:])

    return segments


def _get_overlap(text, overlap):
    if overlap <= 0 or not text:
        return ""

    if len(text) <= overlap:
        return text.strip()

    tail = text[-overlap:]

    if not text[-overlap - 1].isspace() and not tail[0].isspace():
        match = re.search(r"\s", tail)
        if match is None:
            return ""
        tail = tail[match.end():]

    match = OVERLAP_BOUNDARY.search(tail)
    if match is not None and tail[match.end():].strip():
        tail = tail[match.end():]

    return tail.strip()


def _merge_segments(segments, chunk_size, overlap):
    chunks = []
    current = ""

    for separator, segment in segments:
        if not current:
            current = segment
        elif len(current) + len(separator) + len(segment) <= chunk_size:
            current += separator + segment
        else:
            chunks.append(current)
            overlap_text = _get_overlap(current, overlap)
            if (
                overlap_text
                and len(overlap_text) + len(separator) + len(segment) <= chunk_size
            ):
                current = overlap_text + separator + segment
            else:
                current = segment

    if current:
        chunks.append(current)

    return chunks


def split_text(text, chunk_size=800, overlap=150):
    if chunk_size <= 0:
        raise ValueError("chunk_size phải lớn hơn 0.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap phải >= 0 và nhỏ hơn chunk_size.")

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    sections = []
    current_section = ""
    current_lines = []

    for line in lines:
        line = line.strip()
        if line.isupper() and len(line) > 2:
            section_text = "\n".join(current_lines).strip()
            if section_text:
                sections.append({
                    "section": current_section,
                    "text": section_text
                })

            current_section = line
            if current_section == "THỐNG KÊ CƠ BẢN":
                current_section = "MEAN"
            current_lines = []
        else:
            current_lines.append(line)

    section_text = "\n".join(current_lines).strip()
    if section_text:
        sections.append({
            "section": current_section,
            "text": section_text
        })

    chunks = []
    for section in sections:
        segments = _recursive_segments(section["text"], chunk_size)
        for chunk_text in _merge_segments(segments, chunk_size, overlap):
            chunks.append({
                "text": chunk_text,
                "section": section["section"]
            })

    return chunks


# =========================
# SEARCH & RETRIEVAL
# =========================

def search_documents(query, top_k=3):
    if not query or not str(query).strip():
        return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}

    coll = _get_active_collection()
    try:
        if coll.count() == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}
    except Exception:
        pass

    model = _get_active_embedding_model()
    query_embedding = model.encode([query])
    emb_list = query_embedding.tolist() if hasattr(query_embedding, "tolist") else query_embedding

    results = coll.query(
        query_embeddings=emb_list,
        n_results=top_k
    )
    return results


def retrieve_documents(query, top_k=3, distance_threshold=18):
    results = search_documents(query, top_k)
    context = ""
    sources = []

    docs_list = (results.get("documents") or [[]])[0] or []
    metas_list = (results.get("metadatas") or [[]])[0] or []
    dists_list = (results.get("distances") or [[]])[0] or []

    for i in range(len(docs_list)):
        document = docs_list[i]
        metadata = metas_list[i] if i < len(metas_list) else {}
        distance = dists_list[i] if i < len(dists_list) else None

        if distance is not None and distance > distance_threshold:
            continue

        context += document + "\n\n"
        sources.append({
            "filename": metadata.get("filename", "unknown"),
            "chunk": metadata.get("chunk", i),
            "section": metadata.get("section", ""),
            "distance": distance
        })

    return context, sources


def get_context(query, top_k=2):
    results = search_documents(query, top_k)
    documents = (results.get("documents") or [[]])[0] or []
    context = ""
    for document in documents:
        context += document + "\n\n"
    return context


def get_sources(query, top_k=2):
    results = search_documents(query, top_k)
    sources = []
    metas = (results.get("metadatas") or [[]])[0] or []
    for i in range(len(metas)):
        metadata = metas[i]
        sources.append({
            "filename": metadata.get("filename", "unknown"),
            "chunk": metadata.get("chunk", i)
        })
    return sources


def generate_rag_answer(query):
    import ollama
    context, sources = retrieve_documents(query, top_k=2)

    prompt = f"""
Bạn là trợ lý AI trả lời câu hỏi dựa trên tài liệu được cung cấp.

Hãy sử dụng thông tin trong CONTEXT để trả lời câu hỏi.

Nếu CONTEXT không có đủ thông tin để trả lời,
hãy nói rằng tài liệu không cung cấp đủ thông tin.

Không được tự ý bịa thông tin.

CONTEXT:

{context}

CÂU HỎI:

{query}

Hãy trả lời bằng tiếng Việt.
"""

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"num_ctx": 8192}
    )

    answer = response["message"]["content"]
    return {
        "answer": answer,
        "sources": sources
    }


def get_document_list() -> List[str]:
    coll = _get_active_collection()
    try:
        documents = coll.get(include=["metadatas"])
        filenames = set()
        for metadata in documents.get("metadatas", []):
            if metadata and isinstance(metadata, dict) and "filename" in metadata:
                filenames.add(metadata["filename"])
            elif isinstance(metadata, list):
                for m in metadata:
                    if isinstance(m, dict) and "filename" in m:
                        filenames.add(m["filename"])
        return sorted(list(filenames))
    except Exception as err:
        print("[RAG]: Lỗi khi lấy danh sách document từ DB:", err)
        return []


def list_documents() -> List[str]:
    """
    Quét nhanh danh sách file hỗ trợ trong thư mục documents/
    theo extension mà không cần đọc hay parse nội dung file.
    """
    if not os.path.exists(DOCUMENT_FOLDER):
        return []
    files = []
    try:
        for filename in sorted(os.listdir(DOCUMENT_FOLDER)):
            ext = os.path.splitext(filename)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                file_path = os.path.join(DOCUMENT_FOLDER, filename)
                if os.path.isfile(file_path):
                    files.append(filename)
    except Exception as err:
        print("[RAG]: Lỗi khi duyệt thư mục documents/:", err)
    return files


def delete_document_from_database(filename: str) -> int:
    """
    Xóa toàn bộ vector thuộc về một file cụ thể trong ChromaDB.
    Tránh lỗi xóa nhầm prefix của các file tương tự (ví dụ: file_backup.txt).
    """
    coll = _get_active_collection()
    try:
        results = coll.get(where={"filename": filename})
    except Exception:
        results = None

    ids_to_delete = []
    if results and isinstance(results, dict) and "ids" in results and results["ids"]:
        ids_to_delete = results["ids"]
    elif results and hasattr(results, "get"):
        ids_to_delete = results.get("ids", []) or []

    if ids_to_delete and isinstance(ids_to_delete, list):
        coll.delete(ids=ids_to_delete)
        return len(ids_to_delete)
    return 0


def delete_document_file(filename: str) -> bool:
    file_path = os.path.join(DOCUMENT_FOLDER, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    return False


def index_document(filename: str) -> int:
    """
    Index một file cụ thể vào ChromaDB:
    - Xóa các vector cũ của file này trước khi upsert để tránh stale chunks.
    - Trả về số chunks được index.
    """
    file_path = os.path.join(DOCUMENT_FOLDER, filename)
    if not os.path.exists(file_path):
        return 0

    text = read_document(file_path)
    if not text or not text.strip():
        return 0

    chunks = split_text(text)
    if not chunks:
        return 0

    # Dọn sạch vector cũ trước khi nạp mới
    delete_document_from_database(filename)

    embeddings = create_embeddings(chunks)
    save_to_vector_database(chunks, embeddings, filename)
    return len(chunks)


def reindex_all_documents() -> int:
    documents = list_documents()
    total_chunks = 0
    for filename in documents:
        delete_document_from_database(filename)
        chunk_count = index_document(filename)
        total_chunks += chunk_count
    return total_chunks


def index_documents():
    """Batch index toàn bộ tài liệu."""
    documents = load_documents()
    print("Số tài liệu:", len(documents))
    for document in documents:
        filename = document["filename"]
        chunks = split_text(document["text"])
        if chunks:
            delete_document_from_database(filename)
            embeddings = create_embeddings(chunks)
            save_to_vector_database(chunks, embeddings, filename)
            print(f"\nĐã index: {filename} | Số chunk: {len(chunks)}")
    coll = _get_active_collection()
    print("\nTổng số vector:", coll.count())