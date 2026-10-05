import os
import ollama
import chromadb

from pypdf import PdfReader
from docx import Document

from sentence_transformers import SentenceTransformer

MODEL = "qwen3:8b"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

DOCUMENT_FOLDER = "documents"

CHROMA_DB_PATH = "chroma_db"

chroma_client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH
)

collection = chroma_client.get_or_create_collection(
    name="documents"
)

def create_embeddings(chunks):

    texts = []

    for chunk in chunks:
        texts.append(
            chunk["text"]
        )

    embeddings = embedding_model.encode(
        texts
    )

    return embeddings

def save_to_vector_database(
    chunks,
    embeddings,
    filename
):

    ids = []
    metadatas = []
    documents = []

    for i in range(len(chunks)):

        ids.append(
            filename + "_" + str(i)
        )

        metadatas.append({
            "filename": filename,
            "chunk": i,
            "section": chunks[i]["section"]
        })

        documents.append(
            chunks[i]["text"]
        )

    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas
    )

def index_documents():

    documents = load_documents()

    print(
        "Số tài liệu:",
        len(documents)
    )

    for document in documents:

        filename = document["filename"]

        chunks = split_text(
            document["text"]
        )

        embeddings = create_embeddings(
            chunks
        )

        save_to_vector_database(
            chunks,
            embeddings,
            filename
        )

        print(
            "\nĐã index:",
            filename
        )

        print(
            "Số chunk:",
            len(chunks)
        )

    print(
        "\nTổng số vector:",
        collection.count()
    )

def read_txt(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()

    return text


def read_pdf(file_path):

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def read_docx(file_path):

    document = Document(file_path)

    text = ""

    for paragraph in document.paragraphs:

        text += paragraph.text + "\n"

    return text


def read_document(file_path):

    extension = os.path.splitext(
        file_path
    )[1].lower()

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

    for filename in os.listdir(
        DOCUMENT_FOLDER
    ):

        file_path = os.path.join(
            DOCUMENT_FOLDER,
            filename
        )

        text = read_document(
            file_path
        )

        if text.strip():

            documents.append({
                "filename": filename,
                "text": text
            })

    return documents

def split_text(
    text,
    max_chunk_size=1000
):

    lines = text.split("\n")

    sections = []

    current_section = ""
    current_lines = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        if line.isupper() and len(line) > 2:

            if len(current_lines) > 0:

                sections.append({
                    "section": current_section,
                    "text": "\n".join(current_lines)
                })

            current_section = line

            if current_section == "THỐNG KÊ CƠ BẢN":
                current_section = "MEAN"

            current_lines = []

        else:

            current_lines.append(line)

    if len(current_lines) > 0:

        sections.append({
            "section": current_section,
            "text": "\n".join(current_lines)
        })

    chunks = []

    for section in sections:

        section_name = section["section"]
        lines = section["text"].split("\n")

        current_chunk = ""

        for line in lines:

            if not current_chunk:

                current_chunk = line

            elif len(current_chunk) + len(line) + 1 <= max_chunk_size:

                current_chunk += "\n" + line

            else:

                chunks.append({
                    "text": current_chunk,
                    "section": section_name
                })

                current_chunk = line

        if current_chunk:

            chunks.append({
                "text": current_chunk,
                "section": section_name
            })

    return chunks

def search_documents(
    query,
    top_k=3
):

    query_embedding = embedding_model.encode(
        [query]
    )

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k
    )

    return results

def retrieve_documents(
    query,
    top_k=3,
    distance_threshold=18
):

    results = search_documents(
        query,
        top_k
    )

    context = ""

    sources = []

    for i in range(
        len(results["documents"][0])
    ):

        document = results["documents"][0][i]

        metadata = results["metadatas"][0][i]

        distance = results["distances"][0][i]


        # =========================
        # DISTANCE FILTER
        # =========================

        if distance > distance_threshold:
            continue

        context += document + "\n\n"

        sources.append({
            "filename": metadata["filename"],
            "chunk": metadata["chunk"],
            "section": metadata["section"],
            "distance": distance
        })

    return context, sources

def get_context(
    query,
    top_k=2
):
    results = search_documents(
        query,
        top_k
    )

    documents = results["documents"]

    context = ""

    for document in documents[0]:
        context += document + "\n\n"

    return context

def get_sources(
    query,
    top_k=2
):
    results = search_documents(
        query,
        top_k
    )

    sources = []

    for i in range(
        len(results["documents"][0])
    ):
        metadata = results["metadatas"][0][i]

        sources.append({
            "filename": metadata["filename"],
            "chunk": metadata["chunk"]
        })

    return sources

def generate_rag_answer(query):

    context, sources = retrieve_documents(
        query,
        top_k=2
    )

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
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "num_ctx": 8192
        }
    )

    answer = response["message"]["content"]

    return {
        "answer": answer,
        "sources": sources
    }

def get_document_list():

    documents = collection.get(
        include=["metadatas"]
    )

    filenames = set()

    for metadata in documents["metadatas"]:

        filenames.add(
            metadata["filename"]
        )

    return list(filenames)

def list_documents():

    if not os.path.exists(DOCUMENT_FOLDER):
        return []

    files = []

    for filename in os.listdir(DOCUMENT_FOLDER):

        file_path = os.path.join(
            DOCUMENT_FOLDER,
            filename
        )

        if os.path.isfile(file_path):

            text = read_document(
                file_path
            )

            if text.strip():

                files.append(filename)

    return files

def delete_document_from_database(filename):

    results = collection.get()

    ids_to_delete = []

    for i in range(
        len(results["ids"])
    ):

        document_id = results["ids"][i]

        if document_id.startswith(
            filename + "_"
        ):

            ids_to_delete.append(
                document_id
            )

    if len(ids_to_delete) > 0:

        collection.delete(
            ids=ids_to_delete
        )

    return len(ids_to_delete)

def delete_document_file(filename):
    file_path = os.path.join(
        DOCUMENT_FOLDER,
        filename
    )

    if os.path.exists(file_path):
        os.remove(file_path)
        return True

    return False

def index_document(filename):

    file_path = os.path.join(
        DOCUMENT_FOLDER,
        filename
    )

    if not os.path.exists(file_path):

        return 0

    text = read_document(
        file_path
    )

    if not text.strip():

        return 0

    chunks = split_text(
        text
    )

    embeddings = create_embeddings(
        chunks
    )

    save_to_vector_database(
        chunks,
        embeddings,
        filename
    )

    return len(chunks)

def reindex_all_documents():

    documents = list_documents()

    total_chunks = 0

    for filename in documents:

        delete_document_from_database(
            filename
        )

        chunk_count = index_document(
            filename
        )

        total_chunks += chunk_count

    return total_chunks