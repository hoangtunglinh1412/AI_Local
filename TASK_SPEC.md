# TASK SPECIFICATION: LOCAL AI ASSISTANT COMPLETION

## 1. System Context & Hardware Target
- Host Machine: Windows 11, NVIDIA GeForce RTX 3060 Ti (8GB VRAM).
- LLM Runtime: Ollama with model `qwen3:8b` (Q4_K_M, num_ctx = 8192).
- Embedding Engine: SentenceTransformers (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 384 dimensions).
- Vector Database: ChromaDB (persistent path: `chroma_db/`).
- Relational State: SQLite (`chat_history.db`).
- Codebase Directory:
  - `chatbot.py`: Terminal conversation loop, context injection, and command dispatcher.
  - `memory.py`: Database operations for messages, long-term memories, summaries, and checkpoints.
  - `rag.py`: Text extraction (PDF/DOCX/TXT), chunking, embedding generation, ChromaDB vector search.
  - `index_documents.py`: Batch indexing script for files in `documents/`.

---

## 2. Current Implementation State
- Memory Subsystem (V2.5.4 Complete):
  - SQLite tables: `messages`, `memories`, `summaries`, `summary_checkpoints`.
  - Automatic summarization triggers when: `(count_messages - checkpoint) >= 20`.
  - Context window reset: Retains `1 system_prompt` + `10 recent messages` in active RAM after summarization.
  - Manual commands working: `/remember <text>`, `/memory`, `/forget <id>`, `/summarize`.
- RAG Subsystem (V3.8 Prototype):
  - ChromaDB collection `documents` operational.
  - Issue: Current chunking logic in `rag.py` splits text by fixed character boundaries or manual section breaks, causing truncated concepts.
  - RAG search logic (`search_documents`) is not yet wired into `chatbot.py`.

---

## 3. Mandatory Engineering Deliverables

### Deliverable 1: Semantic Recursive Chunking (`rag.py`)
- Refactor `split_text(text, chunk_size=800, overlap=150)` using recursive splitting hierarchy:
  1. Primary split by double line breaks (`\n\n` - paragraphs).
  2. Secondary split by single line break (`\n` - lines).
  3. Tertiary split by sentence terminators (`. `, `? `, `! `).
  4. Fallback split by whitespace.
- Guarantee that individual sentences are not sliced mid-word and chunks retain semantic completeness.

### Deliverable 2: Integrate RAG into Chat Loop (`chatbot.py`)
- Import `search_documents(query, top_k=2)` from `rag.py` into `chatbot.py`.
- Add command toggles:
  - `/rag on`: Enable automatic context injection (default state: ON).
  - `/rag off`: Disable document retrieval.
  - `/rag status`: Print current RAG mode.
- Context Injection Protocol:
  - When RAG is active, query ChromaDB using user's prompt.
  - If relevant chunks are retrieved, format them cleanly into `system_prompt` under a header `CONTEXT TÀI LIỆU:`.
  - Ensure retrieved context does not overwrite Long-term Memory or Conversation Summary in `rebuild_system_prompt()`.

### Deliverable 3: Enable Token Streaming
- Update the main response generation call in `chatbot.py` to use `stream=True`:
  ```python
  response_stream = ollama.chat(
      model=MODEL,
      messages=messages,
      options={"num_ctx": 8192},
      stream=True
  )