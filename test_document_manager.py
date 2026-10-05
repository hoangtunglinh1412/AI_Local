from rag import (
    list_documents,
    index_document,
    delete_document_from_database,
    reindex_all_documents,
    collection
)


print("\n===== DOCUMENT LIST =====")

documents = list_documents()

for document in documents:

    print(
        document
    )


print("\n===== DATABASE =====")

print(
    "Tổng số vector:",
    collection.count()
)


print("\n===== TEST INDEX =====")

if len(documents) > 0:

    filename = documents[0]

    chunk_count = index_document(
        filename
    )

    print(
        "File:",
        filename
    )

    print(
        "Số chunk:",
        chunk_count
    )


print("\n===== DATABASE AFTER INDEX =====")

print(
    "Tổng số vector:",
    collection.count()
)

print("\n===== TEST DELETE =====")

deleted = delete_document_from_database(
    "statistics.txt"
)

print(
    "Đã xóa:",
    deleted,
    "vector"
)

print(
    "Tổng số vector:",
    collection.count()
)