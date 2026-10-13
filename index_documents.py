from rag import (
    list_documents,
    index_document,
    get_collection
)


def main():
    documents = list_documents()

    print(
        "Số tài liệu:",
        len(documents)
    )

    if not documents:
        print("Không có tài liệu nào trong thư mục documents/ để index.")
        return

    total_chunks = 0
    for filename in documents:
        print(f"\nĐang index: {filename}")
        chunk_count = index_document(filename)
        total_chunks += chunk_count
        print(
            "Số chunk:",
            chunk_count
        )

    coll = get_collection()
    print(
        "\nTổng số vector:",
        coll.count()
    )


if __name__ == "__main__":
    main()