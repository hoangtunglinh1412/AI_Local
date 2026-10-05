from rag import (
    load_documents,
    split_text,
    create_embeddings,
    save_to_vector_database,
    collection
)


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