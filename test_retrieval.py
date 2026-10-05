from rag import search_documents


queries = [
    "Mean là gì?",
    "Median có ưu điểm gì?",
    "Variance dùng để làm gì?",
    "Outlier ảnh hưởng đến mean như thế nào?"
]


for query in queries:

    results = search_documents(
        query,
        top_k=3
    )

    print("\n==============================")

    print(
        "CÂU HỎI:",
        query
    )

    print(
        "\nKẾT QUẢ RETRIEVAL:"
    )

    for i in range(
        len(results["documents"][0])
    ):

        document = results["documents"][0][i]

        distance = results["distances"][0][i]

        metadata = results["metadatas"][0][i]

        print(
            f"\n--- Chunk {i + 1} ---"
        )

        print(
            "Distance:",
            distance
        )

        print(
            "File:",
            metadata["filename"]
        )

        print(
            "Chunk ID:",
            metadata["chunk"]
        )

        print(
            "\nNội dung:"
        )

        print(document)