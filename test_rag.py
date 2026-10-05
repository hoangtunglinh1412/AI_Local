from rag import generate_rag_answer


queries = [
    "Mean là gì?",
    "Median có ưu điểm gì?",
    "Variance dùng để làm gì?",
    "Outlier ảnh hưởng đến mean như thế nào?",
    "Regression là gì?"
]


for query in queries:

    result = generate_rag_answer(
        query
    )

    print("\n==============================")

    print(
        "CÂU HỎI:",
        query
    )

    print(
        "\nTRẢ LỜI:"
    )

    print(
        result["answer"]
    )

    print(
        "\n===== SOURCES ====="
    )

    for source in result["sources"]:

        print(
            "File:",
            source["filename"]
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