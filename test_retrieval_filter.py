from rag import retrieve_documents


queries = [
    "Mean là gì?",
    "Median có ưu điểm gì?",
    "Variance dùng để làm gì?",
    "Outlier ảnh hưởng đến mean như thế nào?",
    "Regression là gì?"
]


for query in queries:

    context, sources = retrieve_documents(
        query,
        top_k=3,
        distance_threshold=18
    )

    print("\n==============================")

    print("CÂU HỎI:", query)

    print("\nSOURCES:")

    if len(sources) == 0:

        print("Không có chunk phù hợp.")

    else:

        for source in sources:

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