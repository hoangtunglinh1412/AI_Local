from rag import retrieve_documents

query = "Machine Learning là gì?"

context, sources = retrieve_documents(
    query,
    top_k=3,
    distance_threshold=18
)

print("\n===== SOURCES =====")

if len(sources) == 0:

    print("Không tìm thấy chunk.")

else:

    for source in sources:

        print(
            "\nFile:",
            source["filename"]
        )

        print(
            "Section:",
            source["section"]
        )

        print(
            "Chunk:",
            source["chunk"]
        )

        print(
            "Distance:",
            source["distance"]
        )

print("\n===== CONTEXT =====")
print(context)
