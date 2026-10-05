from rag import get_sources


query = "Mean là gì?"


sources = get_sources(
    query,
    top_k=2
)


print("\n===== SOURCES =====")


for source in sources:

    print(
        "File:",
        source["filename"]
    )

    print(
        "Chunk:",
        source["chunk"]
    )

    print()