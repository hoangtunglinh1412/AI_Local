from rag import retrieve_documents

query = "Mean là gì?"

context, sources = retrieve_documents(
    query,
    top_k=3,
    distance_threshold=18
)

print("\n===== RETRIEVAL SOURCES =====")

for source in sources:

    print("\nFile:", source["filename"])
    print("Section:", source["section"])
    print("Chunk:", source["chunk"])
    print("Distance:", source["distance"])