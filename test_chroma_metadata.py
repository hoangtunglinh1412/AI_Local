from rag import collection

results = collection.get()

print("\n===== CHROMADB METADATA =====")

for i in range(len(results["ids"])):

    print("\n--- VECTOR", i, "---")

    print(
        "ID:",
        results["ids"][i]
    )

    print(
        "Document:",
        results["documents"][i]
    )

    print(
        "Metadata:",
        results["metadatas"][i]
    )