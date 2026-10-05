from rag import load_documents, split_text

documents = load_documents()

for document in documents:

    print("\n==============================")
    print("FILE:", document["filename"])

    chunks = split_text(
        document["text"]
    )

    for i in range(len(chunks)):

        print("\n--- CHUNK", i, "---")

        print(
            "Section:",
            chunks[i]["section"]
        )

        print(
            "Text:"
        )

        print(
            chunks[i]["text"]
        )