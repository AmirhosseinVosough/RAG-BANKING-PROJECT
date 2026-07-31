from modules.vector_store import VectorStore

vector_store = VectorStore()

for chunk in vector_store.all_chunks():
    print("\n--- CHUNK ---")
    print("ID:", chunk.chunk_id)
    print("Metadata:", chunk.metadata)
    print("Text:", chunk.text)