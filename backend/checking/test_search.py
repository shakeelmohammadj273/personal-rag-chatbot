from embeddings import model
from vector_store import VectorStore


store = VectorStore()

query = "Tell me about your Python experience"

query_embedding = model.encode(query).tolist()

results = store.search(query_embedding)

print(results["documents"][0])