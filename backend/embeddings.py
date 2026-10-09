from sentence_transformers import SentenceTransformer

# Load the embedding model once
model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def create_embeddings(chunks):
    """
    Generate embeddings for each text chunk.
    """

    embedded_chunks = []

    for chunk in chunks:

        embedding = model.encode(chunk["text"]).tolist()

        embedded_chunks.append({
            "text": chunk["text"],
            "embedding": embedding,
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"]
        })

    return embedded_chunks