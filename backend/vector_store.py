from pathlib import Path
import chromadb


class VectorStore:

    def __init__(self):

        BASE_DIR = Path(__file__).resolve().parent.parent
        db_path = BASE_DIR / "vector_db"

        self.client = chromadb.PersistentClient(
            path=str(db_path)
        )

        self.collection = self.client.get_or_create_collection(
            name="resume"
        )

    def reset_collection(self):

        try:
            self.client.delete_collection(
                name="resume"
            )

            print("Old collection deleted.")

        except Exception as e:
            print(f"Could not delete old collection: {e}")

        self.collection = self.client.get_or_create_collection(
            name="resume"
        )

        print("New collection created.")

    def add_chunks(self, embedded_chunks):

        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for chunk in embedded_chunks:

            ids.append(
                f"{chunk['source']}_"
                f"{chunk['page']}_"
                f"{chunk['chunk_id']}"
            )

            documents.append(
                chunk["text"]
            )

            embeddings.append(
                chunk["embedding"]
            )

            metadatas.append({
                "source": chunk["source"],
                "page": chunk["page"],
                "type": chunk.get(
                    "type",
                    "unknown"
                ),
                "chunk_id": chunk["chunk_id"]
            })

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(
            f"Stored {len(ids)} chunks."
        )

    def search(
        self,
        query_embedding,
        n_results=3
    ):

        results = self.collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=n_results
        )

        return results