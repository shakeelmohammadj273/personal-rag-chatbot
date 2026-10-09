from pathlib import Path
from pypdf import PdfReader

from text_splitter import split_documents
from embeddings import create_embeddings
from vector_store import VectorStore


BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "documents"


def load_pdf(file_path):
    """Load text from a PDF file."""
    reader = PdfReader(str(file_path))

    documents = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text and text.strip():
            documents.append({
                "text": text,
                "source": file_path.name,
                "page": page_num,
                "type": "pdf"
            })

    return documents


def load_text_file(file_path):
    """Load text from Markdown or TXT files."""
    text = file_path.read_text(
        encoding="utf-8"
    )

    if text.strip():
        return [{
            "text": text,
            "source": file_path.name,
            "page": 0,
            "type": file_path.suffix.lower().replace(".", "")
        }]

    return []


def load_all_documents():
    """Load all supported documents from the documents folder."""

    documents = []

    if not DOCUMENTS_DIR.exists():
        print(f"Documents folder not found: {DOCUMENTS_DIR}")
        return documents

    for file_path in DOCUMENTS_DIR.iterdir():

        if not file_path.is_file():
            continue

        extension = file_path.suffix.lower()

        try:

            if extension == ".pdf":
                loaded = load_pdf(file_path)

            elif extension in [".md", ".txt"]:
                loaded = load_text_file(file_path)

            else:
                print(f"Skipping unsupported file: {file_path.name}")
                continue

            documents.extend(loaded)

            print(
                f"Loaded: {file_path.name} "
                f"({len(loaded)} document sections)"
            )

        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")

    return documents


def main():

    print("=" * 60)
    print("PERSONAL AI CHATBOT - DOCUMENT INDEXING")
    print("=" * 60)

    # -----------------------------------------
    # 1. Load documents
    # -----------------------------------------

    print("\n[1] Loading documents...")

    documents = load_all_documents()

    print(f"\nTotal document sections loaded: {len(documents)}")

    if not documents:
        print("No documents found.")
        return

    # -----------------------------------------
    # 2. Split documents into chunks
    # -----------------------------------------

    print("\n[2] Splitting documents into chunks...")

    chunks = split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    # -----------------------------------------
    # 3. Create embeddings
    # -----------------------------------------

    print("\n[3] Creating embeddings...")

    embedded_chunks = create_embeddings(chunks)

    print(
        f"Total embeddings created: "
        f"{len(embedded_chunks)}"
    )

    # -----------------------------------------
    # 4. Connect to ChromaDB
    # -----------------------------------------

    print("\n[4] Connecting to ChromaDB...")

    store = VectorStore()

    # -----------------------------------------
    # 5. Clear old database
    # -----------------------------------------

    print("\n[5] Clearing old knowledge base...")
    store.reset_collection()

    # -----------------------------------------
    # 6. Store new embeddings
    # -----------------------------------------

    print("\n[6] Storing embeddings...")

    store.add_chunks(embedded_chunks)

    # -----------------------------------------
    # 7. Final information
    # -----------------------------------------

    print("\n" + "=" * 60)
    print("INDEXING COMPLETED")
    print("=" * 60)

    print(
        f"Documents loaded : {len(documents)}"
    )

    print(
        f"Chunks created   : {len(chunks)}"
    )

    print(
        f"Database records : {store.collection.count()}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()