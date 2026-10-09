# from langchain_text_splitters import RecursiveCharacterTextSplitter


# def split_documents(documents):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=100
#     )

#     chunks = []

#     for doc in documents:
#         split_texts = splitter.split_text(doc["text"])

#         for i, chunk in enumerate(split_texts):
#             chunks.append({
#                 "text": chunk,
#                 "source": doc.get(
#                     "source",
#                     "Unknown"
#                 ),
#                 "page": doc.get(
#                     "page",
#                     0
#                 ),
#                 "type": doc.get(
#                     "type",
#                     "unknown"
#                 ),
#                 "chunk_id": i
#             })
          

#     return chunks


from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]

md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=HEADERS,
    strip_headers=False,
)

char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

# pypdf sometimes inserts a space after a capital letter
PDF_FIXES = {
    "T echnologies": "Technologies",
    "A WS": "AWS",
}


def clean_pdf_text(text):
    for wrong, right in PDF_FIXES.items():
        text = text.replace(wrong, right)
    return text


def _breadcrumb(metadata):
    """'EXL Solutions > Major Projects > PPE Detection System'"""
    parts = [metadata.get(k) for k in ("h1", "h2", "h3") if metadata.get(k)]
    return " > ".join(parts)


def _split_markdown(text):
    pieces = []

    for section in md_splitter.split_text(text):
        crumb = _breadcrumb(section.metadata)
        body = section.page_content.strip()

        if not body:
            continue

        sub_chunks = (
            [body] if len(body) <= CHUNK_SIZE
            else char_splitter.split_text(body)
        )

        for sub in sub_chunks:
            # Keep the section path inside every chunk so a chunk like
            # "Automation Engineer" still knows it belongs to EXL Solutions.
            pieces.append(f"[{crumb}]\n{sub}" if crumb else sub)

    return pieces


def split_documents(documents):
    chunks = []

    for doc in documents:
        doc_type = doc.get("type", "unknown")

        if doc_type == "md":
            split_texts = _split_markdown(doc["text"])
        elif doc_type == "pdf":
            split_texts = char_splitter.split_text(clean_pdf_text(doc["text"]))
        else:
            split_texts = char_splitter.split_text(doc["text"])

        for i, chunk in enumerate(split_texts):
            chunks.append({
                "text": chunk,
                "source": doc.get("source", "Unknown"),
                "page": doc.get("page", 0),
                "type": doc_type,
                "chunk_id": i,
            })

    return chunks