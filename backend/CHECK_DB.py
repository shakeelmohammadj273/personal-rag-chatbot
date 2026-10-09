"""Retrieval check: run after re-indexing, and whenever documents change.

Shows the best distances for relevant AND off-topic questions so you can
pick MAX_DISTANCE (in chat.py) between the two groups.
"""
from chat import MAX_DISTANCE, QUERY_PREFIX
from embeddings import model
from vector_store import VectorStore

# (question, expected source file or None for off-topic)
QUESTIONS = [
    ("What is your current job?", "experience.md"),
    ("What are you doing in your current company?", "experience.md"),
    ("Tell me about your AI experience.", "experience.md"),
    ("What are your technical skills?", "skills.md"),
    ("What is your educational background?", "education.md"),
    ("Tell me about your projects.", "projects.md"),
    ("What certifications do you have?", "certificate.md"),
    # Off-topic: these should all be farther than the relevant ones
    ("Who won the football World Cup?", None),
    ("What is the capital of France?", None),
    ("Give me a recipe for biryani.", None),
    ("How do I fix my car engine?", None),
]

SHOW_TEXT = False  # set True to print the retrieved chunk text

store = VectorStore()
print("=" * 60)
print(f"Total chunks in DB: {store.collection.count()}")
print(f"MAX_DISTANCE in chat.py: {MAX_DISTANCE}")
print("=" * 60)

relevant_worst, offtopic_best = [], []

for question, expected in QUESTIONS:
    embedding = model.encode(
        QUERY_PREFIX + question, normalize_embeddings=True
    ).tolist()

    results = store.search(query_embedding=embedding, n_results=5)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    print(f"\nQ: {question}")

    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
        keep = "keep" if dist <= MAX_DISTANCE else "DROP"
        print(f"  {rank}. {dist:.3f} [{keep}] {meta.get('source')}")
        if SHOW_TEXT:
            print("     " + doc.replace("\n", " ")[:200])

    if expected:
        # distance of the first chunk that comes from the expected file
        match = next(
            (d for m, d in zip(metas, dists) if m.get("source") == expected),
            None,
        )
        top_ok = metas[0].get("source") == expected
        print(f"  -> expected {expected}: "
              f"{'top result' if top_ok else 'NOT top'}"
              f"{'' if match is None else f' (best {match:.3f})'}")
        if match is not None:
            relevant_worst.append(match)
    else:
        offtopic_best.append(dists[0])

print("\n" + "=" * 60)
if relevant_worst and offtopic_best:
    print(f"Worst relevant match : {max(relevant_worst):.3f}")
    print(f"Best off-topic match : {min(offtopic_best):.3f}")
    print("Pick MAX_DISTANCE between these two numbers.")
    if max(relevant_worst) >= min(offtopic_best):
        print("They overlap: improve the documents/chunking before tuning.")
print("=" * 60)