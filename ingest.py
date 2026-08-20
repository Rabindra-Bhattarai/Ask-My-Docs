import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

DOCS_DIR ="docs"
CHUNK_SIZE= 500
CHUNK_OVERLAP = 50

def load_pdf_text(path):
    reader =PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text
def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks
def main():
    print("Loading embadding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")


    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection("documents")


    chunk_id = 0
    for filename in os.listdir(DOCS_DIR):
        if not filename.lower().endswith(".pdf"):
            continue


        print(f"Reading {filename}...")
        path = os.path.join(DOCS_DIR, filename)
        text = load_pdf_text(path)
        chunks = chunk_text(text)
        print(f" ->> {len(chunks)} chunks.created.")
        embeddings = model.encode(chunks).tolist(
        )

        ids=[f"chunk_{chunk_id + i}" for i in range(len(chunks))]
        metadatas = [{"source": filename} for _ in range(len(chunks))]


        collection.add(
            ids =ids,
            embeddings = embeddings,
            documents = chunks,
            metadatas = metadatas,

        )

        chunk_id += len(chunks)
    print(f"Done. Total chunks Stored: {collection.count()}")

if __name__ == "__main__":
    main()

