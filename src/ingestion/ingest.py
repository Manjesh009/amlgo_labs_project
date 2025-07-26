import os
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

PDF_PATH    = os.getenv("PDF_PATH", "data/AI Training Document.pdf")
CHROMA_PATH = os.getenv("CHROMA_PATH", "vectordb")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))


def save_chunks(chunks, filename="document_chunks.json"):
    os.makedirs("chunks", exist_ok=True)
    data = []
    for idx, chunk in enumerate(chunks):
        data.append({
            "id": f"chunk_{idx+1}",
            "text": chunk.page_content,
            "page": chunk.metadata.get("page", None)
        })
    with open(f"chunks/{filename}", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"📂 Chunks saved to chunks/{filename}")

def ingest(pdf_path: str = PDF_PATH,
           chroma_path: str = CHROMA_PATH,
           embed_model: str = EMBED_MODEL,
           chunk_size: int = CHUNK_SIZE,
           chunk_overlap: int = CHUNK_OVERLAP):
    print(f"📄 Loading: {pdf_path}")
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    total_pages = len(docs)
    page_nums = [d.metadata.get("page") for d in docs]
    unique_pages = sorted(set(p for p in page_nums if p is not None))
    print(f"✅ Loaded {total_pages} pages")
    print(f"🔍 Page indices detected: {unique_pages}")
    if len(unique_pages) != total_pages:
        print(f"⚠️ Warning: unique page indices ({len(unique_pages)}) "
              f"≠ docs loaded ({total_pages}). PDF may be image-only / odd structure.")

    for i, d in enumerate(docs[:3]):
        print(f"🧾 Page {d.metadata.get('page')}: {d.page_content[:100].strip()}...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    print(f"🧩 Created {len(chunks)} chunks")

    save_chunks(chunks, "document_chunks.json")

    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    db = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=chroma_path
    )
    
    db.persist()
    print(f"💾 Saved to Chroma at: {chroma_path}")
    return len(chunks)

if __name__ == "__main__":
    ingest()
