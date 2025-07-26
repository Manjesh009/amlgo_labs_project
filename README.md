AI-Powered Chatbot with RAG Pipeline 

## **Project Overview**
This project demonstrates a **Retrieval-Augmented Generation (RAG)** pipeline for building an AI-powered chatbot capable of answering user queries based on a provided document set.  

The solution:
- Preprocesses documents into semantic **chunks**.
- Generates **embeddings** and stores them in **ChromaDB**.
- Uses a **Retriever + LLM** combination to provide context-grounded answers.
- Provides an interactive **Streamlit UI** with **real-time streaming responses**.


## **Project Architecture & Flow**

PDF Document(s)
↓
Preprocessing (cleaning, chunking)
↓
Embeddings (MiniLM)
↓
Vector Database (ChromaDB)
↓
Retriever (semantic search)
↓
Prompt Template + LLM (Ollama LLaMA3)
↓
Streamlit Chatbot with Streaming


**Main Components:**
1. **Document Ingestion** – Loads PDFs, splits into chunks, and stores in ChromaDB.
2. **Retriever** – Finds top-k relevant chunks for a user query.
3. **LLM (Generator)** – Uses the retrieved context to produce accurate answers.
4. **Streamlit Chatbot** – Provides a user-friendly interface with streaming.

---

## **Steps to Run**

### **Clone & Setup**
git clone https://github.com/Manjesh009/amlgo_labs_project.git
cd https://github.com/Manjesh009/amlgo_labs_project.git
python -m venv venv
source venv/bin/activate        
pip install -r requirements.txt

Place your PDF documents in the data/ folder.

Run ingestion to generate chunks and store embeddings:

python src/ingestion/ingest.py

This will:

Save chunks to /chunks/document_chunks.json

Persist the ChromaDB inside /vectordb

Run the RAG Pipeline (CLI Testing)

python src/rag_pipeline.py

Ensure Ollama is running locally:

ollama serve

streamlit run app.py

