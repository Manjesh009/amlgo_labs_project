import os
import streamlit as st
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

load_dotenv()

CHROMA_PATH   = os.getenv("CHROMA_PATH", "vectordb")          # align with your ingest
EMBED_MODEL   = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_API    = os.getenv("OLLAMA_API", "http://localhost:11434")
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "llama3")
TOP_K         = int(os.getenv("TOP_K", 3))
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 800))            
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))          

CUSTOM_PROMPT = """
You are an expert assistant helping to answer questions based on a document that describes 
a technical assignment for building a RAG-based chatbot.

Use the provided context (retrieved chunks from the document) to answer the user query. 
If the context does not contain enough information, state clearly: "The document does not 
provide this information."

Instructions:
- Answer in 2-4 clear sentences unless the question requires a detailed explanation.
- Quote or paraphrase relevant parts of the document when possible.
- Do not include information that is not supported by the context.

Context:
{context}

User: {question}
AI Response:
"""

prompt = PromptTemplate(
    template=CUSTOM_PROMPT,
    input_variables=["context", "question"]
)

@st.cache_resource
def load_pipeline():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_API)
    # count chunks/docs
    try:
        num_chunks = vectordb._collection.count()  
    except Exception:
        num_chunks = None
    return llm, retriever, vectordb, num_chunks

def stream_answer(llm: ChatOllama, formatted_prompt: str):
    """Stream tokens from Ollama to Streamlit in real time."""
    container = st.empty()
    full_text = ""
    for chunk in llm.stream([HumanMessage(content=formatted_prompt)]):
        token = chunk.content or ""
        full_text += token
        container.markdown(full_text)
    return full_text

st.set_page_config(page_title="RAG Based Chatbot", layout="wide")
st.title("Hi! I am your AI Buddy")
st.write("Ask anything related to AI Task")

llm, retriever, vectordb, num_chunks = load_pipeline()

# Sidebar
with st.sidebar:
    st.header("ℹ️ Model / Index Info")
    st.write(f"**LLM**: `{OLLAMA_MODEL}` @ {OLLAMA_API}")
    st.write(f"**Embedding**: `{EMBED_MODEL}`")
    st.write(f"**k (top docs)**: {TOP_K}")
    st.write(f"**Chunk size / overlap**: {CHUNK_SIZE} / {CHUNK_OVERLAP}")
    if num_chunks is not None:
        st.write(f"**Indexed chunks**: {num_chunks}")
    else:
        st.write("**Indexed chunks**: (unavailable)")

    if st.button("🧼 Clear chat / reset"):
        st.session_state.chat = []
        st.experimental_rerun()

# Chat history
if "chat" not in st.session_state:
    st.session_state.chat = []

# Chat UI
for turn in st.session_state.chat:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        with st.expander("📚 Sources"):
            for i, d in enumerate(turn["sources"], 1):
                st.markdown(f"**[{i}]** page: `{d['page']}`")
                st.code(d["text"], language="markdown")

# Input box
user_query = st.chat_input("💬 Your question...")
if user_query:

    with st.chat_message("user"):
        st.markdown(user_query)


    with st.spinner("🔎 Retrieving relevant context..."):
        docs = retriever.get_relevant_documents(user_query)
        context = "\n\n".join(d.page_content for d in docs)

    with st.chat_message("assistant"):
        formatted = prompt.format(context=context, question=user_query)
        answer_text = stream_answer(llm, formatted)


        with st.expander("📚 Sources"):
            srcs = []
            for d in docs:
                srcs.append({
                    "text": d.page_content[:1000],  
                    "page": d.metadata.get("page", "NA"),
                    "source": d.metadata.get("source", "Unknown")
                })
                st.markdown(f"- **page:** `{d.metadata.get('page', 'NA')}`, **source:** `{d.metadata.get('source', 'Unknown')}`")
                st.code(d.page_content[:1000], language="markdown")

    st.session_state.chat.append({
        "question": user_query,
        "answer": answer_text,
        "sources": srcs
    })
