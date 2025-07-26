import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()

CHROMA_PATH   = os.getenv("CHROMA_PATH", "vectordb") 
EMBED_MODEL   = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_API    = os.getenv("OLLAMA_API", "http://localhost:11434")
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "llama3")
TOP_K         = int(os.getenv("TOP_K", 3))

PROMPT_TEMPLATE = """
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


def get_qa_chain(
    chroma_path: str = CHROMA_PATH,
    embed_model: str = EMBED_MODEL,
    ollama_base_url: str = OLLAMA_API,
    ollama_model: str = OLLAMA_MODEL,
    k: int = TOP_K,
) -> RetrievalQA:
    # Embeddings
    embedding = HuggingFaceEmbeddings(model_name=embed_model)

    # Vector DB 
    vectordb = Chroma(
        persist_directory=chroma_path,
        embedding_function=embedding,
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": k})

    # LLM (Ollama)
    llm = ChatOllama(model=ollama_model, base_url=ollama_base_url)

    # Prompt
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    # Chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return chain


def answer(query: str, chain: RetrievalQA):
    """Convenience helper to call the chain and return (answer, sources)."""
    result = chain.invoke({"query": query})
    return result["result"], result.get("source_documents", [])


if __name__ == "__main__":
    print("\n🫀 Chatbot is ready! Type your question (or 'exit' to quit).\n")
    qa_chain = get_qa_chain()

    while True:
        query = input("🧍 You: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting. Stay heart healthy!")
            break

        result, sources = answer(query, qa_chain)

        print("\n🤖 AI:", result)
        print("\n📚 Sources:")
        for i, doc in enumerate(sources, 1):
            print(f"- [{i}] {doc.metadata.get('source', 'Unknown')} (page: {doc.metadata.get('page')})")
