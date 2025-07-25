import os
import json
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

VECTOR_DIR = "memory_store"

def get_memory():
    embeddings = OpenAIEmbeddings()
    try:
        if os.path.exists(os.path.join(VECTOR_DIR, "index.faiss")):
            return FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
    except (EOFError, ValueError, OSError):
        # corrupt or incomplete file — optionally delete it
        os.remove(os.path.join(VECTOR_DIR, "index.faiss"))
        if os.path.exists(os.path.join(VECTOR_DIR, "index.pkl")):
            os.remove(os.path.join(VECTOR_DIR, "index.pkl"))
        return None
    return None

def save_memory(memory):
    memory.save_local(VECTOR_DIR)

def lookup_summary(url: str):
    memory = get_memory()
    if memory is None:
        return None

    docs = memory.similarity_search(url, k=1)
    if docs and docs[0].metadata.get("url") == url:
        content = docs[0].page_content
        try:
            return json.loads(content)
        except:
            return {"summary": content, "source": "cached"}
    return None

def store_summary(url: str, data: dict):
    embeddings = OpenAIEmbeddings()
    summary_str = json.dumps(data)
    meta = {"url": url}

    memory = get_memory()
    if memory:
        memory.add_texts([summary_str], metadatas=[meta])
    else:
        memory = FAISS.from_texts([summary_str], embedding=embeddings, metadatas=[meta])

    save_memory(memory)

def list_cached_urls():
    memory = get_memory()
    if memory is None:
        return []
    return memory.get()["metadatas"]

def get_all_documents():
    memory = get_memory()
    if memory is None:
        return []

    # We'll simulate retrieval using dummy queries
    # FAISS does not support direct access to "all documents"
    try:
        docs = memory.similarity_search(" ", k=1000)  # broad query
        return docs
    except Exception:
        return []
