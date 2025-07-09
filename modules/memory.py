# modules/memory.py
import os
import json
import pickle
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

VECTOR_DIR = "memory_store"
INDEX_FILE = os.path.join(VECTOR_DIR, "faiss_memory.pkl")

def get_memory():
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "rb") as f:
            return pickle.load(f)
    return None

def save_memory(memory):
    os.makedirs(VECTOR_DIR, exist_ok=True)
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(memory, f)

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
    return memory.get(include=["metadatas", "documents"])
