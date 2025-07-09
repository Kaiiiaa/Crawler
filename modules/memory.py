# modules/memory.py
import json
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

VECTOR_DIR = "memory_store"

def get_memory():
    return Chroma(
        persist_directory=VECTOR_DIR,
        collection_name="agent_memory",
        embedding_function=OpenAIEmbeddings()
    )

def lookup_summary(url: str):
    memory = get_memory()
    docs = memory.similarity_search(url, k=1)
    if docs and docs[0].metadata.get("url") == url:
        content = docs[0].page_content
        try:
            return json.loads(content)
        except:
            return {"summary": content, "source": "cached"}
    return None

def store_summary(url: str, data: dict):
    memory = get_memory()
    summary_str = json.dumps(data)
    memory.add_texts([summary_str], metadatas=[{"url": url}])

def list_cached_urls():
    memory = get_memory()
    return memory.get()["metadatas"]

def get_all_documents():
    memory = get_memory()
    return memory.get(include=["metadatas", "documents"])