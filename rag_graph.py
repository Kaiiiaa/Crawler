# rag_graph.py
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import Chroma
import requests
from bs4 import BeautifulSoup

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from typing import TypedDict, Optional, List

class RAGState(TypedDict):
    url: str
    html: Optional[str]
    status: Optional[int]
    redirects: Optional[int]
    inspection_notes: Optional[List[str]]
    context: Optional[str]
    summary: Optional[str]
    saved: Optional[bool]
    error: Optional[str]


# Step 1: Fetch the page
def fetch_page(state):
    url = state["url"]
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        html = resp.text[:4000]
        return {"html": html, "status": resp.status_code, "redirects": len(resp.history)}
    except Exception as e:
        return {"error": str(e), "html": "", "status": None, "redirects": 0}


# Step 2: Simple inspection (CAPTCHA, block checks)
def inspect_page(state):
    html = state.get("html", "").lower()
    issues = []

    if "captcha" in html:
        issues.append("🚧 CAPTCHA detected.")
    if "cloudflare" in html:
        issues.append("⚠️ Cloudflare block likely.")
    if state.get("redirects", 0) > 0:
        issues.append("🔁 Redirect chain detected.")
    if len(html) < 500:
        issues.append("❗ Very short HTML — possible soft block.")

    return {"inspection_notes": issues}


# Step 3: RAG context retrieval
def retrieve_rag_context(state):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory="vectorstore", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(state["html"])
    return {"context": "\n\n".join([doc.page_content for doc in docs])}


# Step 4: Final summary with LLM
def summarize_with_llm(state):
    context = state["context"]
    html = state["html"]
    notes = "\n".join(state.get("inspection_notes", []))

    prompt = f"""
You are a senior web scraping engineer. Here's a sample of the HTML content from a web page and inspection notes:

Inspection Findings:
{notes}

Reference Knowledge (retrieved):
{context}

HTML Sample:
{html}

Based on the above, explain what scraping challenges might exist on this site, and suggest next steps.
"""

    llm = ChatOpenAI(model="gpt-4", temperature=0)
    result = llm.invoke(prompt)
    return {"summary": result.content}


# Create LangGraph pipeline
def create_graph():
    builder = StateGraph(RAGState)

    builder.add_node("fetch", RunnableLambda(fetch_page))
    builder.add_node("inspect", RunnableLambda(inspect_page))
    builder.add_node("retrieve", RunnableLambda(retrieve_rag_context))
    builder.add_node("summarize", RunnableLambda(summarize_with_llm))
    builder.add_node("save", RunnableLambda(save_to_vectorstore))

    builder.set_entry_point("fetch")
    builder.add_edge("fetch", "inspect")
    builder.add_edge("inspect", "retrieve")
    builder.add_edge("retrieve", "summarize")
    builder.add_edge("summarize", "save")
    builder.add_edge("save", END)

    return builder.compile()

def save_to_vectorstore(state):
    from langchain_openai import OpenAIEmbeddings
    from langchain.vectorstores import Chroma

    summary = state["summary"]
    url = state["url"]
    notes = "\n".join(state.get("inspection_notes", []))
    status = state.get("status", "Unknown")
    html = state.get("html", "")[:2000]

    metainfo = {
        "url": url,
        "status": status,
        "notes": notes,
    }

    vectorstore = Chroma(
        persist_directory="inspections_store",
        collection_name="page_inspections",
        embedding_function=OpenAIEmbeddings()
    )

    vectorstore.add_texts([summary], metadatas=[metainfo])
    return {"saved": True}