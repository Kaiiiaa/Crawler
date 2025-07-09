import os
import hashlib
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from tqdm import tqdm

# --- Load environment variables ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

INPUT_FOLDER = "input"
PDF_FILES = [
    os.path.join(INPUT_FOLDER, f)
    for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith(".pdf")
]

TEXT_KB_PATH = "music_knowledge_base.txt"
VECTORSTORE_DIR = "vectorstore"
SUMMARY_CACHE_FILE = "summary_cache.json"

# --- Initialize embeddings and summarization chain ---
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
summarize_chain = load_summarize_chain(llm, chain_type="map_reduce")

# --- Load or initialize cache ---
if os.path.exists(SUMMARY_CACHE_FILE):
    with open(SUMMARY_CACHE_FILE, "r", encoding="utf-8") as f:
        summary_cache = json.load(f)
else:
    summary_cache = {}

def hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

# --- Summarize with caching ---
def compress_documents(docs, doc_type="pdf"):
    compressed = []
    for doc in tqdm(docs, desc=f"Summarizing {doc_type}"):
        hash_key = hash_text(doc.page_content)
        if hash_key in summary_cache:
            summary = summary_cache[hash_key]
        else:
            try:
                summary = summarize_chain.run([doc])
                summary_cache[hash_key] = summary
            except Exception as e:
                print(f"❌ Summarization failed: {e}")
                continue

        compressed.append(Document(
            page_content=summary,
            metadata={
                "source": doc.metadata.get("source", "unknown"),
                "type": doc_type,
                "original": doc.page_content[:500] + "... (truncated)"
            }
        ))
    print(f"✅ Compressed {len(compressed)} documents ({doc_type}).")
    return compressed

# --- Load and split PDFs ---
def load_and_split_pdfs(pdf_paths):
    all_docs = []
    for pdf_path in pdf_paths:
        print(f"📄 Loading: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        docs = splitter.split_documents(pages)
        for d in docs:
            d.metadata["source"] = pdf_path
        print(f"✅ {pdf_path} split into {len(docs)} chunks.")
        all_docs.extend(docs)
    return all_docs

# --- Build vectorstore ---
def build_vectorstore(all_docs, persist_dir=VECTORSTORE_DIR):
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)

    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print(f"✅ Vectorstore built and saved to '{persist_dir}'.")

# --- Main execution ---
if __name__ == "__main__":
    # Load raw data
    pdf_docs = load_and_split_pdfs(PDF_FILES)

    # Compress with RAPTOR-style summarization
    compressed_pdfs = compress_documents(pdf_docs, doc_type="pdf")

    # Save summaries to cache file
    with open(SUMMARY_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(summary_cache, f, indent=2)

    # Index summaries
    build_vectorstore(compressed_pdfs)
