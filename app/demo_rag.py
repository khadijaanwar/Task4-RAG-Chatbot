# task4/app/demo_rag.py
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

# Load environment variables from .env file
load_dotenv()
HF_TOKEN = os.getenv("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")

if not HF_TOKEN:
    raise ValueError(
        "❌ Missing HuggingFace API token. Please set HUGGINGFACEHUB_API_TOKEN in a .env file or environment variable."
    )

# Get the base folder (where this file is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def build_index(data_path=None, index_path=None):
    """
    Loads .txt files, splits them, creates embeddings, and stores a FAISS index.
    Automatically resolves paths relative to this file.
    """
    if data_path is None:
        data_path = os.path.join(BASE_DIR, "..", "data")
    if index_path is None:
        index_path = os.path.join(BASE_DIR, "..", "index_store")

    data_path = os.path.abspath(data_path)
    index_path = os.path.abspath(index_path)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    print(f"🔍 Loading documents from: {data_path}")
    loader = DirectoryLoader(data_path, glob="*.txt", loader_cls=TextLoader)
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    print(f"🧠 Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print(f"📦 Building FAISS index...")
    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local(index_path)

    print(f"✅ Index built and saved to: {index_path}")


def query(question, index_path=None):
    """
    Loads FAISS index and queries it using an LLM.
    """
    if index_path is None:
        index_path = os.path.join(BASE_DIR, "..", "index_store")
    index_path = os.path.abspath(index_path)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        huggingfacehub_api_token=HF_TOKEN,  # Pass token here
        model_kwargs={"temperature": 0, "max_length": 512}
    )

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())
    return qa.run(question)
