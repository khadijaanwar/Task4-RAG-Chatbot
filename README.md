# Task 4 - Retrieval-Augmented Generation (RAG) Demo

## 📦 Setup
```bash
pip install streamlit langchain langchain-community sentence-transformers faiss-cpu huggingface_hub
```

## ▶ Run the App
```bash
cd task4/app
streamlit run app_streamlit.py
```

## 📁 Folder Structure
- `app/` → Contains the main Streamlit UI (`app_streamlit.py`) and RAG helper (`demo_rag.py`)
- `data/` → Contains `.txt` documents for indexing
- `index_store/` → Generated FAISS index (created after building index)
