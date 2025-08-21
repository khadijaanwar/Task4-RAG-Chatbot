# Task 4 - Context-Aware Chatbot with LangChain & RAG
## 🎯 Objective
Develop a **conversational chatbot** capable of remembering context and retrieving external knowledge using **LangChain / RAG (Retrieval-Augmented Generation)**.

## 🛠️ Methodology / Approach
- Built conversational memory using LangChain’s **memory modules**.  
- Indexed external knowledge sources (Wikipedia/Custom Docs) into a **vector store**.  
- Implemented **document embeddings** for semantic retrieval.  
- Integrated RAG pipeline to generate context-aware answers.  
- Deployed chatbot via **Streamlit**.  

## 📊 Key Results / Observations
- Chatbot successfully retrieved relevant answers from knowledge base.  
- Maintained **context across turns** for natural conversations.  
- Demonstrated scalable architecture for domain-specific assistants.  


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
