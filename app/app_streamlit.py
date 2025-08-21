# task4/app/app_streamlit.py
import streamlit as st
from demo_rag import build_index, query

st.set_page_config(page_title="RAG Demo", layout="centered")

st.title("📚 Retrieval-Augmented Generation (Task 4)")
st.write("Ask questions based on the documents in the `data` folder.")

# Step 1: Build index
if st.button("🔄 Build / Rebuild Index"):
    with st.spinner("Building index..."):
        build_index(data_path="../data", index_path="../index_store")
    st.success("Index built successfully!")

# Step 2: Ask a question
question = st.text_input("Enter your question:")

if st.button("Ask"):
    if question.strip():
        with st.spinner("Searching and generating answer..."):
            answer = query(question, index_path="../index_store")
        st.write("### 💡 Answer:")
        st.success(answer)
    else:
        st.warning("Please enter a question.")

st.markdown("---")
st.caption("Task 4 - DevelopersHub Internship")
