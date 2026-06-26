# streamlit_app.py

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="centered"
)

st.title("🔬 AI Research Paper Assistant")
st.caption("Powered by RAG + LLaMA3 (Ollama) · 10 AI Research Papers")

# ── Load vectorstore & model (cached) ────────────────────────
@st.cache_resource
def load_resources():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = Ollama(model="llama3")
    return retriever, llm

retriever, llm = load_resources()

# ── Chat history ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input ─────────────────────────────────────────────────────
if query := st.chat_input("Ask anything about the research papers..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            docs = retriever.invoke(query)
            context = "\n\n".join([doc.page_content for doc in docs])

            prompt = f"""
You are a research assistant.
Use context as primary source but allow general reasoning.
Use the context to answer.
If partial information is available, try your best.
Only say "Not found" if completely unrelated.

Context:
{context}

Question:
{query}
"""
            response = llm.invoke(prompt)

        st.markdown(response)

        # Show sources in expander
        with st.expander("📄 Source Chunks Used"):
            for i, doc in enumerate(docs):
                source = doc.metadata.get("source", "Unknown")
                st.markdown(f"**Chunk {i+1}** — `{source}`")
                st.caption(doc.page_content[:300] + "...")

    st.session_state.messages.append({"role": "assistant", "content": response})