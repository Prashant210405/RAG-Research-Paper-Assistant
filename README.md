# 🔬 AI Research Paper Assistant

A Retrieval-Augmented Generation (RAG) based assistant that answers questions from AI research papers using LLaMA3 and HuggingFace embeddings.

## 🚀 Demo

> Ask anything about the research papers and get precise, context-aware answers powered by LLaMA3.

![RAG Pipeline](https://img.shields.io/badge/RAG-Pipeline-blue) ![LLaMA3](https://img.shields.io/badge/LLaMA3-Ollama-green) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

---

## 🧠 How It Works
1. **Load** — 10 AI research papers loaded via `DirectoryLoader`
2. **Chunk** — Split into 800-token chunks with 150 overlap
3. **Embed** — `BAAI/bge-small-en-v1.5` HuggingFace embeddings
4. **Store** — ChromaDB vector store for fast retrieval
5. **Retrieve** — Top 5 relevant chunks fetched per query
6. **Answer** — LLaMA3 (via Ollama) generates precise answers

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | LLaMA3 via Ollama |
| Embeddings | BAAI/bge-small-en-v1.5 (HuggingFace) |
| Vector Store | ChromaDB |
| Framework | LangChain |
| UI | Streamlit |

---

## ⚙️ Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/Prashant210405/RAG-Research-Paper-Assistant.git
cd rag-research-assistant
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Install & run Ollama
```bash
# Install Ollama from https://ollama.com
ollama pull llama3
```

### 4. Add your research papers
### 5. Build the vector store
```bash
python retriever.py
```

### 6. Run the app
```bash
streamlit run streamlit_app.py
```

---

## 📁 Project Structure
---

## 💡 Features

- 💬 Chat-style interface with message history
- 📄 Source chunk viewer — see exactly where the answer came from
- ⚡ Cached model loading for fast performance
- 🔍 Top-5 semantic retrieval per query

---

## 👨‍💻 Author

**Prashant Gour**  
AI & Data Science | SAGE University, Bhopal  
[LinkedIn](https://linkedin.com/in/prashantgour21/) · [GitHub](https://github.com/Prashant210405)