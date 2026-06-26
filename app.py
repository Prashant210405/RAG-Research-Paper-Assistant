from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.llms import Ollama
import os


vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )
)


retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)

# query = "What is BERT?"

# docs = retriever.invoke(query)

# print(f"Total docs: {len(docs)}")

# for doc in docs:
#     print("\n---")
#     print(doc.page_content[:500])








# ✅ Load API key from .env
llm = Ollama(model="llama3")

while True:
    query = input("\nAsk question (or type 'exit'): ")

    if query.lower() == "exit":
        break

    # Step 1: retrieve relevant chunks
    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    # Step 2: ask LLM
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

    print("\n🤖 Answer:")
    print(response)