from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# """"""""""""DocumentLoader""""""""""""
loader = DirectoryLoader(
    "papers",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)

documents = loader.load()

# """""""""""""TextSplitter""""""""""""
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],  # explicit priority
    length_function=len,
)

chunks = splitter.split_documents(documents)
# print(f"Original pages: {len(documents)}")
# print(f"Total chunks: {len(chunks)}")
# print(chunks[0])


# """"""""""""""Embeddings""""""""""""
embeddings = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-small-en-v1.5"
)
# """""""""vectorstore""""""""""""
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)


# """"""""""""Retrivers""""""""""""

