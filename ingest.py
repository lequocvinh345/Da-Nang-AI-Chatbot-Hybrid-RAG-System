import os
import pickle
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

folder_path = r"D:\RAG-chatbot"
loader = DirectoryLoader(
    path = folder_path,
    glob = "*.pdf",
    show_progress=True,
    loader_cls = PyPDFLoader,
    use_multithreading=True

)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size =1200,
    chunk_overlap =200,
    separators=["\n\n", "\n", ".",  " ", ""]
)

chunks = text_splitter.split_documents(documents)

# ==========================================================
# Lưu chunks ra file .pkl
# ==========================================================

chunks_path = os.path.join(folder_path, "chunks.pkl")

with open(chunks_path, "wb") as f:
    pickle.dump(chunks, f)

print(f"Đã lưu chunks vào: {chunks_path}")


embeddings = OllamaEmbeddings(
    model = "bge-m3"
)

embeddings_dir = os.path.join(folder_path, "chroma_db")
print("Đang số hóa dữ liệu và lưu vào ChromaDB (Dense Retrieval)...")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=embeddings_dir
)

vector_retriever = vectorstore.as_retriever(
    search_kwargs = {"k": 3}  # Khi search thì lấy ra 3 kết quả gần nhất theo vector similarity
)

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3 # Khi search lấy ra 3 kết quả gần nhất theo BM25

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)

print("Hoàn thành")




   
    
  
