import os
import pickle
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


# ==========================================================
# 1. Load dữ liệu và tạo Retriever
# ==========================================================

folder_path = r"D:\RAG-chatbot"
embeddings_dir = os.path.join(folder_path, "chroma_db")



chunks_path = os.path.join(folder_path, "chunks.pkl")

print("Đang load chunks...")
with open(chunks_path, "rb") as f:
    chunks = pickle.load(f)

print("Load chunks thành công!")


# ==========================================================
# 2. Load Embedding Model
# ==========================================================

embeddings = OllamaEmbeddings(
    model="bge-m3"
)


# ==========================================================
# 3. Load Vector Database
# ==========================================================

vectorstore = Chroma(
    persist_directory=embeddings_dir,
    embedding_function=embeddings
)

vector_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# ==========================================================
# 4. BM25 Retriever
# ==========================================================

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3


# ==========================================================
# 5. Hybrid Search (BM25 + Dense Retrieval)
# ==========================================================

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)


# ==========================================================
# 6. Load LLM
# ==========================================================

llm = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0
)


# ==========================================================
# 7. Prompt Template
# ==========================================================

prompt = ChatPromptTemplate.from_template(
    """
Bạn là một trợ lý AI trả lời câu hỏi dựa trên tài liệu được cung cấp.

Chỉ sử dụng thông tin trong phần CONTEXT để trả lời.
Nếu trong CONTEXT không có thông tin, hãy trả lời:
"Tôi không tìm thấy thông tin phù hợp trong tài liệu."

------------------------
CONTEXT:
{context}
------------------------

QUESTION:
{question}

ANSWER:
"""
)


# ==========================================================
# 8. Chat Loop
# ==========================================================

print("=" * 60)
print("        RAG CHATBOT - Gõ 'exit' để thoát")
print("=" * 60)

while True:

    # ------------------------------------------------------
    # [Query]
    # ------------------------------------------------------
    query = input("\nBạn: ")

    if query.lower() == "exit":
        print("Kết thúc chương trình.")
        break


    # ------------------------------------------------------
    # [7. Pre-retrieval]
    # (Ở bản đơn giản chỉ chuẩn hóa câu hỏi)
    # ------------------------------------------------------
    query = query.strip()


    # ------------------------------------------------------
    # [8. Retrieval]
    # (Hybrid Search)
    # ------------------------------------------------------
    retrieved_docs = ensemble_retriever.invoke(query)


    # ------------------------------------------------------
    # [9. Post-retrieval]
    # (Ghép các chunk thành context)
    # ------------------------------------------------------
    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )


    # ------------------------------------------------------
    # [10. Prompt]
    # ------------------------------------------------------
    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": query
        }
    )


    # ------------------------------------------------------
    # [11. Generation]
    # ------------------------------------------------------
    response = llm.invoke(final_prompt)


    # ------------------------------------------------------
    # Output
    # ------------------------------------------------------
    print("\nChatbot:")
    print(response.content)