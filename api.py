import os
import pickle
import requests

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever


# ==========================================================
# 1. FASTAPI
# ==========================================================

app = FastAPI(title="RAG API (No Agent)", version="3.1")


# ==========================================================
# 2. LOAD DATA
# ==========================================================

FOLDER_PATH = r"D:\RAG-chatbot"
CHROMA_DB = os.path.join(FOLDER_PATH, "chroma_db")
CHUNKS_PATH = os.path.join(FOLDER_PATH, "chunks.pkl")

with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)


# ==========================================================
# 3. EMBEDDING
# ==========================================================

embeddings = OllamaEmbeddings(model="bge-m3")


# ==========================================================
# 4. VECTOR DB
# ==========================================================

vectorstore = Chroma(
    persist_directory=CHROMA_DB,
    embedding_function=embeddings
)

vector_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 6,
        "lambda_mult": 0.7
    }
)


# ==========================================================
# 5. BM25 + HYBRID
# ==========================================================

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.45, 0.55]
)


# ==========================================================
# 6. LLM
# ==========================================================

llm = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0
)


# ==========================================================
# 7. TOOLS
# ==========================================================

def search_documents(query: str) -> str:
    try:
        docs = ensemble_retriever.invoke(query)
    except Exception as e:
        return "Lỗi retriever (ensemble): " + str(e)

    if not docs:
        return "Không tìm thấy thông tin liên quan."

    out = []

    for i, d in enumerate(docs[:3], 1):
        try:
            text = (d.page_content or "").strip().replace("\n", " ")
        except Exception:
            text = ""

        if not text:
            continue

        out.append(f"[Tài liệu {i}] {text[:500]}")

    return "\n\n".join(out) if out else "Không có dữ liệu hợp lệ."


def weather_da_nang() -> str:
    API_KEY = "41b12eb6713ce73eda67f7db05961746"

    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?q=Da Nang,VN&appid={API_KEY}&units=metric&lang=vi"
    )

    try:
        r = requests.get(url, timeout=5)

        try:
            data = r.json()
        except Exception:
            return "Lỗi parse dữ liệu thời tiết."

        if r.status_code != 200:
            return "Không lấy được dữ liệu thời tiết."

        weather = data.get("weather", [{}])[0].get("description", "không rõ")

        main = data.get("main", {})
        wind = data.get("wind", {})

        return (
            "Thời tiết Đà Nẵng:\n"
            f"- {weather}\n"
            f"- Nhiệt độ: {main.get('temp', 'N/A')}°C\n"
            f"- Độ ẩm: {main.get('humidity', 'N/A')}%\n"
            f"- Gió: {wind.get('speed', 'N/A')} m/s"
        )

    except Exception:
        return "Lỗi kết nối API thời tiết."


# ==========================================================
# 8. ROUTER
# ==========================================================

def route(query: str):
    q = query.lower()

    weather_kw = {
        "thời tiết", "nhiệt độ", "mưa",
        "nắng", "gió", "độ ẩm", "forecast"
    }

    if any(k in q for k in weather_kw):
        return "weather"

    return "rag"


# ==========================================================
# 9. REQUEST MODEL
# ==========================================================

class ChatRequest(BaseModel):
    question: str


# ==========================================================
# 10. CHAT API
# ==========================================================

@app.post("/chat")
def chat(req: ChatRequest):

    query = (req.question or "").strip()

    if not query:
        return {"answer": "Vui lòng nhập câu hỏi"}

    tool = route(query)

    # ===== TOOL CALL =====
    if tool == "weather":
        context = weather_da_nang()
    else:
        context = search_documents(query)

    # ======================================================
    # PROMPT + LLM INVOKE
    # ======================================================

    prompt = f"""
Bạn là hệ thống QA RAG.

QUY TẮC BẮT BUỘC:
- Chỉ sử dụng thông tin trong CONTEXT
- TUYỆT ĐỐI không được bịa thêm hoặc suy đoán
- Nếu CONTEXT không có thông tin liên quan → trả lời: "Không có dữ liệu trong hệ thống"
- Không sử dụng kiến thức bên ngoài dù bạn biết

CÁCH TRẢ LỜI:
- Trả lời đầy đủ các ý quan trọng trong CONTEXT
- Nhưng phải ngắn gọn, rõ ràng, không lan man
- Ưu tiên dạng liệt kê, xuống dòng nếu có nhiều ý
- Không giải thích thêm ngoài dữ liệu có sẵn

CONTEXT:
{context}

CÂU HỎI:
{query}

TRẢ LỜI:
"""

    try:
        answer = llm.invoke(prompt).content
    except Exception as e:
        return {
            "question": query,
            "tool": tool,
            "answer": "Lỗi LLM: " + str(e)
        }

    return {
        "question": query,
        "tool": tool,
        "answer": answer
    }


# ==========================================================
# 11. RUN
# ==========================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)