import requests
import streamlit as st


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Đà Nẵng AI Chat",
    page_icon="🌴",
    layout="wide"
)


# ==========================================================
# CUSTOM CSS (UI ĐẸP HƠN)
# ==========================================================

st.markdown(
    """
    <style>

    /* nền tổng */
    .stApp {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: white;
    }

    /* chat container */
    .block-container {
        padding-top: 2rem;
        max-width: 900px;
    }

    /* user bubble */
    .user-msg {
        background: #2563eb;
        padding: 12px 16px;
        border-radius: 15px;
        margin: 8px 0;
        color: white;
        width: fit-content;
        max-width: 80%;
        margin-left: auto;
    }

    /* assistant bubble */
    .bot-msg {
        background: #1f2937;
        padding: 12px 16px;
        border-radius: 15px;
        margin: 8px 0;
        color: white;
        width: fit-content;
        max-width: 80%;
    }

    /* input box */
    .stChatInput input {
        border-radius: 12px;
    }

    /* title */
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        opacity: 0.7;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# HEADER
# ==========================================================

st.markdown("<div class='title'>🌴 ĐÀ NẴNG AI CHAT</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>RAG System • BM25 + Chroma • FastAPI • Qwen</div>", unsafe_allow_html=True)


# ==========================================================
# SESSION STATE
# ==========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.title("⚙️ Hệ thống")

    st.write("**Backend:** FastAPI")
    st.write("**RAG:** BM25 + ChromaDB")
    st.write("**LLM:** Qwen 2.5 (Ollama)")
    st.write("**Embedding:** bge-m3")

    st.divider()

    if st.button("🗑️ Xóa chat"):
        st.session_state.messages = []
        st.rerun()


# ==========================================================
# CHAT HISTORY
# ==========================================================

for msg in st.session_state.messages:

    if msg["role"] == "user":
        st.markdown(
            f"<div class='user-msg'>{msg['content']}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='bot-msg'>{msg['content']}</div>",
            unsafe_allow_html=True
        )


# ==========================================================
# INPUT
# ==========================================================

question = st.chat_input("Nhập câu hỏi về Đà Nẵng hoặc tài liệu...")

if question:

    # user msg
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    st.markdown(
        f"<div class='user-msg'>{question}</div>",
        unsafe_allow_html=True
    )

    # call API
    with st.spinner("🤖 AI đang suy nghĩ..."):

        try:
            response = requests.post(
                "http://127.0.0.1:8000/chat",
                json={"question": question},
                timeout=300
            )

            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "Không có câu trả lời")
            else:
                answer = f"Lỗi API: {response.status_code}"

        except Exception as e:
            answer = f"Lỗi kết nối backend: {str(e)}"

    # bot msg
    st.markdown(
        f"<div class='bot-msg'>{answer}</div>",
        unsafe_allow_html=True
    )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )