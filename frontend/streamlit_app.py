"""
Streamlit UI — фронтенд чатбота.
Общается с FastAPI бэкендом через REST API.
"""

import streamlit as st
import requests
import time

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Document RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

# ── Стили ──────────────────────────────────────────
st.markdown("""
<style>
    .source-badge {
        background: #e8f4f8;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 13px;
        margin: 2px;
        display: inline-block;
        color: #1a5276;
    }
    .meta-info {
        font-size: 12px;
        color: #888;
        margin-top: 4px;
    }
    .rewrite-box {
        background: #f8f9fa;
        border-left: 3px solid #6c757d;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 13px;
        color: #555;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Заголовок ──────────────────────────────────────
st.title("🤖 Document RAG Chatbot")
st.caption("Claude Sonnet 4.5 + Pinecone | Семантический поиск по документам")

# ── Боковая панель ─────────────────────────────────
with st.sidebar:
    st.header("📄 Загрузка документов")
    uploaded = st.file_uploader("Загрузите PDF", type="pdf", accept_multiple_files=True)

    if uploaded and st.button("Индексировать", type="primary"):
        for f in uploaded:
            with st.spinner(f"Обработка {f.name}..."):
                resp = requests.post(
                    f"{API_URL}/ingest",
                    files={"file": (f.name, f.getvalue(), "application/pdf")},
                )
                if resp.ok:
                    st.success(resp.json()["message"])
                else:
                    st.error(f"Ошибка: {resp.text}")

    st.divider()
    st.header("⚙️ Настройки")
    top_k = st.slider("Количество чанков (top_k)", 2, 8, 4)
    show_meta = st.toggle("Показывать мета-информацию", value=True)

    st.divider()
    try:
        stats = requests.get(f"{API_URL}/stats", timeout=2).json()
        st.metric("Документов в индексе", stats.get("total_vectors", "—"))
    except Exception:
        st.warning("Бэкенд недоступен")

    if st.button("Очистить чат"):
        st.session_state.messages = []
        st.rerun()

# ── История чата ───────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_meta and "meta" in msg:
            meta = msg["meta"]
            if meta.get("rewritten_query"):
                st.markdown(
                    f'<div class="rewrite-box">🔍 Переформулировано: <i>{meta["rewritten_query"]}</i></div>',
                    unsafe_allow_html=True,
                )
            if meta.get("sources"):
                st.markdown(
                    " ".join(f'<span class="source-badge">📎 {s}</span>' for s in meta["sources"]),
                    unsafe_allow_html=True,
                )
            st.markdown(
                f'<div class="meta-info">⏱ {meta.get("response_time_ms", "?")} мс</div>',
                unsafe_allow_html=True,
            )

# ── Ввод вопроса ───────────────────────────────────
if question := st.chat_input("Задайте вопрос по документам..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Ищу в документах..."):
            try:
                resp = requests.post(
                    f"{API_URL}/query",
                    json={"question": question, "top_k": top_k},
                    timeout=30,
                )
                if resp.ok:
                    data = resp.json()
                    st.markdown(data["answer"])

                    if show_meta:
                        if data.get("rewritten_query"):
                            st.markdown(
                                f'<div class="rewrite-box">🔍 Переформулировано: <i>{data["rewritten_query"]}</i></div>',
                                unsafe_allow_html=True,
                            )
                        if data.get("sources"):
                            st.markdown(
                                " ".join(
                                    f'<span class="source-badge">📎 {s}</span>'
                                    for s in data["sources"]
                                ),
                                unsafe_allow_html=True,
                            )
                        st.markdown(
                            f'<div class="meta-info">⏱ {data.get("response_time_ms", "?")} мс</div>',
                            unsafe_allow_html=True,
                        )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "meta": {
                            "rewritten_query": data.get("rewritten_query"),
                            "sources": data.get("sources", []),
                            "response_time_ms": data.get("response_time_ms"),
                        },
                    })
                else:
                    st.error(f"Ошибка сервера: {resp.text}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Бэкенд недоступен. Запустите: `uvicorn app.main:app --reload`")
            except Exception as e:
                st.error(f"Ошибка: {e}")
