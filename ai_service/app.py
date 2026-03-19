# app.py — Streamlit чат для Legal RAG (RU/KZ, дисклеймер, прозрачность AI-закон РК)

import json
import uuid
from datetime import datetime

import streamlit as st

from ai_service.core import config
from ai_service.retrieval.rag_chain import invoke_qa, validate_answer

CHAT_STORE_PATH = "chat_history.json"


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _load_chats() -> dict:
    try:
        with open(CHAT_STORE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"chats": {}, "order": []}


def _save_chats(store: dict) -> None:
    with open(CHAT_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)


def _new_chat(store: dict, title: str | None = None) -> str:
    chat_id = str(uuid.uuid4())
    title = title or "Новый чат"
    store["chats"][chat_id] = {
        "id": chat_id,
        "title": title,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "messages": [],
    }
    store["order"].insert(0, chat_id)
    return chat_id


def _rename_chat(store: dict, chat_id: str, title: str) -> None:
    if chat_id in store["chats"]:
        store["chats"][chat_id]["title"] = title.strip() or "Без названия"
        store["chats"][chat_id]["updated_at"] = _now_iso()


def _delete_chat(store: dict, chat_id: str) -> None:
    if chat_id in store["chats"]:
        store["chats"].pop(chat_id, None)
        store["order"] = [cid for cid in store["order"] if cid != chat_id]


def _title_from_prompt(prompt: str) -> str:
    cleaned = " ".join(prompt.strip().split())
    if not cleaned:
        return "Новый чат"
    words = cleaned.split(" ")
    return " ".join(words[:4]).strip()


# Настройки страницы (должно быть первым вызовом Streamlit)
st.set_page_config(
    page_title="Legal RAG — Помощник по законам РК",
    page_icon="⚖️",
    layout="wide",
)

# Язык интерфейса
if "lang" not in st.session_state:
    st.session_state.lang = "ru"
if "chat_store" not in st.session_state:
    st.session_state.chat_store = _load_chats()
if "current_chat_id" not in st.session_state:
    if not st.session_state.chat_store["order"]:
        st.session_state.current_chat_id = _new_chat(st.session_state.chat_store)
        _save_chats(st.session_state.chat_store)
    else:
        st.session_state.current_chat_id = st.session_state.chat_store["order"][0]

DISCLAIMERS = {"ru": config.DISCLAIMER_RU, "kz": config.DISCLAIMER_KZ}
PLACEHOLDERS = {
    "ru": "Задайте вопрос по законам РК (на русском или казахском)",
    "kz": "ҚР заңдары бойынша сұрақ қойыңыз (орыс немесе қазақ тілінде)",
}
SOURCES_LABEL = {
    "ru": "**Источники (реальные статьи из базы):**",
    "kz": "**Дереккөздер (базадағы мақалалар):**",
}
DOWNLOAD_LABEL = {"ru": "Скачать ответ как TXT", "kz": "Жауапты TXT ретінде жүктеу"}
CLEAR_CHAT = {"ru": "Очистить чат", "kz": "Чатты тазалау"}
NEW_CHAT = {"ru": "Новый чат", "kz": "Жаңа чат"}
DELETE_CHAT = {"ru": "Удалить чат", "kz": "Чатты жою"}
RENAME_CHAT = {"ru": "Переименовать", "kz": "Атауын өзгерту"}
SAVE_CHAT = {"ru": "Сохранить историю", "kz": "Тарихты сақтау"}

# Базовый стиль
st.markdown(
    """
<style>
@import url("https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600;700&family=Space+Grotesk:wght@400;500;600&family=IBM+Plex+Mono:wght@400;600&display=swap");

:root {
  --ink-1: #0c1117;
  --ink-2: #1f2a37;
  --ink-3: #6b7280;
  --paper: #f1efe8;
  --paper-2: #f7f6f2;
  --accent: #c2410c;
  --accent-2: #0f766e;
  --accent-soft: rgba(194,65,12,0.12);
  --border: rgba(15,23,42,0.12);
}

html, body, [class*="stApp"] { font-family: "Space Grotesk", sans-serif; color: var(--ink-2); }
.stApp h1, .stApp h2, .stApp h3 { font-family: "Fraunces", serif; color: var(--ink-1); letter-spacing: 0.2px; }
.stApp {
  background:
    radial-gradient(900px 540px at 6% -10%, rgba(194,65,12,0.2), transparent 60%),
    radial-gradient(820px 520px at 92% 0%, rgba(15,118,110,0.18), transparent 55%),
    radial-gradient(680px 420px at 50% 110%, rgba(15,23,42,0.08), transparent 60%),
    linear-gradient(180deg, var(--paper) 0%, var(--paper-2) 100%);
}
section[data-testid="stSidebar"] {
  background:
    radial-gradient(360px 220px at 8% 0%, rgba(194,65,12,0.08), transparent 60%),
    linear-gradient(180deg, #f4f2ec 0%, #f8f6f1 100%);
  border-right: 1px solid rgba(15,23,42,0.08);
}
section[data-testid="stSidebar"] * { color: var(--ink-2) !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: var(--ink-1) !important; }

.title-bar {
  display: flex; flex-direction: column; gap: 0.4rem; margin-bottom: 0.75rem;
}
.title-row { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
.title-badge {
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.7rem; letter-spacing: 0.18em; text-transform: uppercase;
  color: #0b1220; background: #fde68a; padding: 6px 12px; border-radius: 999px;
}
.title-sub {
  color: var(--ink-3); font-size: 1rem; margin: 0; max-width: 52rem;
}

.chat-shell {
  background: rgba(255,255,255,0.85);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 16px;
  box-shadow: 0 18px 40px rgba(15,23,42,0.1);
  backdrop-filter: blur(6px);
  max-width: 920px;
  margin: 0 auto;
}
.stChatMessage { border-radius: 14px; }
.stChatMessage[data-testid="stChatMessage"] { border: 1px solid rgba(15,23,42,0.08); max-width: 920px; margin-left: auto; margin-right: auto; }
.stChatMessage[data-testid="stChatMessage"] p { line-height: 1.5; }
.stMarkdown, .stMarkdown * { color: var(--ink-2); }
.stCaption { color: var(--ink-3); }

.sources-title {
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.8rem;
  color: var(--ink-2);
  margin-top: 0.75rem;
}
.sources-footer {
  margin-top: 0.75rem;
  padding: 10px 12px;
  border-radius: 12px;
  background: rgba(15,23,42,0.04);
  border: 1px solid rgba(15,23,42,0.08);
}
.source-item { margin-top: 8px; }
.source-meta {
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.78rem;
  color: var(--ink-2);
}
.source-quote {
  margin-top: 6px;
  padding: 10px 12px;
  border-left: 3px solid rgba(15,118,110,0.5);
  background: rgba(15,118,110,0.08);
  border-radius: 10px;
  color: var(--ink-2);
}
.disclaimer-pill {
  margin: 10px 0 6px;
  padding: 8px 12px;
  border-radius: 12px;
  background: rgba(194,65,12,0.08);
  border: 1px solid rgba(194,65,12,0.2);
  color: var(--ink-2);
  font-size: 0.92rem;
}
.content-wrap { max-width: 980px; margin: 0 auto; }
.stChatInputContainer textarea {
  border-radius: 16px !important;
  border: 1px solid rgba(15,23,42,0.18) !important;
  padding: 14px 16px !important;
}
.stChatInputContainer textarea:focus {
  border-color: rgba(194,65,12,0.6) !important;
  box-shadow: 0 0 0 3px rgba(194,65,12,0.12) !important;
}
.stChatInputContainer {
  padding: 8px 0 18px !important;
}
button[data-testid="stChatInputSubmitButton"] {
  background: var(--accent) !important;
  border: none !important;
  color: #fff !important;
  box-shadow: 0 10px 24px rgba(194,65,12,0.2) !important;
}
button[data-testid="stChatInputSubmitButton"][aria-disabled="true"] {
  background: rgba(15,23,42,0.2) !important;
  box-shadow: none !important;
}

@media (max-width: 768px) {
  .chat-shell { padding: 10px; }
  .title-row h1 { font-size: 1.6rem; }
}
</style>
    """,
    unsafe_allow_html=True,
)

# Сайдбар: язык и настройки
with st.sidebar:
    st.header("Legal RAG")
    st.markdown(
        "Помощник по законам РК"
        if st.session_state.lang == "ru"
        else "ҚР заңдары бойынша көмекші"
    )
    if st.button(f"+ {NEW_CHAT[st.session_state.lang]}", key="new_chat"):
        new_id = _new_chat(st.session_state.chat_store)
        st.session_state.current_chat_id = new_id
        _save_chats(st.session_state.chat_store)
        st.rerun()

    # Список чатов
    chat_titles = [
        (cid, st.session_state.chat_store["chats"][cid]["title"])
        for cid in st.session_state.chat_store["order"]
        if cid in st.session_state.chat_store["chats"]
    ]
    if chat_titles:
        selected = st.selectbox(
            "Чаты",
            options=[cid for cid, _ in chat_titles],
            format_func=lambda cid: next(t for c, t in chat_titles if c == cid),
            index=0,
            key="chat_selector",
        )
        st.session_state.current_chat_id = selected

    new_title = st.text_input(
        RENAME_CHAT[st.session_state.lang], value="", key="rename_input"
    )
    if (
        st.button(RENAME_CHAT[st.session_state.lang] + " ✓", key="rename_btn")
        and new_title.strip()
    ):
        _rename_chat(
            st.session_state.chat_store, st.session_state.current_chat_id, new_title
        )
        _save_chats(st.session_state.chat_store)
        st.rerun()

    if st.button(DELETE_CHAT[st.session_state.lang], key="delete_chat"):
        current_id = st.session_state.current_chat_id
        _delete_chat(st.session_state.chat_store, current_id)
        if not st.session_state.chat_store["order"]:
            st.session_state.current_chat_id = _new_chat(st.session_state.chat_store)
        else:
            st.session_state.current_chat_id = st.session_state.chat_store["order"][0]
        _save_chats(st.session_state.chat_store)
        st.rerun()

    with st.expander("Настройки", expanded=False):
        lang_toggle = st.toggle(
            "Қазақша", value=st.session_state.lang == "kz", key="lang_toggle"
        )
        st.session_state.lang = "kz" if lang_toggle else "ru"

# История чата
current_chat = st.session_state.chat_store["chats"][st.session_state.current_chat_id]
messages = current_chat["messages"]

# Заголовок и дисклеймер (на выбранном языке)
st.markdown('<div class="content-wrap">', unsafe_allow_html=True)
if messages:
    st.markdown(
        """
<div class="title-bar">
  <div class="title-row">
    <div class="title-badge">LEGAL RAG</div>
    <h1>Юридический ассистент</h1>
  </div>
  <p class="title-sub">Ответы только по базе Adilet, с цитатами и источниками.</p>
</div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
<div class="title-bar">
  <div class="title-row">
    <div class="title-badge">LEGAL RAG</div>
    <h1>Помощник по законам Республики Казахстан</h1>
  </div>
  <p class="title-sub">Строгие ответы только по базе Adilet, с дословными цитатами и источниками.</p>
</div>
        """,
        unsafe_allow_html=True,
    )
st.markdown(
    f'<div class="disclaimer-pill">{DISCLAIMERS[st.session_state.lang]}</div>',
    unsafe_allow_html=True,
)

# Соответствие закону РК об ИИ (прозрачность)
st.caption(config.AI_LAW_COMPLIANCE_NOTE)
st.markdown('<div class="chat-shell">', unsafe_allow_html=True)
for message in messages:
    avatar = "👤" if message["role"] == "user" else "⚖️"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
st.markdown("</div>", unsafe_allow_html=True)

# Ввод вопроса
prompt = st.chat_input(PLACEHOLDERS[st.session_state.lang])
if prompt:
    messages.append({"role": "user", "content": prompt})
    current_chat["updated_at"] = _now_iso()
    if current_chat["title"] == "Новый чат":
        current_chat["title"] = _title_from_prompt(prompt)
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.spinner(
        "Ищу в текстах законов..."
        if st.session_state.lang == "ru"
        else "Заң мәтінінде іздеймін..."
    ):
        try:
            result = invoke_qa(prompt)
            response = result["result"]
            sources = result["source_documents"]
            response = validate_answer(prompt, response, sources)
            if response == "Информация не найдена в доступных текстах законов.":
                sources = []
        except Exception as e:
            response = f"Ошибка при обработке вопроса: {str(e)}"
            sources = []

    with st.chat_message("assistant", avatar="⚖️"):
        st.markdown(response)
        if sources:
            st.markdown(
                f'<div class="sources-title">{SOURCES_LABEL[st.session_state.lang]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="sources-footer">', unsafe_allow_html=True)
            for i, doc in enumerate(sources, 1):
                src = doc.metadata.get("source", "неизвестно")
                filename = src.split("/")[-1] if "/" in src else src
                code_ru = doc.metadata.get("code_ru", "")
                art = doc.metadata.get("article_number", "")
                preview = doc.page_content[:280].replace("\n", " ").strip()
                title_bits = []
                if code_ru:
                    title_bits.append(f"**{code_ru}**")
                if art:
                    title_bits.append(f"ст.{art}")
                title_text = " • ".join(title_bits)
                st.markdown(
                    f'<div class="source-item">'
                    f'<div class="source-meta">🔗 {i}. <strong>{filename}</strong>'
                    + (f" — {title_text}" if title_text else "")
                    + "</div>"
                    + f'<div class="source-quote">{preview}...</div>'
                    + "</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)
    if sources:
        sources_text = "\n".join(
            [
                f"{j + 1}. {doc.metadata.get('source', '')} — {doc.metadata.get('code_ru', '')} ст.{doc.metadata.get('article_number', '')} — {doc.page_content[:200].replace(chr(10), ' ')}..."
                for j, doc in enumerate(sources)
            ]
        )
        full_text = f"{response}\n\nИсточники:\n{sources_text}"
        st.download_button(
            label=DOWNLOAD_LABEL[st.session_state.lang],
            data=full_text,
            file_name="ответ_по_законам.txt",
            mime="text/plain",
        )
    messages.append({"role": "assistant", "content": response})
    current_chat["updated_at"] = _now_iso()
    _save_chats(st.session_state.chat_store)

if len(messages) > 2 and st.button(
    CLEAR_CHAT[st.session_state.lang] + " (тек осы чат)", key="clear_chat"
):
    current_chat["messages"] = []
    current_chat["updated_at"] = _now_iso()
    _save_chats(st.session_state.chat_store)
    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
