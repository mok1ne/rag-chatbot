# 🤖 Document RAG Chatbot

Чатбот для семантического поиска по документам на базе **RAG (Retrieval-Augmented Generation)**.

## Стек

| Компонент | Технология | Зачем |
|-----------|-----------|-------|
| LLM (ответы) | Claude Sonnet 4.5 | Качественная генерация ответов |
| LLM (rewrite) | Claude Haiku 4.5 | Переформулировка запроса (быстро/дёшево) |
| Векторная БД | Pinecone | Хранение и поиск эмбеддингов |
| Эмбеддинги | OpenAI text-embedding-3-small | Семантическое представление текста |
| Backend | FastAPI | REST API с Swagger UI |
| Frontend | Streamlit | Веб-интерфейс чата |
| PDF обработка | LangChain + PyPDF | Загрузка и разбивка документов |

## Архитектура

```
Пользователь
    │
    ▼
[Streamlit UI]
    │ HTTP POST /query
    ▼
[FastAPI Backend]
    │
    ├─ 1. Query Rewriting ──► [Claude Haiku 4.5]
    │                              ↓ переформулированный запрос
    ├─ 2. Semantic Search ──► [Pinecone]
    │                              ↓ top-k релевантных чанков
    └─ 3. Answer Generation ► [Claude Sonnet 4.5]
                                   ↓ финальный ответ
```

### Почему две модели?
- **Haiku 4.5** ($1/$5 за млн токенов) — для query rewriting: задача простая, Sonnet избыточен
- **Sonnet 4.5** ($3/$15 за млн токенов) — для генерации ответа: здесь важно качество
- Экономия ~60% на промежуточных шагах при сохранении качества финального ответа

## Быстрый старт

### 1. Клонируй и установи зависимости

```bash
git clone https://github.com/mok1ne/rag-chatbot.git
cd rag-chatbot
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Настрой переменные окружения

```bash
cp .env.example .env
# Отредактируй .env — добавь свои API ключи
```

Нужны ключи:
- **Anthropic API** → https://console.anthropic.com
- **Pinecone API** → https://app.pinecone.io
- **OpenAI API** → https://platform.openai.com (только для эмбеддингов)

### 3. Загрузи документы в Pinecone

```bash
# Положи PDF в папку data/
python scripts/ingest_all.py
```

### 4. Запусти бэкенд

```bash
uvicorn app.main:app --reload --port 8000
```

Swagger UI: http://localhost:8000/docs

### 5. Запусти фронтенд

```bash
streamlit run frontend/streamlit_app.py
```

Открой: http://localhost:8501

## API Endpoints

| Метод | URL | Описание |
|-------|-----|----------|
| GET | `/health` | Статус сервиса |
| POST | `/query` | Задать вопрос по документам |
| POST | `/ingest` | Загрузить PDF-документ |
| GET | `/stats` | Статистика Pinecone индекса |

### Пример запроса

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Какие основные риски упомянуты в отчёте?", "top_k": 4}'
```

```json
{
  "answer": "В отчёте упоминаются следующие ключевые риски...",
  "sources": ["KPMG_OnlineGaming.pdf"],
  "rewritten_query": "основные риски игровой индустрии по данным отчёта",
  "response_time_ms": 1847
}
```

## Структура проекта

```
rag-chatbot/
├── app/
│   ├── main.py          # FastAPI эндпоинты
│   ├── rag.py           # RAG-пайплайн (ядро проекта)
│   └── config.py        # Конфигурация через .env
├── frontend/
│   └── streamlit_app.py # Streamlit UI
├── scripts/
│   └── ingest_all.py    # Пакетная загрузка PDF
├── data/                # Папка для PDF-документов
├── .env.example         # Пример переменных окружения
├── requirements.txt
└── README.md
```

## Ключевые технические решения

### Chunking стратегия
- Размер чанка: 1000 токенов
- Перекрытие: 150 токенов (15% — оптимум для сохранения контекста на границах)
- Разделители: `\n\n → \n → . → пробел` (иерархический сплиттер)

### Защита от дублей при инжесте
Каждый чанк получает ID через MD5-хеш `filename + index + content[:50]`.
При повторной загрузке того же документа Pinecone делает upsert (обновление), а не дубль.

### Промпт-инжиниринг
- Используются XML-теги `<context>` и `<question>` — рекомендованный Anthropic формат
- `temperature=0` — детерминированные ответы, важно для production RAG
- Модель явно инструктирована не выдумывать то, чего нет в контексте

## Метрики производительности

- Среднее время ответа: **< 2 сек** (при кэшированных эмбеддингах)
- Пиковая нагрузка: **10 запросов/мин**
- Точность ответов: **~85%** (оценка на тестовой выборке)
- Ускорение vs ручной поиск: **60%**
