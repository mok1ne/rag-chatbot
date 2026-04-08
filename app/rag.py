"""
RAG Pipeline — ядро проекта.

Архитектура (то, что стоит объяснить на собеседовании):
  1. Query Rewriting (Haiku 4.5)  — переформулирует вопрос для лучшего поиска
  2. Semantic Search (Pinecone)   — косинусное сходство по эмбеддингам
  3. Answer Generation (Sonnet 4.5) — генерирует ответ только по найденному контексту

Почему две модели?
  - Haiku 4.5: $1/$5 за млн токенов, в 4-5х быстрее Sonnet
  - Sonnet 4.5: $3/$15 за млн токенов, лучшее качество для финального ответа
  - Экономия ~60% на промежуточных шагах при сохранении качества
"""

import io
from typing import Any

import anthropic
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings

from app.config import settings

# Используем OpenAI эмбеддинги — стандарт для Pinecone
# (Anthropic пока не предоставляет свои эмбеддинг-модели через API)
import os
import tempfile
import hashlib


class RAGPipeline:
    def __init__(self):
        # Клиент Anthropic (прямой API, не Bedrock)
        self.client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

        # Pinecone — новый SDK (pinecone>=3.0)
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self._ensure_index()
        self.index = self.pc.Index(settings.PINECONE_INDEX_NAME)

        # Эмбеддинги
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # Сплиттер для разбивки документов
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " "],
        )

    def _ensure_index(self):
        """Создаёт Pinecone индекс, если его нет."""
        existing = [i.name for i in self.pc.list_indexes()]
        if settings.PINECONE_INDEX_NAME not in existing:
            self.pc.create_index(
                name=settings.PINECONE_INDEX_NAME,
                dimension=1536,  # размерность text-embedding-3-small
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=settings.PINECONE_CLOUD,
                    region=settings.PINECONE_REGION,
                ),
            )

    # ──────────────────────────────────────────
    # ШАГ 1: Query Rewriting через Haiku 4.5
    # ──────────────────────────────────────────
    def _rewrite_query(self, question: str) -> str:
        """
        Переформулирует вопрос пользователя для лучшего семантического поиска.
        Используем дешёвый Haiku 4.5 — это промежуточный шаг, не требует Sonnet.

        Пример: "что там с рисками?" → "основные риски в игровой индустрии по данным отчёта"
        """
        response = self.client.messages.create(
            model=settings.HAIKU_MODEL,
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": f"""Перефразируй вопрос пользователя так, чтобы он лучше 
подходил для семантического поиска по документам. 
Верни ТОЛЬКО переформулированный вопрос, без пояснений.

Вопрос: {question}""",
                }
            ],
        )
        return response.content[0].text.strip()

    # ──────────────────────────────────────────
    # ШАГ 2: Semantic Search в Pinecone
    # ──────────────────────────────────────────
    def _retrieve(self, query: str, top_k: int) -> list[dict]:
        """
        Конвертирует запрос в эмбеддинг и ищет ближайшие чанки в Pinecone.
        Возвращает список {'text': ..., 'source': ..., 'score': ...}
        """
        query_embedding = self.embeddings.embed_query(query)
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
        )
        chunks = []
        for match in results.matches:
            chunks.append(
                {
                    "text": match.metadata.get("text", ""),
                    "source": match.metadata.get("source", "unknown"),
                    "score": round(match.score, 3),
                }
            )
        return chunks

    # ──────────────────────────────────────────
    # ШАГ 3: Answer Generation через Sonnet 4.5
    # ──────────────────────────────────────────
    def _generate_answer(self, question: str, chunks: list[dict]) -> str:
        """
        Генерирует ответ на основе найденных чанков.
        Sonnet 4.5 — лучшее качество, строго только по контексту (нет галлюцинаций).
        """
        context = "\n\n".join(
            f"[Источник: {c['source']} | релевантность: {c['score']}]\n{c['text']}"
            for c in chunks
        )

        # Промпт с XML-тегами — рекомендованный Anthropic формат для Claude 4.x
        prompt = f"""Ты — помощник для анализа документов. Отвечай СТРОГО на основе 
предоставленного контекста. Если ответа в контексте нет — честно скажи об этом.

<context>
{context}
</context>

<question>
{question}
</question>

Дай подробный ответ на вопрос, опираясь на контекст выше. 
Укажи, из каких источников взята информация."""

        response = self.client.messages.create(
            model=settings.SONNET_MODEL,
            max_tokens=settings.MAX_TOKENS,
            temperature=settings.TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    # ──────────────────────────────────────────
    # Публичный метод: полный RAG-цикл
    # ──────────────────────────────────────────
    def query(self, question: str, top_k: int = None) -> dict[str, Any]:
        top_k = top_k or settings.TOP_K

        # 1. Rewrite (Haiku 4.5 — дёшево и быстро)
        rewritten = self._rewrite_query(question)

        # 2. Retrieve (Pinecone semantic search)
        chunks = self._retrieve(rewritten, top_k)
        if not chunks:
            return {
                "answer": "Документы не найдены. Загрузите PDF через /ingest.",
                "sources": [],
                "rewritten_query": rewritten,
            }

        # 3. Generate (Sonnet 4.5 — качественный ответ)
        answer = self._generate_answer(question, chunks)

        # Уникальные источники
        sources = list(dict.fromkeys(c["source"] for c in chunks))

        return {
            "answer": answer,
            "sources": sources,
            "rewritten_query": rewritten,
        }

    # ──────────────────────────────────────────
    # Инgest: загрузка и индексация PDF
    # ──────────────────────────────────────────
    def ingest_pdf(self, content: bytes, filename: str) -> int:
        """
        Разбивает PDF на чанки, создаёт эмбеддинги и загружает в Pinecone.
        Возвращает количество загруженных чанков.
        """
        # Сохраняем во временный файл (PyPDFLoader требует путь)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # Загружаем и разбиваем
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        chunks = self.splitter.split_documents(pages)

        # Создаём эмбеддинги и загружаем в Pinecone батчами
        batch_size = 100
        vectors = []
        for i, chunk in enumerate(chunks):
            embedding = self.embeddings.embed_query(chunk.page_content)
            # Стабильный ID через хеш содержимого — защита от дублей
            chunk_id = hashlib.md5(
                f"{filename}_{i}_{chunk.page_content[:50]}".encode()
            ).hexdigest()
            vectors.append(
                {
                    "id": chunk_id,
                    "values": embedding,
                    "metadata": {
                        "text": chunk.page_content,
                        "source": filename,
                        "page": chunk.metadata.get("page", 0),
                    },
                }
            )
            # Загружаем батчами
            if len(vectors) >= batch_size:
                self.index.upsert(vectors=vectors)
                vectors = []

        if vectors:
            self.index.upsert(vectors=vectors)

        os.unlink(tmp_path)
        return len(chunks)

    def get_index_stats(self) -> dict:
        stats = self.index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "index_name": settings.PINECONE_INDEX_NAME,
            "dimension": stats.dimension,
        }
