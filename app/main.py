"""
RAG Document Chatbot — FastAPI Backend
Стек: Claude Sonnet 4.5 (ответы) + Haiku 4.5 (rewrite запроса) + Pinecone + LangChain
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import time

from app.rag import RAGPipeline
from app.config import settings

app = FastAPI(
    title="Document RAG Chatbot",
    description="Семантический поиск по документам на базе Claude + Pinecone",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация RAG-пайплайна при старте
rag = RAGPipeline()


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 4


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    rewritten_query: str
    response_time_ms: int


@app.get("/health")
def health():
    return {"status": "ok", "model": settings.SONNET_MODEL}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Основной эндпоинт: принимает вопрос, возвращает ответ с источниками.
    Haiku 4.5 переформулирует запрос → Pinecone ищет чанки → Sonnet 4.5 отвечает.
    """
    start = time.time()
    try:
        result = rag.query(request.question, top_k=request.top_k)
        elapsed = int((time.time() - start) * 1000)
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            rewritten_query=result["rewritten_query"],
            response_time_ms=elapsed,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """
    Загрузка PDF-документа: разбивка на чанки → эмбеддинги → Pinecone.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Поддерживается только PDF")

    content = await file.read()
    try:
        count = rag.ingest_pdf(content, filename=file.filename)
        return {"message": f"Загружено {count} чанков из '{file.filename}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def stats():
    """Статистика индекса Pinecone."""
    return rag.get_index_stats()
