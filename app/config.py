"""
Конфигурация проекта — все настройки через .env
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Anthropic API
    ANTHROPIC_API_KEY: str

    # Модели — главная идея: Sonnet для ответов, Haiku для вспомогательных задач
    SONNET_MODEL: str = "claude-sonnet-4-5-20250929"   # генерация ответов
    HAIKU_MODEL: str = "claude-haiku-4-5-20251001"     # rewrite запроса (дешевле/быстрее)

    # Pinecone
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "rag-chatbot"
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"

    # RAG параметры
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    TOP_K: int = 4

    # LLM параметры
    MAX_TOKENS: int = 1500
    TEMPERATURE: float = 0.0   # 0 = детерминированные ответы, важно для RAG

    class Config:
        env_file = ".env"


settings = Settings()
