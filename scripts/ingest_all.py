"""
Скрипт пакетной загрузки всех PDF из папки data/ в Pinecone.
Запуск: python scripts/ingest_all.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.rag import RAGPipeline
from app.config import settings


def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]

    if not pdf_files:
        print("❌ PDF-файлы не найдены в папке data/")
        return

    print(f"🔍 Найдено {len(pdf_files)} PDF-файлов")
    print(f"📦 Индекс: {settings.PINECONE_INDEX_NAME}")
    print()

    pipeline = RAGPipeline()
    total_chunks = 0

    for filename in pdf_files:
        path = os.path.join(data_dir, filename)
        print(f"⏳ Обрабатываю: {filename} ...", end=" ", flush=True)
        start = time.time()

        with open(path, "rb") as f:
            content = f.read()

        count = pipeline.ingest_pdf(content, filename=filename)
        elapsed = round(time.time() - start, 1)
        total_chunks += count
        print(f"✅ {count} чанков ({elapsed}с)")

    stats = pipeline.get_index_stats()
    print()
    print(f"✅ Готово! Загружено {total_chunks} чанков")
    print(f"📊 Всего в индексе: {stats['total_vectors']} векторов")


if __name__ == "__main__":
    main()
