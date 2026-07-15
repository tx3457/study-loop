FROM python:3.11-slim

WORKDIR /app

# Chroma build support plus PDF/OCR runtime dependencies used by services/parser.py.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --retries 5 --timeout 120 -r requirements.txt

COPY . .

# ChromaDB 持久化目录（由 volume 挂载）
RUN mkdir -p /app/chroma_db

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
