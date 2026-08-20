FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

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

RUN groupadd --gid 10001 studyloop \
    && useradd --uid 10001 --gid studyloop --no-create-home \
        --home-dir /app --shell /usr/sbin/nologin studyloop

COPY --chown=studyloop:studyloop . .

# Chroma and local checkpoint/SQLite fallbacks must remain writable when the
# image is run without PostgreSQL as well as when Compose mounts a fresh volume.
RUN mkdir -p /app/chroma_db /app/.checkpoints \
    && chown -R studyloop:studyloop /app

USER studyloop

EXPOSE 8001

HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health/live', timeout=2).read()" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1", "--no-access-log"]
