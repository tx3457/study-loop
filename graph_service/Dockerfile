FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY graph_service/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY graph_service /app/graph_service
CMD ["uvicorn", "graph_service.main:app", "--host", "0.0.0.0", "--port", "8081", "--workers", "1"]
