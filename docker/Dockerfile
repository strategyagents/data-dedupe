FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip httpx qdrant-client flask

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
