FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/api/ src/api/
# Artifacts will be mounted or copied in CI/CD
COPY models/ models/
COPY data/processed/ data/processed/

ENV PYTHONPATH=/app

# Security: Create a non-root user
RUN adduser --disabled-password --gecos '' appuser && chown -R appuser:appuser /app
USER appuser

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

