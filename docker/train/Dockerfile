FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY dvc.yaml .
# In a real scenario, we might mount data or pull from remote storage
# For this reproducible image, we assume we might run 'dvc pull' or mount volumes

# Security: Create a non-root user
RUN adduser --disabled-password --gecos '' appuser && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["dvc", "repro"]

