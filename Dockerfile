FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY tests/ tests/
COPY models/ models/

# Default: PyTorch checkpoint. Override with -e MODEL_PATH=/app/models/spleen_unet.onnx
ENV MODEL_PATH=/app/models/best_model_multicenter.pth
EXPOSE 8000

RUN useradd -m app && chown -R app /app
USER app

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
