FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin tesseract-ocr-ori \
    tesseract-ocr-ben tesseract-ocr-tel tesseract-ocr-tam tesseract-ocr-spa \
    libglib2.0-0 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')" || exit 1
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
