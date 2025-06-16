FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV TORCH_HOME=/app/cache/torch
ENV YOLO_CONFIG_DIR=/app/cache/yolo

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "model_comparison.py", "--input", "/app/input", "--output", "/app/output"]