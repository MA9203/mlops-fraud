FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for xgboost
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src src
COPY data data

CMD ["python", "src/train.py"]
