FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
COPY src/model/requirements.txt src/model/requirements.txt
COPY src/monitoring/requirements.txt src/monitoring/requirements.txt

# Install all dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r src/model/requirements.txt
RUN pip install --no-cache-dir -r src/monitoring/requirements.txt

COPY . .

# 👉 Ajout essentiel
ENV PYTHONPATH="/app"

# Command to run monitoring
CMD ["python", "src/monitoring/scheduled_monitoring.py"]