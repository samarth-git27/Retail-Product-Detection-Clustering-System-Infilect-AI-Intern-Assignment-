# Dockerfile for running the app (CPU)
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps needed for Pillow / Ultralityics
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libglib2.0-0 libsm6 libxrender1 libxext6 wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip --no-cache-dir install --upgrade pip
# Install CPU pytorch wheel first (recommended)
RUN pip --no-cache-dir install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip --no-cache-dir install -r /app/requirements.txt

# Copy app code
COPY . /app

EXPOSE 5000
CMD ["python", "app.py"]
