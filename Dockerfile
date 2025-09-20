# Dockerfile — build lightweight CPU image for F5-TTS + small API wrapper
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    ffmpeg \
    libsndfile1 \
    curl \
    wget \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy files
COPY requirements.txt /app/requirements.txt
COPY app.py /app/app.py
COPY entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

# Python deps
RUN pip install --no-cache-dir -r /app/requirements.txt

# Ensure huggingface cache dir exists (models will be downloaded here)
VOLUME ["/root/.cache/huggingface", "/data"]

EXPOSE 4123

CMD ["/app/entrypoint.sh"]
