#!/bin/bash
set -e

# Optional: if HUGGINGFACE_HUB_TOKEN provided, configure
if [ -n "${HUGGINGFACE_HUB_TOKEN}" ]; then
  mkdir -p /root/.huggingface
  echo -n "${HUGGINGFACE_HUB_TOKEN}" > /root/.huggingface/token
  export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN}"
fi


# create data folders
mkdir -p /data/input /data/output /data/voices

# --- Автоматическая загрузка модели и копирование yaml-конфига ---

# Полная вложенность для yaml: Misha24-10/F5-TTS_RUSSIAN/F5TTS_v1_Base_v2.yaml
MODEL_REPO="Misha24-10/F5-TTS_RUSSIAN"
MODEL_CACHE="/root/.cache/huggingface/hub"
CONFIG_TARGET_DIR="/usr/local/lib/python3.11/site-packages/f5_tts/configs/Misha24-10/F5-TTS_RUSSIAN"
CONFIG_TARGET_FILE="${CONFIG_TARGET_DIR}/F5TTS_v1_Base_v2.yaml"
LOCAL_YAML="/app/f5tts_config/F5TTS_v1_Base_v2.yaml"

# Скачиваем модель, если не скачана
python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${MODEL_REPO}', cache_dir='${MODEL_CACHE}')"

# Копируем yaml-конфиг, если его нет
if [ ! -f "${CONFIG_TARGET_FILE}" ]; then
  mkdir -p "${CONFIG_TARGET_DIR}"
  cp "$LOCAL_YAML" "$CONFIG_TARGET_FILE"
  echo "Copied model config: $LOCAL_YAML -> $CONFIG_TARGET_FILE"
fi

# run the API
exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-4123} --workers 1
