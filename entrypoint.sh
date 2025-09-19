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

# run the API
exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-4123} --workers 1
