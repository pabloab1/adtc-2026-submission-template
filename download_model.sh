#!/usr/bin/env bash
# Downloads the GGUF model weights into model/.
# Rules this must satisfy (per ADTC 2026 submission template):
#   - idempotent: safe to re-run without re-downloading
#   - no credentials required: source must be publicly accessible
#   - downloaded path must exactly match metadata.json's _runtime.model_path

set -euo pipefail

MODEL_DIR="model"
MODEL_FILE="qwen2.5-1.5b-instruct-q4_k_m.gguf"
MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"
MODEL_URL="https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
EMBED_FILE="all-minilm-l6-v2-q8_0.gguf"
EMBED_PATH="${MODEL_DIR}/${EMBED_FILE}"
EMBED_URL="https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/all-MiniLM-L6-v2-Q8_0.gguf"

mkdir -p "${MODEL_DIR}"

# Download Chat Model
if [ -f "${MODEL_PATH}" ]; then
  echo "Chat model already present at ${MODEL_PATH} — skipping."
else
  echo "Downloading ${MODEL_FILE} ..."
  curl -L --fail --retry 5 -C - -o "${MODEL_PATH}.part" "${MODEL_URL}"
  mv "${MODEL_PATH}.part" "${MODEL_PATH}"
fi

# Download Embedding Model
if [ -f "${EMBED_PATH}" ]; then
  echo "Embedding model already present at ${EMBED_PATH} — skipping."
else
  echo "Downloading ${EMBED_FILE} ..."
  curl -L --fail --retry 5 -C - -o "${EMBED_PATH}.part" "${EMBED_URL}"
  mv "${EMBED_PATH}.part" "${EMBED_PATH}"
fi

echo "All models downloaded to ${MODEL_DIR}/"
