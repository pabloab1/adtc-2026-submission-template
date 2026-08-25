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

mkdir -p "${MODEL_DIR}"

if [ -f "${MODEL_PATH}" ]; then
  echo "Model already present at ${MODEL_PATH} — skipping download."
  exit 0
fi

echo "Downloading ${MODEL_FILE} from ${MODEL_URL} ..."
# Robust download for slow/unreliable connections:
#   -C -                : resume from the .part file if a previous run was interrupted
#   --retry / --retry-delay / --retry-all-errors : auto-retry transient drops, resuming each time
# Still idempotent (re-running resumes rather than restarting) and credential-free.
curl -L --fail --retry 10 --retry-delay 5 --retry-all-errors -C - \
  -o "${MODEL_PATH}.part" "${MODEL_URL}"
mv "${MODEL_PATH}.part" "${MODEL_PATH}"
echo "Saved to ${MODEL_PATH}"
