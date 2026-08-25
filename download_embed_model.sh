#!/bin/bash
# Download the GGUF embedding model used by the retrieval engine.
# all-MiniLM-L6-v2 Q8_0 (~25 MB), quantized for llama.cpp embedding mode.
# Usage: ./download_embed_model.sh
set -e
cd "$(dirname "$0")"
mkdir -p model
OUT="model/all-minilm-l6-v2-q8_0.gguf"
URL="https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/all-MiniLM-L6-v2-Q8_0.gguf"
if [ -f "$OUT" ]; then
  echo "Embedding model already present: $OUT ($(du -h "$OUT" | cut -f1))"
else
  echo "Downloading all-MiniLM-L6-v2 Q8_0 GGUF (~25 MB)..."
  curl -L --retry 3 -o "$OUT" "$URL"
  echo "Saved: $OUT ($(du -h "$OUT" | cut -f1))"
fi
