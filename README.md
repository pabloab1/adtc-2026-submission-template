# Farmmate-NG: Offline Agriculture Advisory Chatbot (ADTC 2026)

Farmmate-NG is an offline, on-device conversational RAG chatbot designed for the **Africa Deep Tech Challenge 2026 (Agriculture track)**. It provides sourced agronomic advice to Nigerian farmers entirely on-device, requiring no internet access or cloud services.

## Key Features
- **Crop-Aware RAG:** Prevents cross-crop information leakage (e.g., Groundnut advice in a Beans query) through explicit entity-filtered retrieval.
- **Offline LLM:** Uses Qwen2.5-1.5B-Instruct (GGUF Q4_K_M) served via `llama.cpp` for natural-language response generation.
- **Efficient Embedding:** Uses a GGUF-quantized `all-MiniLM-L6-v2` model through the same `llama.cpp` runtime for semantic search.
- **Safety First:** Mandatory misinformation warnings for low-confidence answers and strict refusal of gibberish or unsupported topics.
- **Hardware Optimized:** Fits within the **7 GB project RAM ceiling** and is optimized for CPU-only inference on budget i5/Ryzen 5 laptops.

## Project Structure
```text
metadata.json          — Submission metadata (team_id: farmmate-ng)
REPORT.md              — Technical report and performance benchmarks
download_model.sh      — Script to download chat and embedding models
dataset/               — Source agricultural guides (DOCX format)
index/                 — Pre-built vector index (chunks.json, chunk_vectors.npy)
chatbot.py             — Primary conversational engine
chunk_retriever.py     — Retrieval logic with crop filtering
rag_llm.py             — llama.cpp LLM wrapper
app.py                 — Streamlit web UI for local testing
test_grounding.py      — Regression suite for crop isolation and safety
```

## Setup Instructions

### Prerequisites
- Python 3.10+
- `pip install -r requirements.txt`

### 1. Download Models
Run the download script to fetch the required GGUF weights:
```bash
bash download_model.sh
```

### 2. Build the Index
Ingest the agricultural guides into the local vector store:
```bash
python3 chunk_retriever.py build
```

### 3. Run the Chatbot
Start the conversational CLI:
```bash
python3 chatbot.py
```

### 4. Run Local UI (Optional)
For a browser-based experience:
```bash
streamlit run app.py
```

## Submission Readiness
- **Metadata:** Fully updated with `team_id: farmmate-ng` and submitter details.
- **Grounding:** Verified by `test_grounding.py` to prevent crop leakage and handle gibberish.
- **Performance:** Profiled on target-class hardware; Peak RSS ~1.7 GB, comfortably under the 7 GB limit.
- **Structure:** Matches the official ADTC 2026 submission template.

## Technical Notes
- **llama.cpp Only:** Both embedding and generation run on `llama-cpp-python`. No Torch or Transformers dependencies at runtime.
- **Language:** English-only for this submission to ensure high-quality, verified output.
- **Entity Filtering:** Retrieval is restricted to the active crop in the conversation to ensure grounding accuracy.
