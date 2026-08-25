# Agriculture Advisory Chatbot — ADTC 2026

Offline, on-device conversational RAG chatbot for agricultural advice, built
for the Africa Deep Tech Challenge 2026 (Agriculture track). The knowledge
base is a set of crop `.docx` guides in `dataset/` — each guide is split
into its real sections (About, History, Planting, Harvest, Crop Rotation,
Pests, Pest Control, Machines and Tools, Care), embedded once into a local
vector store, and retrieved at chat time with cosine similarity. A local
llama.cpp LLM composes the spoken-language answer from the retrieved
sections in its own words — no cloud calls, no API keys.

## Structure (matches the official ADTC 2026 submission template)

```
metadata.json          — required submission metadata (team, model, test prompts)
download_model.sh      — downloads the GGUF model weights (run this first)
REPORT.md              — technical writeup (has [FILL IN] placeholders — complete before submitting)
model/                 — downloaded weights land here (gitignored)
.gitignore             — excludes *.gguf per submission rules

dataset/Rice.docx      — a crop guide; Heading 1 = crop name, Heading 2 = sections
index/                 — built vector store (chunk_vectors.npy, chunks.json, semantic_cache.json)
download_embed_model.sh — downloads the GGUF embedding model (all-MiniLM-L6-v2 Q8_0, ~25 MB)
config/languages.json  — which languages are active
data/qa_dataset.json   — legacy Q&A corpus, kept for the fallback TF-IDF path
chunk_retriever.py     — ingests dataset/*.docx, embeds sections into a local vector store,
                       and retrieves the most relevant sections per question
semantic_cache.py      — meaning-based answer cache (rephrased repeats are instant)
rag_llm.py             — llama.cpp wrapper (the actual on-device LLM)
chatbot.py             — primary engine: cache -> chunk retrieval -> LLM; run for a CLI chat loop
app.py                 — local Streamlit web UI for browser testing (not part of ADTC submission)
test_convo_pipeline.py — smoke test for the conversational engine (no model needed)
test_retrieval.py      — legacy fallback test: every stored question retrieves its own pair
retrieval.py / embedder.py — legacy TF-IDF fallback path
```

## First-time setup

The `.venv` is used instead of a global install. `requirements-dev.txt` is
now llama.cpp-only (torch/sentence-transformers removed — the build also
uses llama-cpp embeddings).

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash download_model.sh          # pulls the chat GGUF model into model/
bash download_embed_model.sh    # pulls the MiniLM Q8_0 GGUF into model/
python chunk_retriever.py build # builds the vector store from dataset/*.docx (one-time)
python test_convo_pipeline.py   # sanity-check the conversational engine (no model needed)
python chatbot.py               # full conversational RAG loop (needs the model downloaded)
python chatbot.py --no-llm      # retrieval-only mode, returns section text verbatim
```

## Local web UI (browser testing only)

Not required for ADTC submission. Install Streamlit once, then:

```bash
pip install streamlit
streamlit run app.py
```

Opens at http://localhost:8501 — a real conversation view with answer origin
(retrieved / LLM-composed / semantic cache), source section, and confidence.

## Before you submit

1. Fill in every `REPLACE_WITH_...` placeholder in `metadata.json` — the
   checklist in the official template requires **no placeholder values
   remain**.
2. Fill in every `[FILL IN]` section in `REPORT.md` with what you actually
   observed — don't leave them as-is.
3. Get the Yoruba entries in `data/qa_dataset.json` reviewed by a native
   speaker; they were machine-drafted.
4. Run the ADTC profiler locally and confirm you're within the 8 GB RAM
   budget before relying on any numbers:
   ```bash
   pip install "git+https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler.git"
   adtc-profiler run --submission . --mode participant --output submission.json --skip-accuracy
   ```
5. Fork the official template (https://github.com/Africa-Deep-Tech-Foundation/adtc-2026-submission-template)
   and move these files into that repo's structure if you haven't already —
   the evaluator expects that exact layout.

## Adding or swapping languages

1. Add the language code + display name to `config/languages.json`'s
   `active_languages`.
2. Add the code's display name to `LANGUAGE_NAMES` in `chatbot.py` (used
   only for the LLM's system prompt wording).
3. Update `language_scope` in `metadata.json` to match what you're actually
   submitting.

## Scaling up the knowledge base (adding crops)

Drop another guide into `dataset/` — e.g. `Maize.docx` — with the same
structure (Heading 1 = crop name, Heading 2 = sections). Then re-run:

```bash
python chunk_retriever.py build   # rebuilds the vector store automatically
python test_convo_pipeline.py
```

Nothing else changes — retrieval always runs against the live docx sections,
so the chatbot answers about maize immediately after ingestion.

## Architecture notes (runtime and rules)

- **llama.cpp only, end to end**: both retrieval and generation run through
  the same llama-cpp-python runtime. The embedding model is the GGUF
  all-MiniLM-L6-v2 Q8_0 (~25 MB) loaded with `embedding=True` in the
  `Llama` constructor — no torch, no sentence-transformers at any phase
  (build or runtime). Generation runs on llama.cpp with the Qwen2.5 GGUF
  weights in `model/`. No API keys, no cloud calls — compliant with ADTC
  2026 rules.
- Download the embedding model with `bash download_embed_model.sh` (or
  `download_embed_model.bat`).
- **8 GB budget-friendly**: ~90 MB embedder resident, 500-character chunk
  truncation, 220-token reply cap, 4 history turns kept for the 1.5B model.
- **Semantic cache**: a rephrased repeat question (embedding similarity
  ≥ 0.92 to a past question) gets an instant, persisted answer with zero
  retrieval and zero LLM call — matters on the target hardware.
- **Conversation memory**: history of up to 4 turns lets follow-ups
  ("what about its pests?") resolve against the crop just discussed.
- **Honesty by design**: below a 0.35 similarity floor the question is
  treated as not covered by the guides — the chatbot says so instead of
  guessing.

The legacy TF-IDF path (`retrieval.py`/`embedder.py`, fallback-only) and its
corpus in `data/qa_dataset.json` are kept so the original self-retrieval
test still passes; do not grow it with generated Q&A pairs — new knowledge
goes into `dataset/<Crop>.docx`.
