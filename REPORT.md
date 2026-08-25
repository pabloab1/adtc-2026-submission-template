# ADTC 2026 — Technical Report

## 1. Problem

Smallholder farmers and agricultural extension officers in Nigeria often lack
reliable, on-demand access to accurate crop and pest advice — especially
where mobile data is expensive or connectivity is unreliable. Existing
advisory channels (NAERLS's National Farmers Helpline, extension officer
visits) are valuable but limited by staffing and reach.

This project is an offline agricultural advisory chatbot: a farmer asks a
question in English, and the system retrieves the relevant fact
from a corpus of real agricultural extension material (IITA agronomy guides,
NAERLS advisory content) and composes a spoken-language answer — entirely
on-device, no cloud calls, no data cost to the farmer.

Target user: a smallholder farmer or extension officer in Nigeria with a
budget laptop and no reliable internet access.

## 2. Design Decisions

**Retrieval:** TF-IDF over character n-grams (scikit-learn), not a
downloaded embedding model — this keeps the retrieval half fully offline
and dependency-light. It's a deliberate simplicity/quality trade-off:
TF-IDF matches on shared substrings, not meaning, so it works best when the
farmer's phrasing overlaps with the stored question's vocabulary. [FILL IN:
note here once you've tested with paraphrased/messy real queries whether
this was good enough, or whether you upgraded to a real sentence-embedding
model — and why.]

**Generation:** Qwen2.5-1.5B-Instruct, GGUF Q4_K_M quantization, served via
llama.cpp (through the llama-cpp-python binding). Chosen for:
- small footprint (~1 GB weights) comfortably inside the 8 GB RAM budget
  alongside the OS and retrieval process
- decent instruction-following at this size for a RAG-style "answer using
  only this context" prompt

**Alternatives considered:** [FILL IN — e.g. did you try a smaller 0.5B
variant for more RAM headroom, or a larger 3B for better English answer
quality? What did you observe on throughput and RAM?]

**Language scope — English only (decision made, not open):** This
submission ships English-only. We tested Qwen2.5-1.5B on Yoruba directly:
it *understands* Yoruba input well (a Yoruba cassava-harvest query retrieved
its correct pair at full confidence), but it *cannot compose* usable Yoruba
output — generated answers were grammatically broken and dropped the key
facts from the grounding text. Its pretraining is English/Chinese-dominant
with limited African-language coverage, so a 1.5B model at Q4_K_M is not a
trustworthy Yoruba writer. Rather than ship output we can't stand behind, we
set `language_scope` to `["en"]` and disabled Yoruba in
`config/languages.json`. The Yoruba question translations (and space for
drafted answers) remain in `data/qa_dataset.json`; re-enabling the language
later is a one-line config change once a complete, native-verified Yoruba
answer set exists. Consequently `african_alpha_claim` is `false` — we are not
claiming the African Use Case Bonus, since we would rather claim it on
delivered multilingual output than on input handling alone.

## 3. Constraints

- **Hardware:** ADTC Standard Laptop (i5/Ryzen 5, 8 GB RAM, integrated
  graphics only, no discrete GPU) — rules out any GPU-accelerated inference
  path.
- **Connectivity:** must run 100% offline at inference time; the only
  network dependency is the one-time model download before evaluation
  begins.
- **Data:** the knowledge base is a **100-pair hand-curated English Q&A set**
  drawn from real agricultural sources: the original 20 pairs (cassava, maize,
  extension) from IITA/NAERLS guides, plus 80 new pairs covering yam, rice,
  sorghum, and millet. The 80 are sourced from Wikipedia crop-production
  articles (Yam production in Nigeria, Rice, Sorghum, Millet) and two
  peer-reviewed field studies — *Field Crops Research* (Ibrahim et al., 2015)
  on millet fertilizer micro-dosing in the Sahel, and *PLOS One* (Mohamed
  et al., 2025) on millet under climate change. Each pair is a hand-written
  question paired with a source-grounded answer and a `source` field, rather
  than auto-extracted PDF chunks, so no OCR/extraction-cleaning step is needed;
  every entry was authored directly from the cited material. Validation: the
  self-retrieval test (`test_retrieval.py`) passes **100/100** — every stored
  question retrieves its own pair as the top match. Scaling further with
  full-text PDF guides as an extraction-based RAG corpus remains future work.
- **Language:** English-only for this submission (see §2). Yoruba question
  translations are retained in the dataset for a future release but are not
  indexed or served; they were machine-drafted and would need native-speaker
  verification before being trusted as ground truth.

## 4. Benchmarks

Measured with the official `adtc-profiler` (participant mode,
`--skip-accuracy`) on the development laptop. **Read the throughput number
in context:** this machine is an Intel Core i5-7Y54 @ 1.2 GHz — a fanless,
dual-core ultra-low-voltage chip that is substantially slower than the ADTC
"Standard Laptop" reference profile (i5/Ryzen 5). Treat generation
throughput below as a pessimistic floor; the audit hardware will be faster.
The figures that matter most for the 8 GB budget — peak RAM and thermal
behavior — are both comfortable.

Reproduce with:

```
bash download_model.sh
adtc-profiler run --submission . --mode participant --output submission.json --skip-accuracy
cat submission.json
```

Observed (`submission.json`, schema 1.1.0):

| Metric | Value | Notes |
| --- | --- | --- |
| Peak RSS | **1,699.8 MB (~1.7 GB)** | ≈21% of the 8 GB budget — large headroom, no OOM risk |
| Steady-state RSS | 1,604.4 MB (~1.6 GB) | model weights + llama.cpp runtime resident |
| Generation throughput | **1.88 tok/s** | on this fanless i5-7Y54 @ 1.2 GHz; standalone `llama-bench` (no profiler samplers) measured 2.31 tok/s |
| First-token latency | 62.4 s | profiler uses a 512-token synthetic prompt; real chatbot prompts are ~45 tokens, so actual time-to-first-token is far lower |
| Thermal throttling | **None** (`throttled: false`) | no thermal penalty; core-temp sensor not readable on Windows (reported `null`) |
| Params check | **Pass** (`params_match: true`) | GGUF header reports 1,543,714,304 params (1.54B), matching the claimed "1.5B" |
| Environment | i5-7Y54, GPU `none`, 7.9 GB RAM | CPU-only inference on a genuine 8 GB-class laptop |

No OOM or crash occurred during profiling. Peak memory of ~1.7 GB on a 7.9 GB
machine leaves roughly 6 GB free, so the OS, the TF-IDF retrieval process, and
the chat UI all fit alongside the model comfortably. The small gap between the
profiler's 1.88 tok/s and the standalone `llama-bench` 2.31 tok/s is expected:
the profiler runs memory and thermal samplers concurrently, which competes for
cycles on this 2-core CPU.

## 5. Demo

The chatbot answers English agriculture queries end-to-end via
`python chatbot.py` (full RAG) or `python chatbot.py --no-llm` (retrieval-only,
returns the stored sourced answer verbatim). Example, in RAG mode:

- **Q:** "How do I control Fall Armyworm in my maize farm?" → retrieved the
  IITA maize-production pair (confidence 0.89) and answered: *"Control Fall
  Armyworm by applying frequent insecticide applications, sometimes using
  multiple chemical types, since it's difficult to control once established."*
- **Q:** "When can I harvest cassava?" → retrieved the IITA cassava-agronomy
  pair (confidence 1.0) and answered from the 7-/18-month grounding fact.

(The official template does not require screenshots or a demo video, so none
are included here; add them if submitting through a channel that asks for
them.)
