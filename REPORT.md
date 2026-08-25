# ADTC 2026 — Technical Report: Farmmate-NG

## 1. Problem Statement

Smallholder farmers and agricultural extension officers in Nigeria often lack reliable, on-demand access to accurate crop and pest advice due to expensive mobile data and unreliable connectivity. **Farmmate-NG** is an offline agricultural advisory chatbot designed to bridge this gap. It provides sourced agronomic advice entirely on-device, ensuring zero data costs and high availability.

The target user is a smallholder farmer or extension officer using a budget laptop (e.g., Intel Core i5, 8 GB RAM) in an environment without internet access.

## 2. Technical Architecture

### 2.1 Retrieval System (RAG)
The project implements a **Retrieval-Augmented Generation (RAG)** architecture using a **GGUF-quantized embedding model** (`all-MiniLM-L6-v2-q8_0.gguf`) through the `llama.cpp` runtime. 
- **Knowledge Base:** Derived from a 134-chunk ingestion of real agricultural extension guides (IITA, NAERLS) in DOCX format.
- **Safety & Grounding:** The engine features a **crop-aware retrieval filter**. When a specific crop (e.g., Beans) is detected in the query or conversation history, the system restricts retrieval to that crop's specific sections and general farm tools, preventing cross-crop information leakage.
- **Scoring:** Uses honest L2-normalized cosine similarity. Queries falling below a 0.45 similarity threshold are flagged with a mandatory warning: `THE DATA BASE DID NOT SPECIFICALLY SAY THIS SO BE WARNED OF MISINFORMATION`.

### 2.2 Generation System
- **Model:** Qwen2.5-1.5B-Instruct (GGUF Q4_K_M quantization).
- **Runtime:** `llama-cpp-python` for CPU-only inference.
- **Memory Management:** The 1.5B model occupies ~1 GB of RAM, fitting comfortably within the 8 GB system limit while providing sufficient instruction-following for strict RAG grounding.

## 3. Implementation Details

- **Offline Execution:** 100% offline after the initial model download. No network calls are made during inference.
- **Cross-Platform:** Optimized for Ubuntu 22.04 LTS while maintaining full Windows compatibility.
- **Gibberish & Out-of-Domain Handling:** A custom `QueryProcessor` detects unreadable input and explicitly refuses queries about unsupported topics (e.g., livestock, wheat) to maintain safety.
- **History Management:** Maintains a sliding window of the last 2 turns to provide conversational continuity (e.g., "how to store it") without risking context drift or leakage.

## 4. Performance Benchmarks

Benchmarks were measured using the official `adtc-profiler` on a local Windows 10 development machine (Intel Family 6 Model 142, 8 GB RAM).

| Metric | Value | Notes |
| --- | --- | --- |
| **Peak RSS** | **1,699.82 MB** | ~24% of the 7 GB project RAM ceiling. |
| **Steady-state RSS** | 1,604.41 MB | Model weights + runtime overhead. |
| **Generation Speed** | **1.88 tokens/sec** | Measured on development hardware; expected higher on target i5-10th/12th Gen. |
| **First-Token Latency** | 62.4s (initial) | Cold start latency; subsequent turns are sub-second. |
| **Thermal Throttling** | None | No performance degradation detected during stress testing. |
| **Model Params** | 1.54B | Verified via GGUF header; matches the 1.5B claim. |

*Note: S_perf and S_eff scores are relative to the audit-wide maximum and will be determined during the official ADTC evaluation.*

## 5. Accuracy and Validation

The system was validated against a deterministic test suite covering:
1. **Crop Isolation:** Verified that Groundnut storage advice does not leak into Beans queries.
2. **Entity Coverage:** 100% retrieval success for all 10 supported crops and Farm Tools.
3. **Typo Tolerance:** Robust handling of minor misspellings and leet-speak (e.g., "b3ans", "m4ize").
4. **Safety Refusals:** Correct rejection of gibberish and unsupported agricultural topics.

## 6. Conclusion

Farmmate-NG delivers a robust, safe, and efficient offline advisory tool. By combining a real-world agricultural corpus with a crop-aware RAG engine and a lightweight LLM, it provides Nigerian farmers with trustworthy advice that fits within strict hardware and connectivity constraints.
