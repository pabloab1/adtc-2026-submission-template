"""
Thin wrapper around llama-cpp-python so the rest of the app doesn't need to
know llama.cpp's API directly. This is the piece that makes the submission
an actual "on-device Language Model" rather than a lookup table — retrieval
finds the grounding facts, this generates the natural-language response.

Requirements this satisfies (ADTC 2026 rules):
  - runtime: llama.cpp only (via llama-cpp-python, the official Python
    binding for llama.cpp — it does not reimplement inference itself)
  - 100% offline at inference time: no network calls happen in this file
  - must run within the 8 GB RAM budget: n_ctx and n_threads below are
    conservative defaults for that budget; tune them after profiling on
    the actual reference laptop, not before

Install locally before testing (not pre-installed in this sandbox):
    pip install llama-cpp-python
"""

from pathlib import Path

DEFAULT_MODEL_PATH = Path(__file__).parent / "model" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"


class LocalLLM:
    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        n_ctx: int = 2048, # Increased to 2048 to prevent crashes with large context
        n_threads: int = 2, # Optimized for i5-7Y54 (2 cores, 4 threads) to avoid overhead
    ):
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python is not installed. Run: pip install llama-cpp-python"
            ) from e

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run `bash download_model.sh` first."
            )

        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )

    def generate(self, prompt: str, max_tokens: int = 300, temperature: float = 0.3) -> str:
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["</s>", "<|im_end|>"],
        )
        return output["choices"][0]["text"].strip()

    def chat(self, system_prompt: str, user_message: str, max_tokens: int = 300,
             messages: "list | None" = None, temperature: float = 0.3, stream: bool = False):
        """Uses llama.cpp's chat-completion helper, which applies the model's
        chat template automatically (Qwen2.5 uses ChatML).

        If `messages` is given, use it as the full conversation (system prompt
        + history + final user turn) — used by the conversational chatbot so
        follow-ups resolve against what was just discussed. "system_prompt" is
        ignored in that case (it's baked into messages[0])."""
        if messages is None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )
        if stream:
            return response
        return response["choices"][0]["message"]["content"].strip()
