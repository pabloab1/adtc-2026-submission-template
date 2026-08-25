"""
Local smoke-test + mini-benchmark for the ADTC agri-chatbot.
Kept OUTSIDE the repo so the submission stays clean.

Runs the two official test prompts (tp_001 English, tp_002 Yoruba) through
BOTH modes, captures raw output verbatim, and measures load time + tokens/sec.
Also probes raw Qwen Yoruba quality directly (bypassing the English fallback)
so we can judge whether claiming Yoruba output is honest.
"""

import sys
import time
from pathlib import Path

REPO = Path(r"C:\Users\mirac\OneDrive\Documents\agri-chatbot-adtc2026\agri-chatbot")
sys.path.insert(0, str(REPO))
# retrieval.py / rag_llm.py resolve data + model paths relative to their own file,
# so cwd doesn't matter, but chdir anyway to be safe.
import os
os.chdir(REPO)

from chatbot import AgriChatbot  # noqa: E402

EN_PROMPT = "How do I control Fall Armyworm in my maize farm?"
YO_PROMPT = "Nígbà wo ni mo lè kórè kasada?"  # tp_002: "When can I harvest cassava?"

SEP = "=" * 70


def show(result: dict):
    print(f"  language answered : {result['language']}")
    print(f"  llm_generated     : {result['llm_generated']}")
    print(f"  confidence        : {result['confidence']}")
    print(f"  low_confidence    : {result['low_confidence']}")
    print(f"  matched question  : {result.get('matched_question')}")
    print(f"  source            : {result.get('source')}")
    if result.get("grounding_answer"):
        print(f"  grounding (en)    : {result['grounding_answer']}")
    print(f"  ANSWER            : {result['answer']}")


print(SEP)
print("PART 1 — retrieval-only mode (--no-llm), no model needed")
print(SEP)
bot_ret = AgriChatbot(use_llm=False)
for label, q in [("EN tp_001", EN_PROMPT), ("YO tp_002", YO_PROMPT)]:
    print(f"\n[{label}] {q}")
    show(bot_ret.ask(q))

print("\n" + SEP)
print("PART 2 — full RAG + LLM mode (loading model now, timing it)")
print(SEP)
t0 = time.time()
bot = AgriChatbot(use_llm=True)
_ = bot.llm  # force load
load_s = time.time() - t0
print(f"\nModel load time: {load_s:.1f}s")

for label, q in [("EN tp_001", EN_PROMPT), ("YO tp_002", YO_PROMPT)]:
    print(f"\n[{label}] {q}")
    t0 = time.time()
    res = bot.ask(q)
    dt = time.time() - t0
    print(f"  wall time         : {dt:.1f}s")
    show(res)

print("\n" + SEP)
print("PART 3 — RAW Yoruba probe: force the LLM to answer tp_002 IN Yoruba")
print("(bypasses the English fallback, so we hear Qwen's own Yoruba)")
print(SEP)
llm = bot.llm
# Ground with the same English fact the pipeline retrieves for tp_002 (cassava harvest).
grounding = ("Cassava can be harvested from 7 months after planting, though most "
             "varieties reach optimum weight and starch content at 18 months.")
sys_prompt_yo = (
    "You are an agricultural advisor helping farmers in Nigeria. "
    "Answer the farmer's question using ONLY the reference information below. "
    "Respond in Yoruba, in 2-4 short sentences.\n\n"
    f"Reference information:\n{grounding}"
)
t0 = time.time()
raw_yo = llm.chat(sys_prompt_yo, YO_PROMPT)
dt = time.time() - t0
print(f"\n  wall time: {dt:.1f}s")
print(f"  RAW YORUBA OUTPUT:\n  {raw_yo}")

print("\n" + SEP)
print("PART 4 — tokens/sec benchmark (direct call, capturing usage)")
print(SEP)
t0 = time.time()
resp = llm.llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are an agricultural advisor for Nigerian farmers. Answer in English, 3-4 sentences."},
        {"role": "user", "content": "Explain in simple terms how to control Fall Armyworm in maize."},
    ],
    max_tokens=200,
    temperature=0.3,
)
dt = time.time() - t0
usage = resp.get("usage", {})
ct = usage.get("completion_tokens", 0)
pt = usage.get("prompt_tokens", 0)
print(f"\n  prompt_tokens     : {pt}")
print(f"  completion_tokens : {ct}")
print(f"  wall time         : {dt:.2f}s")
if ct and dt:
    print(f"  tokens/sec (gen)  : {ct / dt:.2f}")
print(f"  text: {resp['choices'][0]['message']['content'].strip()}")

print("\n" + SEP)
print("DONE")
print(SEP)
