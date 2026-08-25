"""
Smoke test for the conversational RAG engine — no LLM needed.

Exercises:
  1. chunk index builds from dataset/*.docx (Rice sections)
  2. rice queries retrieve rice sections
  3. out-of-domain queries return nothing (below similarity floor)
  4. conversation history builds up correctly
  5. the full AgriChatbot.ask() no-LLM path returns a compatible dict
"""

import chunk_retriever as cr
from chatbot import AgriChatbot

print("=== Conversational RAG engine smoke test (no LLM) ===")

# 1. Index builds from dataset/
vecs, chunks = cr.load_index()
print(f"1. Chunk index: {len(chunks)} chunks from dataset/")
assert len(chunks) >= 9, f"expected at least 9 sections, got {len(chunks)}"

# 2. Rice queries retrieve rice chunks
q1 = "When is rice ready for harvest in Nigeria?"
m = cr.retrieve(q1, top_k=1)[0]
print(f"2. '{q1}' -> [{m['crop']} — {m['section']}] score={m['score']}")
assert m["crop"] == "Rice", "top result must be a Rice section"
res2 = cr.retrieve(q1, top_k=3)
assert any("Harvest" in r["section"] for r in res2), "Harvest must rank in top 3"
print(f"   top-3 sections: {[r['section'] for r in res2]}")

q2 = "what pests should I watch out for?"
res = cr.retrieve(q2, top_k=3)
parts = ["{}:{}(s={})".format(r["crop"], r["section"], r["score"]) for r in res]
print(f"3. '{q2}' -> {parts}")
assert len(res) > 0, "Expected at least one pest-related result"

# 3. Out-of-domain query returns nothing (below floor)
res_none = cr.retrieve("How do I write a business plan for a bank loan?", top_k=3)
print(f"4. out-of-domain query -> {len(res_none)} results (expect 0)")
assert len(res_none) == 0

# 4. Conversation history + ask() dict shape (no-LLM path)
bot = AgriChatbot(use_llm=False)
r1 = bot.ask("When is rice ready for harvest in Nigeria?")
assert not r1["llm_generated"]
assert r1["confidence"] > 0
print(f"5. ask() no-LLM: score={r1['confidence']}, source={r1['source'][:30]}")

# Follow-up context: history grows
assert len(bot.history) >= 2, f"expected history, got {len(bot.history)}"
print(f"6. Conversation history: {len(bot.history)//2} turns recorded")

# Out-of-domain question via ask()
r3 = bot.ask("How do I invest in the stock market?")
assert r3["low_confidence"] and r3["confidence"] == 0.0
print("7. Out-of-domain via ask(): low_confidence flag set, no guess")

print("\nALL SMOKE TESTS PASSED — conversational engine works without the model file.")
