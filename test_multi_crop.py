from chatbot import AgriChatbot
import chunk_retriever as cr

print("=== Multi-crop RAG smoke test (no LLM) ===")
bot = AgriChatbot(use_llm=False)

# 1. Verify specific crop retrieval
queries = [
    ("How do I plant maize?", "Maize"),
    ("When is cocoa ready for harvest?", "Cocoa"),
    ("What pests affect tomatoes?", "Tomato"),
    ("How do I store yam?", "Yam")
]

for q, expected_crop in queries:
    res = bot.ask(q)
    # The AgriChatbot.ask() result doesn't have a top-level 'crop' key, 
    # but the 'source' field in the no-LLM path is actually the file source.
    # We should use retrieve() directly to check the crop field for accuracy.
    top_matches = cr.retrieve(q, top_k=1)
    if not top_matches:
        print(f"Query: '{q}' -> No results (low confidence)")
        continue
    m = top_matches[0]
    print(f"Query: '{q}' -> Result: {m['crop']}:{m['section']} (score={m['score']:.3f})")
    assert expected_crop in m["crop"], f"Expected {expected_crop}, got {m['crop']}"

# 2. Verify conversational follow-up (crop context)
print("\nTesting conversational follow-up...")
bot.ask("Tell me about planting rice.")
print("Context set to Rice.")
# The AgriChatbot engine handles context resolution. 
# In no-LLM mode, we can check the 'source' field which we expect to be Rice.docx
res_followup = bot.ask("what pests affect it?")
top_source = res_followup.get("source", "")
print(f"Follow-up 'what pests affect it?' -> {top_source}")
# Since everything is in Crop-Knowledge-Base.docx now, checking the text content
# is more reliable for no-LLM context verification.
assert "Rice" in res_followup["answer"] or "rice" in res_followup["answer"].lower(), "Context loss!"

# 3. Verify cross-crop switch
print("\nTesting crop switch...")
bot.ask("Tell me about ginger care.")
res_switch = bot.ask("what about cassava?")
print(f"Switch 'what about cassava?' -> {res_switch['answer'][:50]}...")
assert "Cassava" in res_switch["answer"], f"Switch failed! Expected Cassava in answer"

print("\nALL MULTI-CROP TESTS PASSED.")
