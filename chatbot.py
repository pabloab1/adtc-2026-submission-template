import os
import sys
import json
import re
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

# Project internal imports
import chunk_retriever as cr
from rag_llm import LocalLLM as AgriLLM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HISTORY_TURNS_KEPT = 4
MAX_REPLY_TOKENS = 600
LOW_CONFIDENCE_THRESHOLD = 0.45  # Stricter floor to filter out vague matches
LOW_CONFIDENCE_WARNING = "THE DATA BASE DID NOT SPECIFICALLY SAY THIS SO BE WARNED OF MISINFORMATION"

SYSTEM_PROMPT = f"""You are an expert Agricultural Advisor for farmers in Nigeria.
Your task is to answer the user's question using ONLY the information in the provided CONTEXT.

RULES:
1. Use the CONTEXT to provide specific details on planting, spacing, pests, and tools.
2. If the user's crop (e.g., Yam, Beans, Millet) IS in the CONTEXT, provide the advice directly.
3. If the user asks about something NOT in the context (like wheat or livestock), politely say you don't have that information yet.
4. Do NOT say the information is missing if the crop name appears in the context blocks below.
"""

class AgriChatbot:
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.llm = AgriLLM() if use_llm else None
        self.history: List[Dict[str, str]] = []

    def ask(self, query: str, stream: bool = False) -> Dict[str, Any]:
        """
        Primary conversational RAG engine:
        1. Resolves context (handles follow-ups like 'what about pests?')
        2. Retrieves top-k chunks from the GGUF-embedded index
        3. Composes a conversational answer via LLM (or returns raw chunk)
        4. Applies a high-visibility warning if confidence is low.
        """
        # 1. Resolve search query context
        # Clean typos and leet-speak in the search query for better vector matching
        # e.g. "Riice harv3st" -> "rice harvest"
        search_query = query.lower()
        search_query = search_query.replace("0", "o").replace("1", "i").replace("3", "e").replace("4", "a").replace("5", "s").replace("7", "t").replace("!", "i").replace("@", "a")
        
        # Remove common "noise" words that get joined by typos
        noise_words = ["tellmeabout", "howtouse", "howtoplant", "whatis", "canuse", "whatiss", "tellmeab0ut", "h0wtopl4nt"]
        for noise in noise_words:
            search_query = search_query.replace(noise, " ")

        # Normalize double letters carefully (e.g. riice -> rice)
        search_query = re.sub(r'(.)\1+', r'\1', search_query)
        
        # Mapping for common typos to correct terms for retrieval
        typo_map = {
            "maze": "maize", "m4ze": "maize",
            "yap": "yam", "y4m": "yam",
            "caca": "cocoa", "c0c0a": "cocoa",
            "riice": "rice", "r1ce": "rice",
            "b3ans": "beans",
            "t0mato": "tomato",
            "c4ssava": "cassava", "casava": "cassava",
            "g1nger": "ginger",
            "m1llet": "millet", "m1let": "millet", "milet": "millet",
            "gr0undnut": "groundnut",
            "h0e": "hoe",
            "cutl4s": "cutlass", "cutlas": "cutlass",
            "tr4ctor": "tractor", "tr4ct0r": "tractor",
            "shov3l": "shovel",
            "r4ke": "rake",
            "spr4yer": "sprayer",
            "wheelbarr0w": "wheelbarrow",
            "wh33lbarr0w": "wheelbarrow"
        }
        
        for typo, correct in typo_map.items():
            if typo in search_query:
                search_query = search_query.replace(typo, correct)

        # Restore valid crop and tool double letters if they were collapsed
        # This ensures "millet" and "wheelbarrow" stay intact for the vector engine
        restore_entities = ["rice", "maize", "beans", "yam", "tomato", "cocoa", "cassava", "ginger", "millet", "groundnut", "wheelbarrow", "cutlass"]
        for entity in restore_entities:
            collapsed = re.sub(r'(.)\1+', r'\1', entity)
            if collapsed in search_query and entity not in search_query:
                search_query = search_query.replace(collapsed, entity)
            
        last_crop = None
        for turn in reversed(self.history):
            content = turn["content"].lower()
            # Look for the source crop in history
            for crop in ["Rice", "Maize", "Beans", "Yam", "Tomato", "Cocoa", "Cassava", "Ginger", "Millet", "Groundnut"]:
                if crop.lower() in content:
                    last_crop = crop
                    break
            if last_crop: break

        # Only prepend if the user didn't mention a new crop name
        current_mentions_crop = any(c.lower() in search_query for c in ["rice", "maize", "beans", "yam", "tomato", "cocoa", "cassava", "ginger", "millet", "groundnut"])
        if last_crop and not current_mentions_crop and (len(query.split()) < 5 or " it" in search_query or " that" in search_query):
            search_query = f"{last_crop} {search_query}"

        # 2. Retrieval - balanced depth to avoid context overflow
        # 5 chunks provides enough info while staying well within the 2048 context window
        chunks = cr.retrieve(search_query.strip(), top_k=5)
        if not chunks:
            return {
                "answer": f"I don't have information on that in my crop guides yet.\n\n{LOW_CONFIDENCE_WARNING}",
                "language": "en",
                "confidence": 0.0,
                "matched_question": "None",
                "source": "None",
                "low_confidence": True,
                "llm_generated": False,
                "cached": False,
                "alternate_matches": [],
            }

        # 3. Confidence verification
        best = chunks[0]
        score = best["score"]
        
        # Dynamic Confidence Guard
        _, all_chunks = cr.load_index()
        valid_crops = {c.crop.lower() for c in all_chunks}
        valid_tools = {c.section.lower() for c in all_chunks if c.crop.lower() == "farm tools"}
        
        # Handle common typos and leet-speak (e.g. harv3st -> harvest)
        query_clean = query.lower().replace(" ", "")
        query_clean = query_clean.replace("0", "o").replace("1", "i").replace("3", "e").replace("4", "a").replace("5", "s").replace("7", "t").replace("!", "i").replace("@", "a")
        
        # Normalize double letters
        query_normalized = re.sub(r'(.)\1+', r'\1', query_clean)
        
        # Check mentioned entities using normalized query
        user_entities = []
        for entity in valid_crops.union(valid_tools):
            ent_collapsed = re.sub(r'(.)\1+', r'\1', entity)
            # Normal match or collapsed match
            if entity in query_clean or entity in query_normalized or ent_collapsed in query_normalized:
                user_entities.append(entity)
            # Fuzzy typo match (e.g. maze -> maize)
            elif entity == "maize" and any(m in query_normalized for m in ["maze", "m4ze"]):
                user_entities.append(entity)
            elif entity == "yam" and any(y in query_normalized for y in ["yap", "y4m"]):
                user_entities.append(entity)
            elif entity == "cocoa" and any(c in query_normalized for c in ["caca", "c0c0a"]):
                user_entities.append(entity)
            elif entity == "cutlass" and any(c in query_normalized for c in ["cutl4s", "cutlas"]):
                user_entities.append(entity)
            elif entity == "rake" and any(r in query_normalized for r in ["r4ke"]):
                user_entities.append(entity)
            elif entity == "millet" and any(m in query_normalized for m in ["m1let", "milet"]):
                user_entities.append(entity)
            elif entity == "tractor" and any(t in query_normalized for t in ["tr4ctor", "trac", "tr4ct0r"]):
                user_entities.append(entity)
        
        if any(kw in query_clean or kw in query_normalized for kw in ["tool", "machine", "equipment", "implement"]):
            user_entities.append("farm tools")

        # Determine if low confidence
        is_low = score < LOW_CONFIDENCE_THRESHOLD
        
        if user_entities:
            # Check if ANY top-k chunk matches ANY user-mentioned entity
            matches_any = False
            is_tool_query = "farm tools" in user_entities or any(t in valid_tools for t in user_entities)
            
            # For short queries (like just "yam"), the score might be lower but it's a direct hit.
            # We are more lenient if the entity name is explicitly in the top chunk.
            for entity in user_entities:
                # Check top 3 for crops, all retrieved (5) for tools
                depth = 5 if is_tool_query else 3
                if any(entity in c["crop"].lower() or entity in c["section"].lower() for c in chunks[:depth]):
                    matches_any = True
                    break
            
            if not matches_any:
                # If no direct entity match in top chunks, be very strict
                if score < 0.60: 
                    is_low = True
            else:
                # DIRECT MATCH CASE:
                # If the user mentioned a valid crop/tool and it's in our top results, 
                # we should ALMOST ALWAYS trust it unless the score is extremely poor (< 0.25).
                is_low = score < 0.25
        else:
            # Trap for common out-of-domain topics
            out_of_domain = ["banana", "wheat", "orange", "pineapple", "chicken", "poultry", "livestock", "cow", "pig", "apple", "strawberry", "grape", "potato"]
            if any(other in query_clean for other in out_of_domain):
                is_low = True
            elif score < 0.55: # Strict floor for non-entity queries
                is_low = True

        if is_low:
            score = 0.0

        # 4. Mode routing
        if not self.use_llm:
            answer = best["text"]
            if is_low:
                answer = f"{answer}\n\n{LOW_CONFIDENCE_WARNING}"
            
            res = {
                "answer": answer,
                "language": "en",
                "confidence": score,
                "matched_question": f"[{best['crop']} — {best['section']}] {best['text'][:80]}",
                "source": best["source"],
                "low_confidence": is_low,
                "llm_generated": False,
                "cached": False,
                "alternate_matches": chunks[1:],
            }
        else:
            # LLM path
            context_block = "\n\n".join(f"[{c['crop']} — {c['section']}]\n{c['text']}" for c in chunks)
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages.extend(self.history[-HISTORY_TURNS_KEPT:])
            messages.append({
                "role": "user",
                "content": f"CONTEXT:\n{context_block}\n\nFARMER'S QUESTION: {query}",
            })

            if stream:
                # Return a generator for streaming
                stream_gen = self.llm.chat("", query, messages=messages, max_tokens=MAX_REPLY_TOKENS, temperature=0.4, stream=True)
                
                def response_generator():
                    full_answer = ""
                    for chunk in stream_gen:
                        delta = chunk["choices"][0]["delta"]
                        if "content" in delta:
                            text = delta["content"]
                            full_answer += text
                            yield text
                    
                    if is_low:
                        yield f"\n\n{LOW_CONFIDENCE_WARNING}"
                        full_answer += f"\n\n{LOW_CONFIDENCE_WARNING}"
                    
                    # Update history at the end of streaming
                    self.history.append({"role": "user", "content": query})
                    self.history.append({"role": "assistant", "content": full_answer})

                return {
                    "answer": response_generator(),
                    "language": "en",
                    "confidence": score,
                    "matched_question": f"[{best['crop']} — {best['section']}] {best['text'][:80]}",
                    "source": best["source"],
                    "low_confidence": is_low,
                    "llm_generated": True,
                    "cached": False,
                    "alternate_matches": chunks[1:],
                    "is_stream": True
                }

            answer = self.llm.chat("", query, messages=messages, max_tokens=MAX_REPLY_TOKENS, temperature=0.4)
            if is_low:
                answer = f"{answer}\n\n{LOW_CONFIDENCE_WARNING}"

            res = {
                "answer": answer,
                "language": "en",
                "confidence": score,
                "matched_question": f"[{best['crop']} — {best['section']}] {best['text'][:80]}",
                "source": best["source"],
                "low_confidence": is_low,
                "llm_generated": True,
                "cached": False,
                "alternate_matches": chunks[1:],
                "is_stream": False
            }

        # Update history (for non-streaming LLM path or retrieved path)
        if not res.get("is_stream"):
            self.history.append({"role": "user", "content": query})
            self.history.append({"role": "assistant", "content": res["answer"]})
        return res

def chat_loop():
    bot = AgriChatbot(use_llm=True)
    print("Agri chatbot ready [retrieval + LLM]. Conversational — follow-ups resolve against the crop just discussed.")
    print("Type a question (e.g. 'Tell me about rice planting') or 'exit' to quit.\n")
    
    while True:
        try:
            query = input("You: ").strip()
            if query.lower() in ["exit", "quit"]:
                break
            if not query:
                continue
            
            res = bot.ask(query, stream=True)
            
            # Formatting the terminal output
            print(f"Bot (en, {'LLM' if res['llm_generated'] else 'retrieved'}, score={res['confidence']}): ", end="", flush=True)
            
            full_answer = ""
            if res.get("is_stream"):
                for chunk in res["answer"]:
                    if LOW_CONFIDENCE_WARNING in chunk:
                        print(f"\033[1;31m{chunk}\033[0m", end="", flush=True)
                    else:
                        print(chunk, end="", flush=True)
                    full_answer += chunk
                print()
            else:
                display_answer = res['answer'].replace(LOW_CONFIDENCE_WARNING, "").strip()
                print(display_answer)
                if res['low_confidence']:
                    print(f"\033[1;31m{LOW_CONFIDENCE_WARNING}\033[0m")
                print()
                
            print(f"      source: {res['source']}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    chat_loop()
