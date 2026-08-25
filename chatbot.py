import os
import re
import json
import difflib
from typing import List, Dict, Any, Optional, Set
from pathlib import Path

# Internal imports
from chunk_retriever import retrieve
from rag_llm import LocalLLM

LOW_CONFIDENCE_WARNING = "THE DATA BASE DID NOT SPECIFICALLY SAY THIS SO BE WARNED OF MISINFORMATION"

# Canonical crops from dataset
CROPS = ["beans", "cassava", "cocoa", "ginger", "groundnut", "maize", "millet", "rice", "tomato", "yam"]
# Tools found in the Farm Tools section of the DOCX
SUPPORTED_TOOLS = [
    "tractor", "combine harvester", "motorized sprayer", "walk-behind tractor",
    "hoe", "shovel", "spade", "pickaxe", "mattock", "cutlass", "machete",
    "garden fork", "pitchfork", "pruning shears", "wheelbarrow"
]

class QueryProcessor:
    @staticmethod
    def normalize(text: str) -> str:
        t = text.lower().strip()
        t = re.sub(r'[^a-z0-9\s]', '', t)
        # Leet-speak
        leet = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t"}
        for k, v in leet.items(): t = t.replace(k, v)
        # Collapse repeats for matching
        return re.sub(r'([a-z])\1+', r'\1', t)

    @staticmethod
    def extract_crop(query: str) -> Optional[str]:
        norm = QueryProcessor.normalize(query)
        for crop in CROPS:
            # Direct match
            if crop in norm or crop[:-1] in norm: # handle plural
                return crop
        # Fuzzy match
        tokens = norm.split()
        for token in tokens:
            if len(token) > 3:
                matches = difflib.get_close_matches(token, CROPS, n=1, cutoff=0.7)
                if matches: return matches[0]
        return None

    @staticmethod
    def extract_tools(query: str) -> Set[str]:
        norm = QueryProcessor.normalize(query)
        found = set()
        for tool in SUPPORTED_TOOLS:
            if tool in norm: found.add(tool)
        return found

class AgriChatbot:
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.llm = LocalLLM() if use_llm else None
        self.history = []
        self.active_crop = None

    def ask(self, query: str, stream: bool = False) -> Dict[str, Any]:
        # 1. Gibberish check
        clean_q = re.sub(r'[^a-z]', '', query.lower())
        if not query.strip() or len(clean_q) < 4 or all(c in "asdfghjklqwertyuiopzxcvbnm"[:5] for c in clean_q[:4]):
            if not QueryProcessor.extract_crop(query) and not QueryProcessor.extract_tools(query):
                return self._format_refusal("I didn't get that, please retype it.")

        # 2. Entity Extraction
        detected_crop = QueryProcessor.extract_crop(query)
        detected_tools = QueryProcessor.extract_tools(query)
        
        # 3. Contextual Carryover
        if not detected_crop and self.active_crop:
            # Check if query is a follow-up
            pronouns = ["it", "them", "these", "those", "that", "its", "storage", "planting", "pests"]
            if any(p in query.lower().split() for p in pronouns) or len(query.split()) < 5:
                detected_crop = self.active_crop

        # 4. Retrieval with Crop Filter
        # Explicit refusal for known out-of-domain crops
        out_of_domain = ["wheat", "banana", "orange", "pineapple", "livestock", "chicken", "pig", "cow"]
        if any(ood in query.lower() for ood in out_of_domain):
            return self._format_refusal("I don't have specific information on that in my database.", score=0.0)

        chunks = retrieve(query, top_k=5, crop_filter=detected_crop)
        
        # 5. Confidence Policy
        if not chunks:
            return self._format_refusal("I don't have specific information on that in my database.", score=0.0)
        
        best_score = chunks[0]["score"]
        is_low = best_score < 0.45
        
        # Update active crop if we are confident
        if not is_low and chunks[0]["crop"].lower() in CROPS:
            self.active_crop = chunks[0]["crop"].lower()

        # 6. Response Generation
        if not self.use_llm:
            answer = f"[{chunks[0]['crop']} - {chunks[0]['section']}]: {chunks[0]['text']}"
            res = self._format_response(answer, best_score, is_low, chunks[0]["source"])
            self._update_history(query, res["answer"])
            return res

        # LLM Prompt Construction
        context = "\n\n".join([f"[{c['crop']} - {c['section']}]: {c['text']}" for c in chunks])
        
        # Explicit crop grounding in prompt
        system_instruction = (
            "You are a Senior Agricultural Extension Officer in Nigeria. "
            f"The user is asking about {detected_crop if detected_crop else 'agriculture'}. "
            "Answer ONLY using the provided context. If the context is about a different crop than the user asked, "
            "politely state you only have information on the crops in the context. "
            "NEVER invent facts or mix crop details."
        )

        user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {query}"
        
        # Only keep last 2 turns of history to prevent leakage/drift
        messages = [{"role": "system", "content": system_instruction}]
        messages.extend(self.history[-4:])
        messages.append({"role": "user", "content": user_prompt})

        if stream:
            return {
                "is_stream": True,
                "context": context,
                "confidence": best_score,
                "low_confidence": is_low,
                "source": chunks[0]["source"],
                "llm_generated": True,
                "detected_crop": detected_crop
            }

        answer = self.llm.chat(system_instruction, user_prompt, messages=messages)
        res = self._format_response(answer, best_score, is_low, chunks[0]["source"])
        self._update_history(query, res["answer"])
        return res

    def _format_response(self, answer: str, score: float, is_low: bool, source: str) -> Dict[str, Any]:
        final_answer = answer
        if is_low and LOW_CONFIDENCE_WARNING not in final_answer:
            final_answer = f"{final_answer}\n\n{LOW_CONFIDENCE_WARNING}"
        return {
            "answer": final_answer,
            "confidence": score,
            "low_confidence": is_low,
            "source": source,
            "llm_generated": self.use_llm,
            "is_stream": False
        }

    def _format_refusal(self, msg: str, score: float = 0.0) -> Dict[str, Any]:
        return self._format_response(msg, score, True, "System")

    def _update_history(self, q: str, a: str):
        self.history.append({"role": "user", "content": q})
        self.history.append({"role": "assistant", "content": a})

if __name__ == "__main__":
    bot = AgriChatbot(use_llm=False)
    while True:
        try:
            q = input("\nUser: ")
            if q.lower() in ["exit", "quit"]: break
            res = bot.ask(q)
            if res["low_confidence"]: print(f"\033[91m{LOW_CONFIDENCE_WARNING}\033[0m")
            print(f"Bot: {res['answer']}\n(Score: {res['confidence']})")
        except KeyboardInterrupt: break
