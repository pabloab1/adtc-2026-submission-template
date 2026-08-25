"""
Retrieval wiring. Loads the QA dataset and the active-language config,
indexes every (pair, language) question, and answers a similarity search.

Nothing here hardcodes "English" or "Yoruba" — it reads whatever languages
are listed in config/languages.json and whatever languages appear in
data/qa_dataset.json. Add/remove a language in the config, and this file
does not need to change.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from embedder import TfidfEmbedder  # swap this import when you add a real embedder

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "qa_dataset.json"
CONFIG_PATH = BASE_DIR / "config" / "languages.json"


@dataclass
class IndexEntry:
    pair_id: int
    language: str
    question_text: str
    topic: str
    source: str


class RetrievalIndex:
    def __init__(self, data_path: Path = DATA_PATH, config_path: Path = CONFIG_PATH):
        self.data_path = data_path
        self.config_path = config_path
        self.embedder = TfidfEmbedder()  # <-- swap for APIEmbedder(...) later, nothing else changes
        self.entries: list[IndexEntry] = []
        self.pairs_by_id: dict[int, dict] = {}
        self.vectors = None
        self._load_and_build()

    def _load_and_build(self) -> None:
        with open(self.config_path, encoding="utf-8") as f:
            self.config = json.load(f)
        with open(self.data_path, encoding="utf-8") as f:
            dataset = json.load(f)

        active_langs = set(self.config["active_languages"].keys())

        self.entries = []
        self.pairs_by_id = {}
        for pair in dataset["pairs"]:
            self.pairs_by_id[pair["id"]] = pair
            for lang, question_text in pair["questions"].items():
                if lang not in active_langs:
                    continue  # config controls what's searchable, without editing the dataset
                if not question_text:
                    continue
                self.entries.append(
                    IndexEntry(
                        pair_id=pair["id"],
                        language=lang,
                        question_text=question_text,
                        topic=pair.get("topic", ""),
                        source=pair.get("source", ""),
                    )
                )

        if not self.entries:
            raise ValueError(
                "No indexable questions found. Check that config/languages.json's "
                "active_languages match the language codes used in data/qa_dataset.json."
            )

        all_texts = [e.question_text for e in self.entries]
        self.embedder.fit(all_texts)
        self.vectors = self.embedder.encode(all_texts)

    def reload(self) -> None:
        """Call after editing the dataset or config JSON files without restarting."""
        self._load_and_build()

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Returns up to top_k matches, each as:
        {pair_id, language, question_text, topic, source, score}
        sorted by descending similarity score.
        """
        query_vec = self.embedder.encode([query])
        scores = self.embedder.similarity(query_vec, self.vectors)

        ranked = sorted(
            zip(self.entries, scores), key=lambda pair: pair[1], reverse=True
        )[:top_k]

        return [
            {
                "pair_id": entry.pair_id,
                "language": entry.language,
                "question_text": entry.question_text,
                "topic": entry.topic,
                "source": entry.source,
                "score": float(score),
            }
            for entry, score in ranked
        ]

    def get_answer(self, pair_id: int, preferred_language: str) -> tuple[str, str]:
        """
        Returns (answer_text, language_actually_used).
        Falls back to config's default_language if the preferred language's
        answer isn't translated yet (i.e. is null in the dataset).
        """
        pair = self.pairs_by_id[pair_id]
        answers = pair.get("answers", {})

        if answers.get(preferred_language):
            return answers[preferred_language], preferred_language

        fallback = self.config["default_language"]
        if answers.get(fallback):
            return answers[fallback], fallback

        # last resort: any non-null answer available
        for lang, text in answers.items():
            if text:
                return text, lang

        return "(no answer available for this entry yet)", preferred_language
