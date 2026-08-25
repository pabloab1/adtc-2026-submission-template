"""
Sanity check for the retrieval wiring: for every question in the dataset,
in every active language, query the index with that exact text and check
it retrieves its own pair as the #1 result.

This does NOT prove the chatbot handles paraphrased/messy real-world
queries well — it only proves the wiring (loading, indexing, scoring,
answer lookup, language fallback) works end to end. Treat a 100% score
here as "the pipe isn't broken," not "the chatbot is smart."
"""

from collections import defaultdict
from retrieval import RetrievalIndex


def run_self_retrieval_test():
    index = RetrievalIndex()

    results_by_lang = defaultdict(lambda: {"correct": 0, "total": 0})
    failures = []

    for entry in index.entries:
        matches = index.search(entry.question_text, top_k=1)
        top = matches[0]
        results_by_lang[entry.language]["total"] += 1
        if top["pair_id"] == entry.pair_id:
            results_by_lang[entry.language]["correct"] += 1
        else:
            failures.append(
                {
                    "language": entry.language,
                    "asked": entry.question_text,
                    "expected_pair_id": entry.pair_id,
                    "got_pair_id": top["pair_id"],
                    "got_question": top["question_text"],
                    "score": round(top["score"], 3),
                }
            )

    print("=== Self-retrieval test ===\n")
    for lang, stats in results_by_lang.items():
        pct = 100 * stats["correct"] / stats["total"]
        lang_name = index.config["active_languages"].get(lang, lang)
        print(f"{lang_name} ({lang}): {stats['correct']}/{stats['total']} correct ({pct:.0f}%)")

    if failures:
        print(f"\n{len(failures)} failure(s):")
        for f in failures:
            print(f"  [{f['language']}] asked: {f['asked']!r}")
            print(f"    expected pair {f['expected_pair_id']}, got pair {f['got_pair_id']} "
                  f"({f['got_question']!r}, score={f['score']})")
    else:
        print("\nAll questions correctly retrieved their own pair.")

    print(f"\nTotal indexed entries: {len(index.entries)}")
    return len(failures) == 0


if __name__ == "__main__":
    ok = run_self_retrieval_test()
    exit(0 if ok else 1)
