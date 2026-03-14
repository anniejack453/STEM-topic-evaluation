import json
import re
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_FILE = Path(__file__).resolve().parent / 'significant_stem_topics_all.jsonl'
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[+#]+[a-z0-9]*)*")


def normalize_tokens(value):
    return tuple(TOKEN_PATTERN.findall(str(value).lower()))


def normalize_topic(value):
    return ' '.join(normalize_tokens(value))


def build_topic_trie(topics):
    trie = {}

    for topic in set(topics):
        tokens = normalize_tokens(topic)
        if not tokens:
            continue

        node = trie
        for token in tokens:
            node = node.setdefault(token, {})
        node.setdefault('_topics', set()).add(topic)

    return trie


def iter_values(value):
    if isinstance(value, list):
        return value
    if value:
        return [value]
    return []


def find_matching_topics(values, topic_trie):
    matches = set()

    for value in values:
        tokens = normalize_tokens(value)

        for start_index in range(len(tokens)):
            node = topic_trie.get(tokens[start_index])
            if not node:
                continue

            matches.update(node.get('_topics', ()))

            for token in tokens[start_index + 1:]:
                node = node.get(token)
                if not node:
                    break
                matches.update(node.get('_topics', ()))

    return matches

# Load STEM topics and lowercase them
stem_topics = []
with open(DATA_DIR / 'STEM_topics.txt', 'r', encoding='utf-8') as f:
    stem_topics = [line.strip().lower() for line in f if line.strip()]

normalized_topics = [normalize_topic(topic) for topic in stem_topics]
normalized_topics = [topic for topic in normalized_topics if topic]

# Build a trie so topics are matched as whole words/phrases, not substrings
topic_trie = build_topic_trie(normalized_topics)

# Count books for each STEM topic
topic_book_counts = defaultdict(int)

# Load and process books in one pass
with open(DATA_DIR / 'books_with_subjects_complete.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        book = json.loads(line)
        values_to_check = [
            *iter_values(book.get('LoC_subjects')),
            *iter_values(book.get('Google_categories')),
        ]

        for topic in find_matching_topics(values_to_check, topic_trie):
            topic_book_counts[topic] += 1

# Filter and sort
significant_topics = {
    topic: count 
    for topic, count in topic_book_counts.items() 
    if count >= 50
}

sorted_topics = sorted(significant_topics.items(), key=lambda x: x[1], reverse=True)

# Display and save
print(f"STEM topics with at least 50 books: {len(sorted_topics)}\n")
for topic, count in sorted_topics:
    print(f"{topic}: {count} books")

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for topic, count in sorted_topics:
        json.dump({'topic': topic, 'book_count': count}, f)
        f.write('\n')

print(f"\nResults saved to '{OUTPUT_FILE}'")