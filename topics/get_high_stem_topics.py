import json
from collections import defaultdict

# Load STEM topics and lowercase them
stem_topics = []
with open('data/STEM_topics.txt', 'r', encoding='utf-8') as f:
    stem_topics = [line.strip().lower() for line in f if line.strip()]

# Create a set for faster lookup
stem_topics_set = set(stem_topics)

# Count books for each STEM topic
topic_book_counts = defaultdict(int)

# Load and process books in one pass
with open('data/books_with_subjects_complete.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        book = json.loads(line)
        loc_subjects = str(book.get('LoC_subjects', '')).lower()
        google_categories = str(book.get('Google_categories', '')).lower()
        combined = loc_subjects + ' ' + google_categories
        
        # Check which topics appear in this book
        for topic in stem_topics_set:
            if topic in combined:
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

with open('significant_stem_topics.jsonl', 'w', encoding='utf-8') as f:
    for topic, count in sorted_topics:
        json.dump({'topic': topic, 'book_count': count}, f)
        f.write('\n')

print(f"\nResults saved to 'significant_stem_topics.jsonl'")