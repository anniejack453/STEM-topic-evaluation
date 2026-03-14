import argparse
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def normalize_topic_values(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def normalize_phrase(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_topic_components(topic_values):
    components = set()
    for raw in topic_values:
        lowered = normalize_phrase(raw)
        if not lowered:
            continue

        components.add(lowered)

        for part in lowered.split("--"):
            part = normalize_phrase(part)
            if part:
                components.add(part)

    return components


def load_top_topics(topics_file: Path, top_n: int = 50):
    topics = []
    with topics_file.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            topic = normalize_phrase(row.get("topic", ""))
            if topic:
                topics.append(topic)
            if len(topics) >= top_n:
                break
    return topics


def load_book_vectors(vectors_file: Path):
    vectors_by_isbn = {}
    with vectors_file.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            isbn = str(row.get("isbn", "")).strip()
            if not isbn:
                continue
            vectors_by_isbn[isbn] = {
                "emotion_intensity": row.get("emotion_intensity", {}),
                "emotion": row.get("emotion", {}),
            }
    return vectors_by_isbn


def build_output(vectors_by_isbn, subjects_file: Path, top_topics):
    results = []
    top_topics_set = set(top_topics)

    with subjects_file.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            isbn = str(row.get("ISBN", "")).strip()
            if not isbn or isbn not in vectors_by_isbn:
                continue

            loc_topics = normalize_topic_values(row.get("LoC_subjects"))
            google_topics = normalize_topic_values(row.get("Google_categories"))
            combined_topics = []
            seen = set()
            for topic in loc_topics + google_topics:
                if topic not in seen:
                    seen.add(topic)
                    combined_topics.append(topic)

            topic_components = extract_topic_components(combined_topics)
            if not (topic_components & top_topics_set):
                continue

            vector_data = vectors_by_isbn[isbn]
            results.append(
                {
                    "isbn": isbn,
                    "title": row.get("Book-Title", ""),
                    "emotion_intensity": vector_data["emotion_intensity"],
                    "emotion": vector_data["emotion"],
                    "topics": combined_topics,
                }
            )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Create a JSON file of vectorized books matching top STEM topics."
    )
    parser.add_argument(
        "--vectors",
        default="data/book_vectors.jsonl",
        help="Path to book vectors JSONL file.",
    )
    parser.add_argument(
        "--subjects",
        default="data/books_with_subjects_complete.jsonl",
        help="Path to books with subjects JSONL file.",
    )
    parser.add_argument(
        "--topics",
        default="topics/significant_stem_topics1.jsonl",
        help="Path to significant STEM topics JSONL file.",
    )
    parser.add_argument(
        "--output",
        default="topics/book_vectors_top50_stem_topics_all.json",
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of top STEM topics to include.",
    )
    args = parser.parse_args()

    vectors_file = Path(args.vectors)
    subjects_file = Path(args.subjects)
    topics_file = Path(args.topics)
    output_file = Path(args.output)

    if not vectors_file.is_absolute():
        vectors_file = PROJECT_ROOT / vectors_file
    if not subjects_file.is_absolute():
        subjects_file = PROJECT_ROOT / subjects_file
    if not topics_file.is_absolute():
        topics_file = PROJECT_ROOT / topics_file
    if not output_file.is_absolute():
        output_file = PROJECT_ROOT / output_file

    top_topics = load_top_topics(topics_file, top_n=args.top_n)
    vectors_by_isbn = load_book_vectors(vectors_file)
    results = build_output(vectors_by_isbn, subjects_file, top_topics)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    print(f"Top topics considered: {len(top_topics)}")
    print(f"Books in vectors file: {len(vectors_by_isbn)}")
    print(f"Matched books written: {len(results)}")
    print(f"Output file: {output_file}")


if __name__ == "__main__":
    main()
