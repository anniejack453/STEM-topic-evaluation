import json

# Read ISBNs from txt file
isbns = set()
with open('books_read_by_youth.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            isbns.add(line)

# Track matched ISBNs
matched_isbns = set()

# Filter books from JSONL and write to new file
output_file = 'youth_books_with_subjects.jsonl'
matched_count = 0

with open('books_with_subjects_complete.jsonl', 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        book = json.loads(line)
        
        if book.get('ISBN') in isbns:
            matched_isbns.add(book.get('ISBN'))
            # Create new record with only desired sections
            filtered_book = {
                'ISBN': book.get('ISBN'),
                'Book-Title': book.get('Book-Title'),
                'Book-Author': book.get('Book-Author'),
                'description': book.get('description'),
                'LoC_subjects': book.get('LoC_subjects'),
                'Google_categories': book.get('Google_categories')
            }
            outfile.write(json.dumps(filtered_book) + '\n')
            matched_count += 1

# Find unmatched ISBNs
unmatched_isbns = isbns - matched_isbns

print(f"Matched {matched_count} books from books_with_subjects_complete.jsonl")
print(f"Total ISBNs searched: {len(isbns)}")
print(f"Unmatched ISBNs: {len(unmatched_isbns)}")

# Search unmatched ISBNs in stem_books.jsonl
stem_matched_count = 0
with open('stem_books.jsonl', 'r', encoding='utf-8') as infile:
    for line in infile:
        book = json.loads(line)
        
        if book.get('ISBN') in unmatched_isbns:
            matched_isbns.add(book.get('ISBN'))
            # Create new record with only desired sections
            filtered_book = {
                'ISBN': book.get('ISBN'),
                'Book-Title': book.get('Book-Title'),
                'Book-Author': book.get('Book-Author'),
                'description': book.get('description'),
                'LoC_subjects': book.get('LoC_subjects'),
                'Google_categories': book.get('Google_categories')
            }
            with open(output_file, 'a', encoding='utf-8') as outfile:
                outfile.write(json.dumps(filtered_book) + '\n')
            stem_matched_count += 1

print(f"Matched {stem_matched_count} additional books from stem_books.jsonl")
print(f"Total matched: {matched_count + stem_matched_count}")

# Final unmatched ISBNs
final_unmatched = isbns - matched_isbns
print(f"Final unmatched ISBNs: {len(final_unmatched)}")