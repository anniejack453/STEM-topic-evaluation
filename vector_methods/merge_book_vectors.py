import json
import os
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
book_vectors_path = os.path.join(BASE_DIR, "data", "book_vectors.jsonl")
glove_vectors_path = os.path.join(BASE_DIR, "data", "book_glove_vectors.jsonl")
output_path = os.path.join(BASE_DIR, "data", "book_vectors_combined.jsonl")

# Load GloVe vectors into a dict (key: ISBN)
print("Loading GloVe vectors...")
glove_dict = {}
with open(glove_vectors_path, 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        record = json.loads(line)
        isbn = record.get("ISBN")
        if isbn:
            glove_dict[isbn] = record.get("glove")

# Merge with book_vectors
print("Merging vectors...")
with open(book_vectors_path, 'r', encoding='utf-8') as infile, \
     open(output_path, 'w', encoding='utf-8') as outfile:
    
    for line in tqdm(infile):
        record = json.loads(line)
        isbn = record.get("isbn")
        
        # Add GloVe vector if available
        if isbn in glove_dict:
            record["glove"] = glove_dict[isbn]
        
        outfile.write(json.dumps(record) + "\n")

print(f"Combined vectors saved to: {output_path}")