import json
import os
from tqdm import tqdm
from vector_methods.glove_vector_maker import GloVeVectorMaker

# Path to your pre-trained GloVe embeddings file
GLOVE_PATH = "data/wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt"  # Update this path

# Initialize GloVe vectorizer
glove = GloVeVectorMaker(GLOVE_PATH)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(BASE_DIR, "data", "youth_books_with_subjects.jsonl")
output_path = os.path.join(BASE_DIR, "data", "book_glove_vectors.jsonl")

print(f"Loading books from: {input_path}")
print(f"GloVe embedding dimension: {glove.dim}")

with open(input_path, "r", encoding="utf-8") as infile, \
     open(output_path, "w", encoding="utf-8") as outfile:
    
    for line in tqdm(infile):
        record = json.loads(line)
        isbn = record.get("ISBN")
        description = record.get("description", "")
        
        # Generate GloVe vector for the description
        glove_vector = glove.text_to_vector(description).tolist()
        
        # Write ISBN and vector to output file
        output_record = {
            "ISBN": isbn,
            "glove": glove_vector
        }
        outfile.write(json.dumps(output_record) + "\n")

print(f"GloVe vectors saved to: {output_path}")