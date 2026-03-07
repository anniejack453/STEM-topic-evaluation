import json
import os
from tqdm import tqdm
from vector_methods.glove_vector_maker import G

# Paths
JSONL_PATH = "processed_data/books_with_subjects_complete.jsonl"
YOUTH_ISBN_PATH = "data_exploring/books_read_by_youth.txt"

def get_descriptions():
    with open(YOUTH_ISBN_PATH, "r", encoding="utf-8") as f:
        youth_isbns = {line.strip() for line in f if line.strip()}

    descriptions = []

    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            book = json.loads(line)
            isbn = book.get("ISBN")

            if isbn in youth_isbns:
                description = book.get("description")
                if description:  # avoid None or empty
                    descriptions.append((isbn, description))
    
    return descriptions

book_descriptions = get_descriptions()


print("Vector Makers all made")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(
    BASE_DIR, "processed_data", "book_vectors.jsonl"
)

with open(output_path, "w", encoding="utf-8") as outfile:
    i = 0
    for isbn, description in tqdm(book_descriptions):
        glove_vector = glove.text_to_vector(description)
        book = {
            "isbn" : isbn,
            "emotion_intensity" : emo_intensity_vec,
            "emotion" : emo_vec,
            "empath" : empath_vec,
            "tf_idf" : tf_idf_vec.tolist()
        }
        outfile.write(json.dumps(book) + "\n")
