import re
import numpy as np


class GloVeVectorMaker:
    """
    Lightweight GloVe vectorizer:
    - loads pre-trained GloVe text file
    - converts text -> vector by averaging known word vectors
    """

    def __init__(self, glove_path: str, lowercase: bool = True):
        self.glove_path = glove_path
        self.lowercase = lowercase
        self.embeddings = {}
        self.dim = 0
        self._load_embeddings()

    def _load_embeddings(self) -> None:
        with open(self.glove_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split()
                if len(parts) < 2:
                    continue
                word = parts[0]
                try:
                    vec = np.asarray(parts[1:], dtype=np.float32)
                except (ValueError, TypeError):
                    # Skip lines with invalid numbers
                    continue
                if self.dim == 0:
                    self.dim = vec.shape[0]
                if vec.shape[0] != self.dim:
                    continue
                self.embeddings[word] = vec

        if self.dim == 0:
            raise ValueError(f"No valid embeddings found in: {self.glove_path}")

    def _tokenize(self, text: str):
        if self.lowercase:
            text = text.lower()
        return re.findall(r"[a-zA-Z0-9']+", text)

    def get_word_vector(self, word: str):
        key = word.lower() if self.lowercase else word
        return self.embeddings.get(key)

    def text_to_vector(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        vectors = [self.embeddings[t] for t in tokens if t in self.embeddings]
        if not vectors:
            return np.zeros(self.dim, dtype=np.float32)
        return np.mean(vectors, axis=0).astype(np.float32)

    def texts_to_matrix(self, texts):
        return np.vstack([self.text_to_vector(t) for t in texts])