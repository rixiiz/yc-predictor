import numpy as np
from sentence_transformers import SentenceTransformer

class TextEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str]) -> np.ndarray:
        X = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(X, dtype=np.float32)
