"""Embedder module for transforming text into vector embeddings."""
from sentence_transformers import SentenceTransformer
from typing import List
from config import settings

class Embedder:
    """Handles text embedding using sentence-transformers."""
    
    def __init__(self, model_name: str = settings.embedding_model, prefix: str = settings.embedding_prefix):
        """Initializes the Embedder with the specified model."""
        self.prefix = prefix
        # Load the model; this will download it if not already cached
        self.model = SentenceTransformer(model_name)
        
    def embed(self, text: str) -> List[float]:
        """Generates a dense vector embedding for the given text."""
        # For e5 models, queries often need a specific prefix
        full_text = f"{self.prefix}{text}"
        embedding = self.model.encode(full_text, convert_to_numpy=True)
        return embedding.tolist()
