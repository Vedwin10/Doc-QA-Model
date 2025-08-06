from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Union, List
import torch

class EmbeddingGenerator:

    MODEL_MAP = {
        "miniLM": "sentence-transformers/all-MiniLM-L6-v2",
        "gte-base": "thenlper/gte-base",
        "mpnet": "sentence-transformers/all-mpnet-base-v2"    # note: cosine similarity tests found that mpnet is best
    }
    
    def __init__(self, model_name: str = "mpnet", device: str = "auto"):
      if model_name not in self.MODEL_MAP:
        raise ValueError(f"Invalid model name. Choose from: {list(self.MODEL_MAP.keys())}")
      
      self.model_name = model_name
      self.model = self._load_model(device)
        
    # function to load in selected model
    def _load_model(self, device: str) -> SentenceTransformer:
      # use cpu vs gpu
      # change in google colab via runtime type
      if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
      
      return SentenceTransformer(
        self.MODEL_MAP[self.model_name],
        device=device
      )
    
    # function that embeds the chunks using the chosen embedding model
    def embed_text(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
      if isinstance(texts, str):
        texts = [texts]
          
      return self.model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
      )
    
    # for displaying specs of the model used; mini-LM has 384 dimensions, the other 2 have 768
    @property
    def embedding_size(self) -> int:
      return self.model.get_sentence_embedding_dimension()
    
    def get_model_info(self) -> dict:
      return {
        "name": self.model_name,
        "dimensions": self.embedding_size,
        "max_sequence_length": self.model.max_seq_length,
        "device": str(self.model.device)
      }