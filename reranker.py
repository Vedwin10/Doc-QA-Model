from sentence_transformers import CrossEncoder
from typing import List, Dict
import torch

class Reranker:

  def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "auto"):
    self.model = CrossEncoder(model_name)

    if device == "auto":
      self.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
      self.device = device

    self.model.model.to(self.device)

  # reranking function
  def rerank(self, query: str, candidates: List[Dict], top_k: int = 3) -> List[Dict]:
    chunk_texts = [candidate["chunk"] for candidate in candidates]
    pairs = [(query, chunk_text) for chunk_text in chunk_texts]

    scores = self.model.predict(pairs)

    for i in range(len(candidates)):
      candidates[i]["relevance"] = float(scores[i])

    candidates.sort(key=lambda x: x["relevance"], reverse=True)

    return candidates[:top_k]