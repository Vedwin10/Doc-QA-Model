import faiss
import pickle
from typing import Tuple, List, Dict
import numpy as np
import os

class VectorStore:

  def __init__(self, dimension: int, index_path: str = None):
    self.dimension = dimension
    self.index_path = index_path
    self.metadata_store = []

    self.index = faiss.IndexFlatIP(dimension)   # using inner product for index construction because it is faster and same as cosine similarity if vectors are normalized

    # if we want to store the index locally
    if index_path and os.path.exists(index_path):
      self.load_index(index_path)

    # gpu acceleration
    if faiss.get_num_gpus() > 0:
      res = faiss.StandardGpuResources()
      self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

  # function to add embedded chunks and metadata into vector store
  def add_chunks(self, embeddings: np.ndarray, chunks: List[Dict]):
    if len(embeddings) != len(chunks):
      raise ValueError("Mismatch between embeddings and chunks count")

    faiss.normalize_L2(embeddings)      # faster than cosine_similarity because of faiss's C++, SIMD, and GPU acceleration; cos(A, B) = A * B = IP for unit vectors

    self.index.add(embeddings)

    self.metadata_store.extend(chunks)

  # similarity search function
  def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
    if query_embedding.ndim == 1:
      query_embedding = np.expand_dims(query_embedding, 0)
    faiss.normalize_L2(query_embedding)

    distances, indices = self.index.search(query_embedding, k)

    results = []
    for i in range(len(indices[0])):
      idx = indices[0][i]
      if idx >= 0:
        results.append({
            "chunk": self.metadata_store[idx]['text'],
            "metadata": self.metadata_store[idx]['metadata'],
            "similarity": float(distances[0][i])
        })
    return results

  # function to save index and metadata to disk
  def save_index(self, path: str = None):
    path = path or self.index_path
    if not path:
      raise ValueError("No path specified for saving index")

    # handle gpu usage with index saving
    if 'Gpu' in type(self.index).__name__:
      cpu_index = faiss.index_gpu_to_cpu(self.index)
    else:
      cpu_index = self.index
    
    faiss.write_index(cpu_index, path + ".index")

    with open(path + ".meta", "wb") as f:
      pickle.dump(self.metadata_store, f)

    print(f"Index was loaded from {path}.index and {path}.meta")

  # function to load in index and metadata from disk
  def load_index(self, path: str = None):
    path = path or self.index_path
    if not path:
      raise ValueError("No path specified for loading index")

    # handle index loading with gpu
    cpu_index = faiss.read_index(path + ".index")
    if faiss.get_num_gpus() > 0:
      res = faiss.StandardGpuResources()
      self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    else:
      self.index = cpu_index
    
    self.index = faiss.read_index(path + ".index")

    with open(path + ".meta", "rb") as f:
      self.metadata_store = pickle.load(f)

    print(f"Index loaded from {path}.index and {path}.meta")

  def get_index_size(self) -> int:
    return self.index.ntotal