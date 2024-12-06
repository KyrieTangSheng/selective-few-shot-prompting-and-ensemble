import faiss
import numpy as np
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class SearchResult:
    distances: np.ndarray
    indices: np.ndarray
    data: List[Dict]

class VectorStore:
    def __init__(self, dimension: int, metric: str = 'l2'):
        self.dimension = dimension
        
        if metric == 'l2':
            self.index = faiss.IndexFlatL2(dimension)
        elif metric == 'ip':
            self.index = faiss.IndexFlatIP(dimension)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
            
        self.id_to_data = {}
    
    def add_vectors(self, vectors: np.ndarray, original_data: List[Dict]) -> None:
        vectors = vectors.astype(np.float32)
        
        if len(vectors) != len(original_data):
            raise ValueError("Number of vectors and data entries must match")
            
        start_id = len(self.id_to_data)
        self.index.add(vectors)
        
        for i, data in enumerate(original_data):
            self.id_to_data[start_id + i] = data
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> SearchResult:
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
            
        query_vector = query_vector.astype(np.float32)
        
        distances, indices = self.index.search(query_vector, k)
        
        data = self.get_data_by_ids(indices[0])
        
        return SearchResult(distances, indices, data)
    
    def get_data_by_ids(self, ids: List[int]) -> List[Dict]:
        return [self.id_to_data[id] for id in ids]