from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from typing import List, Union, Optional
from utils.device_utils import get_device

class TextEncoder:
    def __init__(
        self, 
        model_name: str = 'all-MiniLM-L6-v2',
        device: Optional[str] = None
    ):
        self.device = device if device else get_device()
        print(f"Using device: {self.device}")
        
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)
        
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def validate_input(self, texts: Union[str, List[str]]) -> List[str]:
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise ValueError("Input must be a string or list of strings")
        return texts
    
    @torch.no_grad()
    def encode(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        texts = self.validate_input(texts)
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                device=self.device
            )
            
            # Verify output shape
            expected_shape = (len(texts), self.embedding_dim)
            if embeddings.shape != expected_shape:
                raise ValueError(
                    f"Unexpected embedding shape. Expected {expected_shape}, got {embeddings.shape}"
                )
                
            return embeddings
            
        except Exception as e:
            raise RuntimeError(f"Error during encoding: {str(e)}")