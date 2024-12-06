from typing import List, Dict, Union, Optional, Set
import numpy as np
from sklearn.cluster import KMeans
from dataclasses import dataclass
from database.vector_store import SearchResult

@dataclass
class SelectionResult:
    examples: List[Dict]
    distances: np.ndarray
    diversity_scores: Optional[np.ndarray] = None
    selection_scores: Optional[np.ndarray] = None

class BaseSelector:
    """Base class for all selectors"""
    def select(self, search_results, k=5) -> SelectionResult:
        raise NotImplementedError

class ExampleSelector(BaseSelector):
    valid_strategies: Set[str] = {'similarity', 'diversity', 'hybrid', 'cluster', 'zeroshot', 'fewshot'}
    
    def __init__(self, strategy: str = 'similarity', **kwargs):
        self.strategy = strategy
        self.params = kwargs
        self._validate_params()
        
    def _validate_params(self):
        if self.strategy not in self.valid_strategies:
            raise ValueError(f"Invalid strategy. Must be one of {self.valid_strategies}")

class BaselineSelector(ExampleSelector):
    """Selector for baseline strategies (zero-shot and few-shot)"""
    def __init__(self, strategy: str, **kwargs):
        self.strategy = strategy
        self.params = kwargs
        self._validate_params()
        
    def _validate_params(self):
        valid_strategies = {'zeroshot', 'fewshot'}
        if self.strategy not in valid_strategies:
            raise ValueError(f"Invalid baseline strategy. Must be one of {valid_strategies}")
    
    def select(self, search_results: SearchResult, k: int = 3) -> SelectionResult:
        """Select examples based on strategy."""
        if self.strategy == 'zero_shot':
            # Return empty selection for zero-shot
            return SelectionResult(
                examples=[],
                selection_scores=[],
                strategy=self.strategy
            )
        else:
            # For few-shot, take top k examples by similarity
            examples = search_results.examples[:k]
            scores = search_results.scores[:k]
            return SelectionResult(
                examples=examples,
                selection_scores=scores,
                strategy=self.strategy
            )

class ExampleSelector(BaseSelector):
    def __init__(
        self,
        strategy: str = 'hybrid',
        diversity_weight: float = 0.3,
        n_clusters: int = 3
    ):
        """
        Initialize example selector.
        
        Args:
            strategy: Selection strategy ('similarity', 'diversity', 'hybrid', 'cluster')
            diversity_weight: Weight for diversity in hybrid strategy (0-1)
            n_clusters: Number of clusters for cluster-based selection
        """
        self.strategy = strategy
        self.diversity_weight = diversity_weight
        self.n_clusters = n_clusters
        
        self._validate_params()
    
    def _validate_params(self):
        """Validate initialization parameters."""
        valid_strategies = {'similarity', 'diversity', 'hybrid', 'cluster', 'zeroshot', 'fewshot'}
        if self.strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy. Must be one of {valid_strategies}")
        
        if not 0 <= self.diversity_weight <= 1:
            raise ValueError("diversity_weight must be between 0 and 1")
    
    def select(
        self,
        search_result: SearchResult,
        k: int = 5
    ) -> SelectionResult:
        """
        Select examples based on the chosen strategy.
        
        Args:
            search_result: SearchResult from vector store
            k: Number of examples to select
        """
        if self.strategy == 'similarity':
            return self._similarity_based(search_result, k)
        elif self.strategy == 'diversity':
            return self._diversity_based(search_result, k)
        elif self.strategy == 'hybrid':
            return self._hybrid_selection(search_result, k)
        else:  # cluster
            return self._cluster_based(search_result, k)
    
    def _similarity_based(
        self,
        search_result: SearchResult,
        k: int
    ) -> SelectionResult:
        """Simple similarity-based selection."""
        # Take top k by distance
        indices = np.argsort(search_result.distances[0])[:k]
        
        return SelectionResult(
            examples=[search_result.data[i] for i in indices],
            distances=search_result.distances[0][indices]
        )
    
    def _calculate_diversity_scores(
        self,
        vectors: np.ndarray
    ) -> np.ndarray:
        """Calculate diversity scores based on average distance to other points."""
        # Compute pairwise distances
        n = len(vectors)
        diversity_scores = np.zeros(n)
        
        for i in range(n):
            distances = np.linalg.norm(vectors - vectors[i], axis=1)
            diversity_scores[i] = np.mean(distances)
            
        return diversity_scores
    
    def _diversity_based(
        self,
        search_result: SearchResult,
        k: int
    ) -> SelectionResult:
        """Select diverse examples."""
        # Calculate diversity scores
        diversity_scores = self._calculate_diversity_scores(search_result.indices)
        
        # Select top k diverse examples
        indices = np.argsort(diversity_scores)[-k:]
        
        return SelectionResult(
            examples=[search_result.data[i] for i in indices],
            distances=search_result.distances[0][indices],
            diversity_scores=diversity_scores[indices]
        )
    
    def _hybrid_selection(
        self,
        search_result: SearchResult,
        k: int
    ) -> SelectionResult:
        """Combine similarity and diversity scores."""
        # Limit k to the number of available examples
        k = min(k, len(search_result.indices))
        
        # Normalize distances (similarity scores)
        sim_scores = 1 - (search_result.distances[0] / np.max(search_result.distances[0]))
        
        # Get diversity scores
        diversity_scores = self._calculate_diversity_scores(search_result.indices)
        # Handle case where all diversity scores are the same
        max_diversity = np.max(diversity_scores)
        diversity_scores = diversity_scores / max_diversity if max_diversity != 0 else np.zeros_like(diversity_scores)
        
        # Combine scores
        combined_scores = (
            (1 - self.diversity_weight) * sim_scores +
            self.diversity_weight * diversity_scores
        )
        
        # Select top k by combined score
        indices = np.argsort(combined_scores)[-k:]
        
        return SelectionResult(
            examples=[search_result.data[i] for i in indices],
            distances=search_result.distances[0][indices],
            diversity_scores=diversity_scores[indices],
            selection_scores=combined_scores[indices]
        )
    
    def _cluster_based(
        self,
        search_result: SearchResult,
        k: int
    ) -> SelectionResult:
        """Select examples using clustering."""
        vectors = search_result.indices
        
        # Adjust n_clusters if needed
        n_clusters = min(self.n_clusters, k, len(vectors))
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(vectors)
        
        # Select closest examples to cluster centers
        selected_indices = []
        for i in range(n_clusters):
            cluster_points = np.where(clusters == i)[0]
            if len(cluster_points) > 0:
                # Find point closest to cluster center
                center = kmeans.cluster_centers_[i]
                distances = np.linalg.norm(vectors[cluster_points] - center, axis=1)
                closest_idx = cluster_points[np.argmin(distances)]
                selected_indices.append(closest_idx)
        
        # Fill remaining slots with similarity-based selection
        while len(selected_indices) < k:
            for i in np.argsort(search_result.distances[0]):
                if i not in selected_indices:
                    selected_indices.append(i)
                    break
        
        selected_indices = np.array(selected_indices[:k])
        
        return SelectionResult(
            examples=[search_result.data[i] for i in selected_indices],
            distances=search_result.distances[0][selected_indices]
        )