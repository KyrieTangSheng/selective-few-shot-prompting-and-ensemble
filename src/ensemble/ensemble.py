from typing import List, Dict, Optional, Tuple
import numpy as np
from .member import EnsembleMember, MemberPrediction
from .config import EnsembleConfig
from database.vector_store import SearchResult

class EnsemblePredictor:
    def __init__(
        self,
        members: List[EnsembleMember],
        config: EnsembleConfig
    ):
        self.members = members
        self.config = config
        self.prediction_history = []
    
    def predict(
        self,
        review: str,
        search_results: SearchResult,
        true_label: Optional[int] = None
    ) -> MemberPrediction:
        member_predictions = []
        for member in self.members:
            pred = member.predict(review, search_results, true_label)
            member_predictions.append({
                'member_name': member.name,
                'prediction': pred.prediction,
                'confidence': pred.confidence,
                'weight': member.weight
            })
        
        votes = np.zeros(2)  # Binary classification (0 or 1)
        for pred in member_predictions:
            weight = pred['weight'] * pred['confidence']
            votes[pred['prediction']] += weight
        
        winner = int(np.argmax(votes))
        confidence = float(votes[winner] / np.sum(votes))
        
        if true_label is not None:
            self.prediction_history.append({
                'prediction': winner,
                'confidence': confidence,
                'true_label': true_label,
                'member_predictions': member_predictions
            })
        
        return MemberPrediction(
            prediction=winner,
            confidence=confidence,
            metadata={'member_predictions': member_predictions}
        )
    
    def _weighted_vote(self, predictions: List[Dict]) -> Tuple[int, float]:
        votes = np.zeros(2)
        for pred in predictions:
            weight = pred['weight'] * pred['confidence']
            votes[pred['prediction']] += weight
        
        winner = int(np.argmax(votes))
        confidence = float(votes[winner] / np.sum(votes))
        return winner, confidence
    
    def get_performance_stats(self) -> Dict[str, any]:
        stats = {
            'overall_accuracy': 0.0,
            'member_stats': {},
            'predictions_made': len(self.prediction_history)
        }
        
        correct_predictions = sum(
            1 for p in self.prediction_history
            if p['true_label'] is not None and p['prediction'] == p['true_label']
        )
        if stats['predictions_made'] > 0:
            stats['overall_accuracy'] = correct_predictions / stats['predictions_made']
        
        for member in self.members:
            stats['member_stats'][member.name] = member.get_performance_stats()
        
        return stats