from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
from selection.selector import ExampleSelector, SelectionResult
from prompting.prompt_builder import PromptBuilder
from llm.interface import LLMInterface
from database.vector_store import SearchResult

@dataclass
class MemberPrediction:
    prediction: int
    confidence: float
    metadata: Dict[str, Any]

class EnsembleMember:
    def __init__(
        self,
        name: str,
        selector: ExampleSelector,
        prompt_builder: PromptBuilder,
        llm: LLMInterface,
        weight: float = 1.0
    ):
        self.name = name
        self.selector = selector
        self.prompt_builder = prompt_builder
        self.llm = llm
        self.weight = weight
        
        self.predictions = []
        self.accuracies = []
    
    def predict(
        self,
        review: str,
        search_results: SearchResult,
        true_label: Optional[int] = None
    ) -> MemberPrediction:
        """Make prediction for a single review."""
        selection_results = self.selector.select(search_results, k=3)
        
        template_name = self.name.split('_')[1]
        
        prompt_result = self.prompt_builder.build_prompt(
            selection_results.examples,
            review,
            template_name=template_name
        )
        
        prediction = self.llm.predict(prompt_result)
        
        if true_label is not None:
            accuracy = 1.0 if prediction['sentiment'] == true_label else 0.0
            self.accuracies.append(accuracy)
        
        return MemberPrediction(
            prediction=prediction['sentiment'],
            confidence=prediction['confidence'],
            metadata={
                'reasoning': prediction.get('reasoning'),
                'selection_scores': selection_results.selection_scores
            }
        )
    
    def _calculate_confidence(self, response: Any) -> float:
        """Calculate confidence score from response."""
        content = response.choices[0].message.content.lower()
        
        confidence_factors = []
        
        certainty_words = ['definitely', 'clearly', 'certainly', 'absolutely']
        uncertainty_words = ['maybe', 'perhaps', 'possibly', 'unclear']
        
        confidence = 0.7  # Base confidence
        
        if any(word in content for word in certainty_words):
            confidence += 0.2
        if any(word in content for word in uncertainty_words):
            confidence -= 0.2
            
        if len(content.split()) > 50:  
            confidence += 0.1
            
        return min(max(confidence, 0.0), 1.0)
    
    def get_performance_stats(self) -> Dict[str, float]:
        return {
            'accuracy': np.mean(self.accuracies) if self.accuracies else 0.0,
            'predictions_made': len(self.predictions),
            'weight': self.weight
        }